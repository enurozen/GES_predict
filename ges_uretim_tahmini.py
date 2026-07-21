"""
GES (Güneş Enerjisi Santrali) Üretim Tahmin Modeli
=====================================================
Hibrit yaklaşım: Fiziksel (astronomik + GHI tabanlı) baz model
                 + ML (Random Forest) ile kalıntı (residual) düzeltmesi

Mantık:
  1. Güneşin konumunu (zenith açısı) astronomik formüllerle KESİN olarak hesapla.
     (Bu, rüzgardaki gibi "belirsiz" değil - fizik/geometri.)
  2. GHI (Global Horizontal Irradiance) tahmininden teorik PV gücünü çıkar.
  3. Sıcaklık derating uygula (panel sıcakken verimi düşer).
  4. Geçmiş gerçek üretim ile bu fiziksel tahmin arasındaki farkı (residual)
     bir ML modeline öğret. Bu fark; soiling, gölgeleme, inverter clipping,
     curtailment gibi "formülle yazılamayan" kayıpları temsil eder.
  5. Nihai tahmin = Fiziksel tahmin + ML düzeltmesi

Girdi olarak beklenen geçmiş veri (pandas DataFrame), saatlik:
    timestamp        : datetime
    production_mwh    : gerçekleşen üretim (MWh) - GEÇMİŞ veri için gerekli
    ghi_forecast      : W/m^2 cinsinden ışınım tahmini (EPİAŞ/meteoroloji kaynağından)
    temp_c            : ortam sıcaklığı (°C)
    cloud_cover       : opsiyonel, 0-1 arası bulut kapanım oranı

Santral parametreleri:
    capacity_mw       : kurulu güç (MWp)
    lat, lon          : santral koordinatları
    tilt, azimuth     : panel açıları (derece) - opsiyonel, basitleştirilmiş modelde
                        düz (horizontal) kabul edip GHI'yi direkt kullanıyoruz
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ----------------------------------------------------------------------
# 1) FİZİKSEL BAZ MODEL: Astronomik hesaplar + basit PV dönüşümü
# ----------------------------------------------------------------------

def solar_position(timestamp: pd.Timestamp, lat: float, lon: float) -> tuple:
    """
    Güneşin zenith açısını (dikey açı, 0=tam üstte, 90=ufukta) hesaplar.
    Basitleştirilmiş astronomik formül (Cooper's equation + hour angle).
    Bu KESİN bir hesaptır, tahmin değil - rüzgardaki stokastik yapıdan farkı budur.
    """
    day_of_year = timestamp.dayofyear
    hour = timestamp.hour + timestamp.minute / 60.0

    # Güneş deklinasyonu (dünyanın eksen açısına bağlı mevsimsel kayma)
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

    # Saat açısı (yerel güneş öğlenine göre kayma); basitlik için lon düzeltmesi
    # olmadan, yerel saat dilimini yaklaşık kabul ediyoruz.
    hour_angle = 15 * (hour - 12)

    lat_r = np.radians(lat)
    dec_r = np.radians(declination)
    ha_r = np.radians(hour_angle)

    cos_zenith = (np.sin(lat_r) * np.sin(dec_r) +
                  np.cos(lat_r) * np.cos(dec_r) * np.cos(ha_r))
    cos_zenith = np.clip(cos_zenith, -1, 1)
    zenith_deg = np.degrees(np.arccos(cos_zenith))
    return zenith_deg, cos_zenith


def clearsky_ghi_estimate(cos_zenith: float, ghi_toa_max: float = 1000.0) -> float:
    """
    Çok basitleştirilmiş açık-gökyüzü (clear-sky) GHI tahmini.
    Gerçek projede pvlib.clearsky (Ineichen/Haurwitz modeli) kullanmak
    daha doğru olur; burada kavramı göstermek için basit tutuyoruz.
    """
    if cos_zenith <= 0:
        return 0.0
    return max(0.0, ghi_toa_max * cos_zenith)


def physical_pv_power(ghi_forecast: float, temp_c: float, capacity_mw: float,
                       ghi_clearsky: float, ref_temp: float = 25.0,
                       temp_coeff: float = -0.004,
                       efficiency_scale: float = 1.0) -> float:
    """
    GHI tahmininden ve sıcaklıktan teorik PV gücünü (MW) hesaplar.

    - GHI oranı: gerçek/beklenen ışınımın kurulu güce oranı (basit lineer model)
    - Sıcaklık derating: panel sıcaklığı arttıkça verim düşer
      (tipik silikon panel katsayısı ~ -0.4%/°C, 25°C referans)
    - efficiency_scale: nominal kapasiteye göre santralin gerçek etkin verimi
      (calibrate_site_parameters'tan gelir; datasheet yoksa varsayılan 1.0)
    """
    if ghi_forecast <= 0:
        return 0.0

    irradiance_ratio = ghi_forecast / 1000.0  # 1000 W/m^2 = STC referansı
    # Panel sıcaklığı ortam sıcaklığından biraz yüksek olur (basit yaklaşım: +25°C)
    panel_temp = temp_c + 25.0
    temp_derating = 1 + temp_coeff * (panel_temp - ref_temp)
    temp_derating = np.clip(temp_derating, 0.5, 1.05)

    effective_capacity = capacity_mw * efficiency_scale
    power_mw = effective_capacity * irradiance_ratio * temp_derating
    return max(0.0, power_mw)


def build_physical_baseline(df: pd.DataFrame, lat: float, lon: float,
                             capacity_mw: float, temp_coeff: float = -0.004,
                             efficiency_scale: float = 1.0) -> pd.Series:
    """Tüm zaman serisi için fiziksel baz tahmini üretir (MWh, saatlik ise MW=MWh)."""
    baseline = []
    for _, row in df.iterrows():
        zenith, cos_z = solar_position(row['timestamp'], lat, lon)
        ghi_cs = clearsky_ghi_estimate(cos_z)
        p = physical_pv_power(row['ghi_forecast'], row['temp_c'], capacity_mw, ghi_cs,
                               temp_coeff=temp_coeff, efficiency_scale=efficiency_scale)
        baseline.append(p)
    return pd.Series(baseline, index=df.index)


# ----------------------------------------------------------------------
# 1b) SANTRALE ÖZGÜ KALİBRASYON: Datasheet yerine geçmiş veriden parametre tahmini
# ----------------------------------------------------------------------

def calibrate_site_parameters(df_calib: pd.DataFrame, lat: float, lon: float,
                               nominal_capacity_mw: float) -> dict:
    """
    Panel markası/teknolojisi, invertör kapasitesi gibi datasheet bilgisi
    OLMADAN, sadece geçmiş (üretim, hava) verisinden santrale özgü fiziksel
    parametreleri tahmin eder (least-squares kalibrasyon).

    Bulunan parametreler:
      - efficiency_scale : nominal kapasiteye göre santralin "gerçek" etkin
                            verimi (panel teknolojisi + kurulum kaybı + kirlenme
                            ortalamasının bileşik etkisi)
      - temp_coeff        : santrale özgü sıcaklık katsayısı
      - ac_capacity_mw    : invertör tavanı (clipping) tahmini - gözlemlenen
                            üretimin üst yüzdelik dilimi

    NOT: Kalibrasyon verisi mümkünse farklı hava koşullarını (açık/bulutlu
    günler, farklı sıcaklıklar) kapsamalı - dar bir aralık overfit'e yol açar.
    En az 2-3 haftalık çeşitli veri önerilir.
    """
    from scipy.optimize import least_squares

    cos_zeniths = np.array([
        solar_position(ts, lat, lon)[1] for ts in df_calib['timestamp']
    ])
    ghi = df_calib['ghi_forecast'].values
    temp = df_calib['temp_c'].values
    actual = df_calib['production_mwh'].values

    # Gece saatlerini kalibrasyondan çıkar (zenith>90 -> gündüz yok, sinyal taşımaz)
    daylight_mask = cos_zeniths > 0.05
    ghi_d, temp_d, actual_d = ghi[daylight_mask], temp[daylight_mask], actual[daylight_mask]

    def residuals(params):
        efficiency_scale, temp_coeff = params
        panel_temp = temp_d + 25.0
        derating = np.clip(1 + temp_coeff * (panel_temp - 25.0), 0.5, 1.05)
        pred = nominal_capacity_mw * efficiency_scale * (ghi_d / 1000.0) * derating
        return np.clip(pred, 0, None) - actual_d

    result = least_squares(
        residuals, x0=[0.85, -0.004],
        bounds=([0.05, -0.05], [1.05, -0.0005])
    )
    efficiency_scale, temp_coeff = result.x

    # İnvertör tavanı: gözlemlenen en yüksek üretimlerin persentili
    # (tepe saatlerde sürekli aynı tavana çarpıyorsa bu clipping'in izidir)
    ac_capacity_estimate = float(np.percentile(actual_d, 99.5))

    return {
        "efficiency_scale": round(float(efficiency_scale), 4),
        "temp_coeff": round(float(temp_coeff), 5),
        "ac_capacity_mw": round(ac_capacity_estimate, 3),
        "n_daylight_samples": int(daylight_mask.sum()),
    }


# ----------------------------------------------------------------------
# 2) ML DÜZELTME KATMANI: Fiziksel modelin kaçırdığı saha kayıplarını öğren
# ----------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Residual modeli için özellik mühendisliği."""
    feats = pd.DataFrame(index=df.index)
    feats['hour'] = df['timestamp'].dt.hour
    feats['day_of_year'] = df['timestamp'].dt.dayofyear
    feats['month'] = df['timestamp'].dt.month
    feats['temp_c'] = df['temp_c']
    feats['ghi_forecast'] = df['ghi_forecast']
    if 'cloud_cover' in df.columns:
        feats['cloud_cover'] = df['cloud_cover']
    # Mevsimsellik için döngüsel kodlama (saat 23 ile saat 0 birbirine yakın olsun)
    feats['hour_sin'] = np.sin(2 * np.pi * feats['hour'] / 24)
    feats['hour_cos'] = np.cos(2 * np.pi * feats['hour'] / 24)
    feats['doy_sin'] = np.sin(2 * np.pi * feats['day_of_year'] / 365)
    feats['doy_cos'] = np.cos(2 * np.pi * feats['day_of_year'] / 365)
    # Sabah/akşam ayrımı: düşük GHI sabah (gün doğumu, üretim genelde var) ile
    # akşam (gün batımı, santral genelde o saatte üretimi kesiyor) arasında çok
    # farklı davranıyor - saat tek başına bunu ayırt ettiremiyor çünkü mevsime
    # göre kayıyor, bu yüzden GHI ile etkileşimli bir "öğleden sonra mı" bayrağı.
    feats['is_afternoon'] = (feats['hour'] > 12).astype(int)
    feats['ghi_x_afternoon'] = feats['ghi_forecast'] * feats['is_afternoon']
    return feats


def train_residual_model(df_train: pd.DataFrame, physical_baseline: pd.Series):
    """
    Gerçek üretim - fiziksel tahmin farkını (residual) öğrenen RF modeli.
    Neden residual öğreniyoruz (ham üretimi değil)?
    -> Fiziksel model zaten büyük varyansı (gece/gündüz, mevsim) açıklıyor.
       ML sadece kalan "saha kaybı" örüntüsünü öğrenir -> daha az veriyle
       daha kararlı bir model elde edilir.
    """
    X = build_features(df_train)
    residual = df_train['production_mwh'] - physical_baseline

    X_train, X_val, y_train, y_val = train_test_split(
        X, residual, test_size=0.2, shuffle=False  # zaman serisi: kronolojik böl
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
    )
    model.fit(X_train, y_train)

    val_pred_residual = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred_residual)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred_residual))
    print(f"[Residual model doğrulama] MAE={mae:.3f} MWh, RMSE={rmse:.3f} MWh")

    return model


def predict_production(df_new: pd.DataFrame, lat: float, lon: float,
                        capacity_mw: float, residual_model,
                        temp_coeff: float = -0.004, efficiency_scale: float = 1.0,
                        ac_capacity_mw: float | None = None) -> pd.Series:
    """Yeni (gelecek) veri için nihai üretim tahmini: fiziksel + ML düzeltme.

    ac_capacity_mw verilirse üst sınır olarak kullanılır (invertör/şebeke
    tavanı, nominal capacity_mw'den düşük olabilir - clipping'in nedeni budur);
    verilmezse capacity_mw'ye geri döner.
    """
    baseline = build_physical_baseline(df_new, lat, lon, capacity_mw,
                                        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale)
    X_new = build_features(df_new)
    residual_pred = residual_model.predict(X_new)

    final_pred = baseline + residual_pred
    upper_bound = ac_capacity_mw if ac_capacity_mw is not None else capacity_mw
    # Fiziksel sınırlar: negatif olamaz, şebeke/invertör tavanını aşamaz
    final_pred = final_pred.clip(lower=0, upper=upper_bound)
    return final_pred


# ----------------------------------------------------------------------
# 3) DEMO: Sentetik veriyle uçtan uca çalıştırma
#    (Gerçek EPİAŞ/saha verinle bu kısmı kendi CSV'ni okuyarak değiştir)
# ----------------------------------------------------------------------

def _generate_synthetic_data(n_days=60, capacity_mw=10.0, lat=39.9, lon=32.8):
    """Sadece demo amaçlı - gerçek projede bu fonksiyonu KULLANMA."""
    rng = pd.date_range("2025-05-01", periods=n_days * 24, freq="h")
    rows = []
    for ts in rng:
        zenith, cos_z = solar_position(ts, lat, lon)
        ghi_clear = clearsky_ghi_estimate(cos_z)
        cloud = np.clip(np.random.normal(0.25, 0.2), 0, 0.9)
        ghi_forecast = ghi_clear * (1 - cloud) + np.random.normal(0, 15)
        ghi_forecast = max(0, ghi_forecast)
        temp = 20 + 10 * np.sin(2 * np.pi * (ts.hour - 6) / 24) + np.random.normal(0, 1.5)

        p_theoretical = physical_pv_power(ghi_forecast, temp, capacity_mw, ghi_clear)
        # gerçek üretime saha kaybı ekleyelim (soiling + rastgele curtailment)
        saha_kaybi_orani = 0.92 - 0.03 * np.sin(2 * np.pi * ts.dayofyear / 365)
        actual = p_theoretical * saha_kaybi_orani + np.random.normal(0, 0.15)
        actual = max(0, actual)

        rows.append({
            "timestamp": ts, "ghi_forecast": ghi_forecast,
            "temp_c": temp, "cloud_cover": cloud, "production_mwh": actual
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    CAPACITY_MW = 10.0
    LAT, LON = 39.9, 32.8  # örnek: Ankara civarı

    print("Sentetik geçmiş veri üretiliyor (demo)...")
    data = _generate_synthetic_data(n_days=60, capacity_mw=CAPACITY_MW, lat=LAT, lon=LON)

    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].reset_index(drop=True)
    test_df = data.iloc[split_idx:].reset_index(drop=True)

    print("\nFiziksel baz model hesaplanıyor...")
    baseline_train = build_physical_baseline(train_df, LAT, LON, CAPACITY_MW)

    print("\nResidual (ML düzeltme) modeli eğitiliyor...")
    model = train_residual_model(train_df, baseline_train)

    print("\nTest seti için nihai tahmin üretiliyor...")
    final_predictions = predict_production(test_df, LAT, LON, CAPACITY_MW, model)

    test_mae = mean_absolute_error(test_df['production_mwh'], final_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_df['production_mwh'], final_predictions))
    naive_baseline_mae = mean_absolute_error(
        test_df['production_mwh'], build_physical_baseline(test_df, LAT, LON, CAPACITY_MW)
    )

    print(f"\n=== SONUÇLAR (test seti) ===")
    print(f"Sadece fiziksel model MAE : {naive_baseline_mae:.3f} MWh")
    print(f"Hibrit model (fiziksel+ML) MAE : {test_mae:.3f} MWh")
    print(f"Hibrit model RMSE : {test_rmse:.3f} MWh")
    print(f"\nİyileşme: %{(1 - test_mae/naive_baseline_mae)*100:.1f}")
