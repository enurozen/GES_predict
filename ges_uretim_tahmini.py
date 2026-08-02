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
    ghi_forecast      : W/m^2 cinsinden yatay düzlem ışınım (GHI) tahmini
    dni_forecast      : W/m^2 cinsinden doğrudan normal ışınım (DNI), opsiyonel
    dhi_forecast      : W/m^2 cinsinden yatay düzlem difüz ışınım (DHI), opsiyonel
                        (dni/dhi yoksa GHI'den Erbs ayrıştırmasıyla tahmin edilir)
    temp_c            : ortam sıcaklığı (°C)
    cloud_cover       : opsiyonel, 0-1 arası bulut kapanım oranı

Santral parametreleri:
    capacity_mw       : kurulu güç (MWp)
    lat, lon          : santral koordinatları
    tilt_deg          : panel eğim açısı (derece, 0=yatay) - bilinmiyorsa None
                        bırakılır ve düz-GHI (horizontal) baz modele dönülür
    panel_azimuth_deg : panelin baktığı yön (derece; Güney=0, Batı=+90, Doğu=-90 -
                        Duffie & Beckman kongvansiyonu)
    tracker_type      : None (sabit/bilinmiyor) | "single_axis_horizontal"
                        (yatay, Kuzey-Güney eksenli tek-eksen tracker)
    module_noct_c     : NOCT (Nominal Operating Cell Temperature), datasheet
                        yoksa endüstri-tipik varsayılan 45°C

tilt_deg/tracker_type verilmediği sürece (varsayılan None) davranış tamamen
eskisiyle aynıdır - santral geometrisi bilinmeyen bir plant.yaml kaydı için
hiçbir sayı değişmez, uydurma bir tilt/azimuth değeri KULLANILMAZ.
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


def solar_azimuth_deg(timestamp: pd.Timestamp, lat: float, lon: float,
                       zenith_deg: float | None = None, cos_zenith: float | None = None) -> float:
    """
    Güneş azimut açısı (Güney=0°, Batı=+90°, Doğu=-90° - Duffie & Beckman
    "Solar Engineering of Thermal Processes" kongvansiyonu). Düz-GHI
    varsayımından farklı olarak eğik/tracker'lı panel için ışınımı (POA)
    hesaplarken zenith açısı tek başına yetmez, panelin güneşe göre HANGİ
    yönde durduğu da gerekir.
    """
    if zenith_deg is None or cos_zenith is None:
        zenith_deg, cos_zenith = solar_position(timestamp, lat, lon)

    day_of_year = timestamp.dayofyear
    hour = timestamp.hour + timestamp.minute / 60.0
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
    hour_angle = 15 * (hour - 12)

    lat_r, dec_r, ha_r = np.radians(lat), np.radians(declination), np.radians(hour_angle)
    sin_zenith = np.sqrt(max(0.0, 1 - cos_zenith ** 2))
    if sin_zenith < 1e-6:
        return 0.0  # güneş tam tepede - azimut tanımsız/önemsiz, POA'yı etkilemez

    sin_az = np.cos(dec_r) * np.sin(ha_r) / sin_zenith
    cos_az = (cos_zenith * np.sin(lat_r) - np.sin(dec_r)) / (sin_zenith * np.cos(lat_r))
    return float(np.degrees(np.arctan2(np.clip(sin_az, -1, 1), np.clip(cos_az, -1, 1))))


SOLAR_CONSTANT_W_M2 = 1367.0  # gerçek atmosfer-dışı ışınım sabiti - clearsky_ghi_estimate'teki
                               # 1000 basitleştirmesinden farklı, Erbs ayrıştırması bunu gerektirir


def erbs_decomposition(ghi: float, cos_zenith: float) -> tuple[float, float]:
    """
    GHI'den DNI/DHI ayrıştırması (Erbs ve ark. 1982 korelasyonu) - santrale
    ÖZGÜ bir tahmin değil, yayınlanmış genel bir ampirik model. dni_forecast/
    dhi_forecast eksik olduğunda (örn. dni/dhi sütunu eklenmeden ÖNCE çekilmiş
    eski data/ CSV'leri) POA hesaplaması için kullanılır.
    """
    if ghi <= 0 or cos_zenith <= 0.01:
        return 0.0, 0.0

    ghi_extraterrestrial = SOLAR_CONSTANT_W_M2 * cos_zenith
    if ghi_extraterrestrial <= 0:
        return 0.0, ghi

    kt = np.clip(ghi / ghi_extraterrestrial, 0.0, 1.0)  # clearness index
    if kt <= 0.22:
        diffuse_fraction = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        diffuse_fraction = (0.9511 - 0.1604 * kt + 4.388 * kt ** 2
                             - 16.638 * kt ** 3 + 12.336 * kt ** 4)
    else:
        diffuse_fraction = 0.165
    diffuse_fraction = np.clip(diffuse_fraction, 0.0, 1.0)

    dhi = diffuse_fraction * ghi
    # cos_zenith gün doğumu/batımına yakın (0.01'e çok yakın ama üstünde)
    # çok küçük olabilir - bu bölmeyi sayısal olarak kararsız hale getirip
    # DNI'yi fiziksel olarak imkansız değerlere şişirebilir (gözlemlenen:
    # 8000+ W/m^2). DNI, atmosfer-dışı güneş sabitini (SOLAR_CONSTANT_W_M2)
    # hiçbir zaman aşamaz - bu gerçek bir fiziksel üst sınır, santrale özgü
    # bir varsayım değil. Tracker'lı panellerde bu hata özellikle önemli:
    # tracker güneşe yakından baktığı (cos_aoi≈1) için şişirilmiş DNI'yi
    # neredeyse olduğu gibi yutar, düz panel ise aynı saatte zaten düşük
    # cos_zenith ile çarptığı için doğal olarak korunur.
    dni = np.clip((ghi - dhi) / cos_zenith, 0.0, SOLAR_CONSTANT_W_M2)
    return dni, dhi


MAX_TRACKER_ROTATION_DEG = 60.0  # tipik tek-eksen tracker mekanik dönüş limiti
GROUND_ALBEDO = 0.2  # tipik zemin yansıma katsayısı (PVWatts/pvlib varsayılanı)
DEFAULT_GCR = 0.4  # tipik utility-scale tek-eksen tracker ground coverage ratio (Wc/pitch) - datasheet yoksa endüstri-tipik varsayılan


def _cos_aoi_fixed_tilt(zenith_deg: float, azimuth_deg_sun: float,
                         tilt_deg: float, panel_azimuth_deg: float) -> float:
    """Sabit-tilt panel için güneş geliş açısı (AOI) kosinüsü (Duffie & Beckman eşitliği 1.6.2)."""
    zen_r, tilt_r = np.radians(zenith_deg), np.radians(tilt_deg)
    daz_r = np.radians(azimuth_deg_sun - panel_azimuth_deg)
    return np.cos(zen_r) * np.cos(tilt_r) + np.sin(zen_r) * np.sin(tilt_r) * np.cos(daz_r)


def _apply_backtracking(true_rotation_r: float, gcr: float) -> float:
    """
    True-tracking dönüş açısını (güneşe tam bakan, gölgelemeyi hesaba
    katmayan açı) backtracking ile düzeltir - düşük güneş açılarında
    (gün doğumu/batımı) sıra-arası kendi-gölgelemeyi önlemek için trackerın
    tam güneşi takip ETMEYİP daha yataya yakın durması.

    Standart formül (Lorenzo ve ark. 2011 "Tracking and back-tracking";
    pvlib.tracking.singleaxis ile aynı): GCR = Wc/pitch (panel genişliği /
    sıra aralığı).

    Doğrulama (asimptotik sınırlar):
      - GCR=1 (sıra arası boşluk yok) -> HER açıda tam düzleşme (rotation=0),
        çünkü herhangi bir dönüş anında gölgeleme kaçınılmaz.
      - GCR->0 (sıralar sonsuz uzak) -> hiç backtracking yok, true-tracking
        aynen korunur.
      - true_rotation=0 (öğlen) -> backtracking düzeltmesi 0.
    Bu sınırların hepsi fiziksel olarak beklenen davranışla eşleşiyor.
    """
    if gcr <= 0:
        return true_rotation_r
    temp = min(1.0, np.cos(true_rotation_r) / gcr)
    backtrack_correction_r = -np.sign(true_rotation_r) * np.arccos(np.clip(temp, -1.0, 1.0))
    return true_rotation_r + backtrack_correction_r


def _tracker_rotation_and_cos_aoi(zenith_deg: float, azimuth_deg_sun: float,
                                   max_rotation_deg: float = MAX_TRACKER_ROTATION_DEG,
                                   gcr: float = DEFAULT_GCR) -> tuple[float, float]:
    """
    Tek-eksen, YATAY, Kuzey-Güney doğrultulu tracker için dönüş açısı ve AOI
    kosinüsü (en yaygın utility-scale tracker konfigürasyonu; farklı eksen
    azimutu/eğimi desteklenmiyor).

    Türetme: güneş vektörünün panel normaliyle iç çarpımını (cos AOI)
    maksimize eden dönüş açısı (true-tracking). Eksen sabit K-G olduğu için
    güneşin K-G bileşeni tracker tarafından hiç yakalanamaz - bu, tek-eksen
    tracker'ların bilinen bir "cosine loss" kaynağıdır, fiziksel olarak
    beklenen bir sınır.

    gcr (ground coverage ratio) verilirse backtracking uygulanır (bkz.
    _apply_backtracking) - bu OLMADAN model, gerçek trackerların düşük
    güneş açılarında YAPMADIĞI bir tam rotasyonu varsayıp ışınımı
    olduğundan fazla tahmin eder (gözlemlenen etki: Karapınar'da
    backtracking'siz tracker modeli düz-GHI'den DAHA KÖTÜ performans
    gösterdi - bkz. proje geçmişi).
    """
    zen_r = np.radians(zenith_deg)
    az_r = np.radians(azimuth_deg_sun)
    true_rotation_r = np.arctan2(-np.sin(zen_r) * np.sin(az_r), np.cos(zen_r))
    true_rotation_r = np.clip(true_rotation_r, np.radians(-max_rotation_deg), np.radians(max_rotation_deg))

    rotation_r = _apply_backtracking(true_rotation_r, gcr)

    cos_aoi = np.cos(zen_r) * np.cos(rotation_r) - np.sin(zen_r) * np.sin(az_r) * np.sin(rotation_r)
    return float(np.degrees(rotation_r)), float(cos_aoi)


def poa_irradiance(dni: float, dhi: float, ghi: float, zenith_deg: float,
                    azimuth_deg_sun: float, tilt_deg: float | None,
                    panel_azimuth_deg: float | None, tracker_type: str | None,
                    albedo: float = GROUND_ALBEDO, gcr: float = DEFAULT_GCR) -> float:
    """
    Panel düzlemindeki (plane-of-array, POA) toplam ışınımı hesaplar -
    izotropik gökyüzü modeli (Liu-Jordan): POA = doğrudan + izotropik difüz +
    zeminden yansıyan. Düz-GHI varsayımından farkı: panel eğik veya
    tracker'lıysa güneşin geliş açısını (AOI) ve panelin gökyüzü/zemin görme
    faktörünü hesaba katar - tracker'lı bir santralde düz GHI kullanmak
    sistematik hata üretir.

    tilt_deg/panel_azimuth_deg None ve tracker_type None ise bu fonksiyon
    çağrılmamalı - çağıran taraf flat-GHI yoluna düşmeli (bkz.
    build_physical_baseline / _row_irradiance_and_temp). Bilinmeyen bir
    santral geometrisi için değer UYDURULMAZ.
    """
    if zenith_deg >= 90:
        return 0.0

    if tracker_type == "single_axis_horizontal":
        effective_tilt_deg, cos_aoi = _tracker_rotation_and_cos_aoi(zenith_deg, azimuth_deg_sun, gcr=gcr)
    else:
        effective_tilt_deg = tilt_deg if tilt_deg is not None else 0.0
        cos_aoi = _cos_aoi_fixed_tilt(zenith_deg, azimuth_deg_sun, effective_tilt_deg,
                                       panel_azimuth_deg if panel_azimuth_deg is not None else 0.0)

    cos_aoi = max(0.0, cos_aoi)
    tilt_r = np.radians(effective_tilt_deg)

    direct = dni * cos_aoi
    diffuse = dhi * (1 + np.cos(tilt_r)) / 2
    ground_reflected = ghi * albedo * (1 - np.cos(tilt_r)) / 2
    return max(0.0, direct + diffuse + ground_reflected)


def cell_temperature(poa_irradiance_w_m2: float, temp_c: float, noct_c: float = 45.0) -> float:
    """
    NOCT (Nominal Operating Cell Temperature) modeliyle modül/hücre
    sıcaklığını tahmin eder. "Ortam+25°C" sabit yaklaşımından farkı, gerçek
    ışınım seviyesine göre ölçeklenmesi (düşük ışınımda ısınma daha azdır -
    sabah/akşam saatlerinde önemli bir fark yaratır).
    noct_c: üreticinin datasheet'inde verdiği NOCT değeri; datasheet yoksa
    tipik silikon panellerdeki endüstri-standart varsayılan ~45°C kullanılır.
    """
    return temp_c + (noct_c - 20.0) / 800.0 * poa_irradiance_w_m2


def clearsky_ghi_estimate(cos_zenith: float, ghi_toa_max: float = 1000.0) -> float:
    """
    Çok basitleştirilmiş açık-gökyüzü (clear-sky) GHI tahmini.
    Gerçek projede pvlib.clearsky (Ineichen/Haurwitz modeli) kullanmak
    daha doğru olur; burada kavramı göstermek için basit tutuyoruz.
    """
    if cos_zenith <= 0:
        return 0.0
    return max(0.0, ghi_toa_max * cos_zenith)


def _pv_power_core(irradiance_w_m2: float, panel_temp_c: float, capacity_mw: float,
                    ref_temp: float = 25.0, temp_coeff: float = -0.004,
                    efficiency_scale: float = 1.0) -> float:
    """
    Işınım (W/m^2) ve ZATEN HESAPLANMIŞ panel/hücre sıcaklığından PV gücünü
    (MW) çıkarır - ışınım oranı + sıcaklık derating + kapasite ölçekleme.
    physical_pv_power (düz-GHI, "ortam+25°C" varsayımı) ve
    build_physical_baseline'ın POA yolu (gerçek NOCT hücre sıcaklığı) bu
    ortak çekirdeği kullanır, ikisi arasında tutarsızlık kalmasın diye.
    """
    if irradiance_w_m2 <= 0:
        return 0.0

    irradiance_ratio = irradiance_w_m2 / 1000.0  # 1000 W/m^2 = STC referansı
    temp_derating = np.clip(1 + temp_coeff * (panel_temp_c - ref_temp), 0.5, 1.05)
    power_mw = capacity_mw * efficiency_scale * irradiance_ratio * temp_derating
    return max(0.0, power_mw)


def physical_pv_power(ghi_forecast: float, temp_c: float, capacity_mw: float,
                       ghi_clearsky: float, ref_temp: float = 25.0,
                       temp_coeff: float = -0.004,
                       efficiency_scale: float = 1.0) -> float:
    """
    GHI tahmininden ve sıcaklıktan teorik PV gücünü (MW) hesaplar - santral
    geometrisi (tilt/azimuth/tracker) bilinmeyen durumlar için düz-yüzey
    (horizontal) baz model.

    - GHI oranı: gerçek/beklenen ışınımın kurulu güce oranı (basit lineer model)
    - Sıcaklık derating: panel sıcaklığı arttıkça verim düşer
      (tipik silikon panel katsayısı ~ -0.4%/°C, 25°C referans)
    - efficiency_scale: nominal kapasiteye göre santralin gerçek etkin verimi
      (calibrate_site_parameters'tan gelir; datasheet yoksa varsayılan 1.0)
    """
    if ghi_forecast <= 0:
        return 0.0
    # Panel sıcaklığı ortam sıcaklığından biraz yüksek olur (basit yaklaşım: +25°C)
    panel_temp = temp_c + 25.0
    return _pv_power_core(ghi_forecast, panel_temp, capacity_mw,
                           ref_temp=ref_temp, temp_coeff=temp_coeff, efficiency_scale=efficiency_scale)


def _row_irradiance_and_temp(timestamp: pd.Timestamp, ghi: float, dni, dhi, temp_c: float,
                              lat: float, lon: float, tilt_deg: float | None,
                              panel_azimuth_deg: float | None, tracker_type: str | None,
                              module_noct_c: float, gcr: float = DEFAULT_GCR) -> tuple[float, float]:
    """
    Tek bir saat için (ışınım, panel/hücre sıcaklığı) çiftini döner -
    build_physical_baseline ve calibrate_site_parameters AYNI hesaplamayı
    kullansın diye tek yerden yönetiliyor (aralarında tutarsızlık olursa
    kalibrasyon yanlış fiziksel modele göre yapılmış olur).

    Santral geometrisi bilinmiyorsa (tilt_deg=None ve tracker_type=None,
    varsayılan) eski düz-GHI + "ortam+25°C" yaklaşımına döner - davranış
    hiç değişmez. Geometri biliniyorsa DNI/DHI'den POA ışınımı ve NOCT
    tabanlı gerçek hücre sıcaklığı hesaplanır; dni/dhi eksikse (örn. eski
    veri) Erbs ayrıştırmasıyla GHI'den tahmin edilir.
    """
    known_geometry = tilt_deg is not None or tracker_type is not None
    if not known_geometry:
        return ghi, temp_c + 25.0

    zenith, cos_z = solar_position(timestamp, lat, lon)
    if dni is None or dhi is None or pd.isna(dni) or pd.isna(dhi):
        dni, dhi = erbs_decomposition(ghi, cos_z)
    azimuth_sun = solar_azimuth_deg(timestamp, lat, lon, zenith, cos_z)
    poa = poa_irradiance(dni, dhi, ghi, zenith, azimuth_sun, tilt_deg, panel_azimuth_deg, tracker_type, gcr=gcr)
    return poa, cell_temperature(poa, temp_c, module_noct_c)


def build_physical_baseline(df: pd.DataFrame, lat: float, lon: float,
                             capacity_mw: float, temp_coeff: float = -0.004,
                             efficiency_scale: float = 1.0,
                             tilt_deg: float | None = None,
                             panel_azimuth_deg: float | None = None,
                             tracker_type: str | None = None,
                             module_noct_c: float = 45.0,
                             gcr: float = DEFAULT_GCR) -> pd.Series:
    """
    Tüm zaman serisi için fiziksel baz tahmini üretir (MWh, saatlik ise MW=MWh).

    tilt_deg (veya tracker_type) verilirse: df'teki dni_forecast/dhi_forecast
    kullanılarak plane-of-array (POA) ışınımı ve NOCT tabanlı gerçek hücre
    sıcaklığı hesaplanır - eğik/tracker'lı panel geometrisi bilinen
    santraller için (bkz. poa_irradiance, cell_temperature). tracker_type
    "single_axis_horizontal" ise gcr (ground coverage ratio) backtracking
    için kullanılır - bkz. _apply_backtracking.

    tilt_deg=None ve tracker_type=None (varsayılan, santral geometrisi
    bilinmiyorsa): eski düz-GHI davranışı korunur, hiçbir sonuç değişmez.
    """
    baseline = []
    for _, row in df.iterrows():
        irradiance, panel_temp = _row_irradiance_and_temp(
            row['timestamp'], row['ghi_forecast'], row.get('dni_forecast'), row.get('dhi_forecast'),
            row['temp_c'], lat, lon, tilt_deg, panel_azimuth_deg, tracker_type, module_noct_c, gcr,
        )
        p = _pv_power_core(irradiance, panel_temp, capacity_mw,
                            temp_coeff=temp_coeff, efficiency_scale=efficiency_scale)
        baseline.append(p)
    return pd.Series(baseline, index=df.index)


# ----------------------------------------------------------------------
# 1b) SANTRALE ÖZGÜ KALİBRASYON: Datasheet yerine geçmiş veriden parametre tahmini
# ----------------------------------------------------------------------

def calibrate_site_parameters(df_calib: pd.DataFrame, lat: float, lon: float,
                               nominal_capacity_mw: float,
                               tilt_deg: float | None = None,
                               panel_azimuth_deg: float | None = None,
                               tracker_type: str | None = None,
                               module_noct_c: float = 45.0,
                               gcr: float = DEFAULT_GCR) -> dict:
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

    tilt_deg/tracker_type verilirse kalibrasyon POA ışınımı ve gerçek NOCT
    hücre sıcaklığı üzerinden yapılır (build_physical_baseline'ın kullandığı
    _row_irradiance_and_temp ile aynı hesap - aksi halde efficiency_scale/
    temp_coeff yanlış fiziksel modele göre fit edilmiş olur ve POA yolunda
    kullanıldığında anlamsızlaşır).

    NOT: Kalibrasyon verisi mümkünse farklı hava koşullarını (açık/bulutlu
    günler, farklı sıcaklıklar) kapsamalı - dar bir aralık overfit'e yol açar.
    En az 2-3 haftalık çeşitli veri önerilir.
    """
    from scipy.optimize import least_squares

    dni_col = df_calib['dni_forecast'] if 'dni_forecast' in df_calib.columns else [None] * len(df_calib)
    dhi_col = df_calib['dhi_forecast'] if 'dhi_forecast' in df_calib.columns else [None] * len(df_calib)

    irradiance_vals, temp_vals, cos_zeniths = [], [], []
    for ts, ghi, dni, dhi, temp in zip(
        df_calib['timestamp'], df_calib['ghi_forecast'], dni_col, dhi_col, df_calib['temp_c'],
    ):
        irr, panel_temp = _row_irradiance_and_temp(
            ts, ghi, dni, dhi, temp, lat, lon, tilt_deg, panel_azimuth_deg, tracker_type, module_noct_c, gcr,
        )
        irradiance_vals.append(irr)
        temp_vals.append(panel_temp)
        cos_zeniths.append(solar_position(ts, lat, lon)[1])

    irradiance = np.array(irradiance_vals)
    temp = np.array(temp_vals)
    cos_zeniths = np.array(cos_zeniths)
    actual = df_calib['production_mwh'].values

    # Gece saatlerini kalibrasyondan çıkar (zenith>90 -> gündüz yok, sinyal taşımaz)
    daylight_mask = cos_zeniths > 0.05
    irradiance_d, temp_d, actual_d = irradiance[daylight_mask], temp[daylight_mask], actual[daylight_mask]

    def residuals(params):
        efficiency_scale, temp_coeff = params
        derating = np.clip(1 + temp_coeff * (temp_d - 25.0), 0.5, 1.05)
        pred = nominal_capacity_mw * efficiency_scale * (irradiance_d / 1000.0) * derating
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

def build_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """
    Residual modeli için özellik mühendisliği.

    lat/lon: güneş yüksekliği ve clear-sky index için gerekli. Bunlar ham
    GHI/saat yerine NORMALİZE edilmiş özellikler olduğundan (kapasiteden ve
    coğrafyadan bağımsız), aynı model farklı santrallerin verisiyle
    eğitildiğinde ("fleet"/çoklu-santral eğitim) çok daha iyi genelleşir.
    """
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

    cos_zeniths = np.array([solar_position(ts, lat, lon)[1] for ts in df['timestamp']])
    zenith_deg = np.degrees(np.arccos(np.clip(cos_zeniths, -1, 1)))
    # Güneş yüksekliği: "öğleden sonra mı" ikili bayrağından farkı, sürekli/
    # nicel olması - mevsime göre kayan gün doğumu/batımı saatini otomatik
    # hesaba katar, sabit "12'den sonra mı" eşiğine ihtiyaç duymaz.
    feats['solar_elevation_deg'] = 90.0 - zenith_deg

    # Clear-sky index (GHI/açık-gökyüzü GHI): ham watt yerine "bulutluluk
    # oranı" - farklı kapasitedeki/coğrafyadaki santraller arasında
    # genellenebilir bir bulut sinyali.
    ghi_clearsky = np.array([clearsky_ghi_estimate(cz) for cz in cos_zeniths])
    with np.errstate(divide='ignore', invalid='ignore'):
        clear_sky_index = np.where(ghi_clearsky > 1.0, feats['ghi_forecast'].values / ghi_clearsky, 0.0)
    feats['clear_sky_index'] = np.clip(clear_sky_index, 0.0, 1.5)

    # Ramp-rate: önceki saate göre GHI değişimi - gün doğumu/batımındaki
    # keskin geçişleri yakalamaya yardımcı olur.
    feats['ghi_ramp_1h'] = df['ghi_forecast'].diff().fillna(0.0).values

    return feats


def train_residual_model(df_train: pd.DataFrame, physical_baseline: pd.Series,
                          lat: float, lon: float):
    """
    Gerçek üretim - fiziksel tahmin farkını (residual) öğrenen RF modeli.
    Neden residual öğreniyoruz (ham üretimi değil)?
    -> Fiziksel model zaten büyük varyansı (gece/gündüz, mevsim) açıklıyor.
       ML sadece kalan "saha kaybı" örüntüsünü öğrenir -> daha az veriyle
       daha kararlı bir model elde edilir.
    """
    X = build_features(df_train, lat, lon)
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
                        ac_capacity_mw: float | None = None,
                        tilt_deg: float | None = None,
                        panel_azimuth_deg: float | None = None,
                        tracker_type: str | None = None,
                        module_noct_c: float = 45.0,
                        gcr: float = DEFAULT_GCR) -> pd.Series:
    """Yeni (gelecek) veri için nihai üretim tahmini: fiziksel + ML düzeltme.

    ac_capacity_mw verilirse üst sınır olarak kullanılır (invertör/şebeke
    tavanı, nominal capacity_mw'den düşük olabilir - clipping'in nedeni budur);
    verilmezse capacity_mw'ye geri döner.

    tilt_deg/panel_azimuth_deg/tracker_type/gcr: santral geometrisi biliniyorsa
    POA tabanlı fiziksel model kullanılır (bkz. build_physical_baseline);
    train_residual_model/calibrate_site_parameters ile AYNI değerler
    geçilmeli, aksi halde model kendi eğitildiği fiziksel varsayımla
    tutarsız bir baz üzerine kurulur.
    """
    baseline = build_physical_baseline(df_new, lat, lon, capacity_mw,
                                        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale,
                                        tilt_deg=tilt_deg, panel_azimuth_deg=panel_azimuth_deg,
                                        tracker_type=tracker_type, module_noct_c=module_noct_c, gcr=gcr)
    X_new = build_features(df_new, lat, lon)
    residual_pred = residual_model.predict(X_new)

    final_pred = baseline + residual_pred
    upper_bound = ac_capacity_mw if ac_capacity_mw is not None else capacity_mw
    # Fiziksel sınırlar: negatif olamaz, şebeke/invertör tavanını aşamaz
    final_pred = final_pred.clip(lower=0, upper=upper_bound)
    return final_pred


def predict_production_from_fleet(df_new: pd.DataFrame, lat: float, lon: float,
                                   capacity_mw: float, fleet_model,
                                   temp_coeff: float = -0.004, efficiency_scale: float = 1.0,
                                   ac_capacity_mw: float | None = None,
                                   tilt_deg: float | None = None,
                                   panel_azimuth_deg: float | None = None,
                                   tracker_type: str | None = None,
                                   module_noct_c: float = 45.0,
                                   gcr: float = DEFAULT_GCR) -> pd.Series:
    """
    predict_production'ın train_fleet.py çıktısı (birden fazla santralin
    verisi üzerinde eğitilmiş, havuzlanmış model) için karşılığı.

    Fark: fleet_model, MWh değil KAPASİTEYE GÖRE NORMALİZE residual (MW/MWp)
    tahmin ediyor (bkz. train_fleet.build_pooled_dataset) - çünkü farklı
    boyuttaki santrallerin ham MWh residual'ları doğrudan havuzlanamaz. Bu
    yüzden model çıktısı burada capacity_mw ile GERİ ÇARPILIYOR; sıradan
    predict_production (tek santral, MWh residual) ile karıştırılmamalı.
    """
    baseline = build_physical_baseline(df_new, lat, lon, capacity_mw,
                                        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale,
                                        tilt_deg=tilt_deg, panel_azimuth_deg=panel_azimuth_deg,
                                        tracker_type=tracker_type, module_noct_c=module_noct_c, gcr=gcr)
    X_new = build_features(df_new, lat, lon)
    residual_pred_normalized = fleet_model.predict(X_new)

    final_pred = baseline + residual_pred_normalized * capacity_mw
    upper_bound = ac_capacity_mw if ac_capacity_mw is not None else capacity_mw
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
    model = train_residual_model(train_df, baseline_train, LAT, LON)

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
