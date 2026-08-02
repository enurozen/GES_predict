"""
EPİAŞ dengesizlik maliyetine duyarlı değerlendirme ve bid optimizasyonu.

evaluate.py/backtest_forecast.py simetrik hata (MAE/RMSE, nMAE) ölçüyor -
ama EPİAŞ'ta dengesizlik cezası yöne göre asimetrik olabilir (taahhüdün
altında/üstünde kalmanın MWh başına maliyeti farklı olabilir; sistem yönüne
ve YEKDEM durumuna göre değişir). Bu modül iki şeyi ayırıyor:

  1. optimal_bid_quantile: klasik "newsvendor" problemi - asimetrik maliyet
     biliniyorsa, medyan tahmin yerine hangi kantili bidlemenin beklenen
     maliyeti minimize ettiğini hesaplar. SAF MATEMATİK, veri gerektirmez,
     bugün kullanılabilir.
  2. imbalance_cost / evaluate_financial: gerçek saatlik dengesizlik
     fiyatlarıyla bir tahmin serisinin GERÇEK maliyetini hesaplar. Gerçek
     fiyat verisi olmadan da (örnek/varsayımsal fiyatlarla) çalışır -
     epias.fetch_imbalance_price_range (şu an ŞABLON, bkz. epias.py)
     bağlanınca bu modülde hiçbir değişiklik gerekmez.
"""

import numpy as np
import pandas as pd


def optimal_bid_quantile(cost_under_mwh: float, cost_over_mwh: float) -> float:
    """
    Newsvendor çözümü: taahhüdün ALTINDA kalmanın (kısa/short - gerçek
    üretim bidden az) MWh başına maliyeti cost_under_mwh, ÜSTÜNDE kalmanın
    (uzun/long - fazla üretim) maliyeti cost_over_mwh ise, beklenen
    dengesizlik maliyetini minimize eden bid, tahmin dağılımının şu
    kantilinde:

        q* = cost_under_mwh / (cost_under_mwh + cost_over_mwh)

    cost_under_mwh == cost_over_mwh (simetrik ceza) ise q*=0.5, yani medyan
    tahmin - mevcut modelin ürettiği nokta tahmine eşdeğer. Asimetri yoksa bu
    fonksiyonun bir faydası olmaz; mevcut MAE-tabanlı model zaten optimaldir.
    """
    if cost_under_mwh < 0 or cost_over_mwh < 0:
        raise ValueError("Maliyetler negatif olamaz.")
    if cost_under_mwh + cost_over_mwh == 0:
        return 0.5
    return cost_under_mwh / (cost_under_mwh + cost_over_mwh)


def imbalance_cost(actual_mwh, bid_mwh, price_short_mwh, price_long_mwh) -> np.ndarray:
    """
    Saatlik dengesizlik maliyeti: gerçekleşen üretim bidlenen değerin
    ALTINDA kalırsa (kısa/short - taahhüdü tutturamama) price_short_mwh
    üzerinden, ÜSTÜNDE kalırsa (uzun/long - fazla üretim) price_long_mwh
    üzerinden fiyatlanır.

    price_short_mwh/price_long_mwh saatlik SMF/PTF farkından türetilen
    gerçek EPİAŞ dengesizlik fiyatları olmalı (bkz.
    epias.fetch_imbalance_price_range - şu an şablon); skaler veya dizi
    olabilir (tüm saatler için sabit fiyat varsayımıyla da kullanılabilir).
    """
    actual_mwh = np.asarray(actual_mwh, dtype=float)
    bid_mwh = np.asarray(bid_mwh, dtype=float)
    price_short_mwh = np.asarray(price_short_mwh, dtype=float)
    price_long_mwh = np.asarray(price_long_mwh, dtype=float)

    deviation = actual_mwh - bid_mwh  # >0: fazla ürettik (long), <0: kısa kaldık (short)
    return np.where(deviation < 0, -deviation * price_short_mwh, deviation * price_long_mwh)


def evaluate_financial(actual_mwh: pd.Series, predictions: dict[str, pd.Series],
                        price_short_mwh, price_long_mwh) -> pd.DataFrame:
    """
    Birden fazla tahmin serisini (ör. {"fiziksel": ..., "hibrit": ...,
    "mükemmel": actual_mwh}) aynı fiyat serisiyle karşılaştırıp toplam/
    ortalama dengesizlik maliyetini döner - nMAE gibi soyut bir metrik
    yerine "bu tahminle gerçekte ne kadar ödenirdi" sorusuna cevap verir.
    Sonuç en ucuzdan en pahalıya sıralanır.
    """
    rows = []
    for label, pred in predictions.items():
        cost = imbalance_cost(actual_mwh.values if hasattr(actual_mwh, "values") else actual_mwh,
                               pred.values if hasattr(pred, "values") else pred,
                               price_short_mwh, price_long_mwh)
        rows.append({
            "model": label,
            "toplam_maliyet": float(np.sum(cost)),
            "ortalama_saatlik_maliyet": float(np.mean(cost)),
        })
    return pd.DataFrame(rows).sort_values("toplam_maliyet").reset_index(drop=True)
