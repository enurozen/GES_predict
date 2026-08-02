"""
Gün-içi piyasası (intraday) yeniden-tahmin karşılaştırması.

Gerçek emir gönderimi (trading) KAPSAM DIŞI ve BİLİNÇLİ olarak implement
EDİLMEDİ - EPİAŞ üye/trading API entegrasyonu ve finansal risk gerektirir.
Bu script sadece gün-öncesi bidinde kullanılan tahminle, daha sonra (gerçek
zamana daha yakın, daha güncel forecast weather ile) üretilen bir "refresh"
tahmini karşılaştırıp sapma belirlenen bir eşiği aşarsa işaretler - gün-içi
piyasasında pozisyon düzeltmesi değerlendirilmesi gereken saatleri insan
karar vericiye göstermek için.

Kullanım: predict.py'yi iki farklı zamanda çalıştırıp iki CSV üretin
(ör. gün-öncesi kapanışında ve birkaç saat önce), sonra:

    python reforecast_check.py \
        --day-ahead predictions/day_ahead.csv \
        --refresh predictions/refresh.csv \
        --capacity-mw 1000 --tolerance-pct 3
"""

import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compare_forecasts(day_ahead: pd.DataFrame, refresh: pd.DataFrame,
                       capacity_mw: float, tolerance_pct: float = 3.0) -> pd.DataFrame:
    """
    İki predict.py çıktısını (timestamp, predicted_mwh sütunlu) timestamp
    üzerinden birleştirip saatlik sapmayı (refresh - day_ahead) ve bunun
    kapasiteye göre yüzdesini hesaplar; hangi saatlerin tolerance_pct'i
    aştığını (gün-içi piyasasında düzeltme değerlendirilmesi gereken saatler)
    işaretler.
    """
    merged = day_ahead[["timestamp", "predicted_mwh"]].rename(
        columns={"predicted_mwh": "day_ahead_mwh"}
    ).merge(
        refresh[["timestamp", "predicted_mwh"]].rename(columns={"predicted_mwh": "refresh_mwh"}),
        on="timestamp", how="inner",
    )
    merged["deviation_mwh"] = merged["refresh_mwh"] - merged["day_ahead_mwh"]
    merged["deviation_pct_capacity"] = (merged["deviation_mwh"] / capacity_mw * 100).round(2)
    merged["needs_review"] = merged["deviation_pct_capacity"].abs() >= tolerance_pct
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a day-ahead prediction against a later refresh, flag hours that deviate."
    )
    parser.add_argument("--day-ahead", required=True, help="predict.py output CSV used for the day-ahead bid")
    parser.add_argument("--refresh", required=True, help="predict.py output CSV from a later re-run")
    parser.add_argument("--capacity-mw", type=float, required=True)
    parser.add_argument("--tolerance-pct", type=float, default=3.0, help="Flag threshold, %% of capacity")
    parser.add_argument("--output", default=None, help="Optional CSV path to write the full comparison")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        day_ahead = pd.read_csv(args.day_ahead, parse_dates=["timestamp"])
        refresh = pd.read_csv(args.refresh, parse_dates=["timestamp"])
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    comparison = compare_forecasts(day_ahead, refresh, args.capacity_mw, args.tolerance_pct)
    flagged = comparison[comparison["needs_review"]]

    logger.info(
        "%d/%d saatte sapma ±%%%.1f kapasiteyi aştı.",
        len(flagged), len(comparison), args.tolerance_pct,
    )
    for _, row in flagged.iterrows():
        logger.info(
            "  %s: gün-öncesi=%.2f MWh, refresh=%.2f MWh, sapma=%%%.2f",
            row["timestamp"], row["day_ahead_mwh"], row["refresh_mwh"], row["deviation_pct_capacity"],
        )

    if args.output:
        comparison.to_csv(args.output, index=False)
        logger.info("Karşılaştırma %s dosyasına yazıldı.", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
