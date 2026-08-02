"""
One-off diagnostic: calls the (endpoint-confirmed, field-names-unverified)
EPİAŞ SMF and imbalance-amount fetchers for a single day and prints the raw
rows - so we can see the real response field names and confirm/fix
epias.py's _PRICE_FIELD_CANDIDATES / _AMOUNT_FIELD_CANDIDATES.

Usage:
    export EPIAS_USERNAME="you@example.com"
    export EPIAS_PASSWORD="your-epias-password"
    python check_epias_endpoints.py --date 2026-08-01
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta

from epias import EpiasError, fetch_imbalance_amount_for_date, fetch_system_marginal_price_for_date, get_tgt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print raw EPİAŞ SMF/imbalance-amount rows for one day.")
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="YYYY-MM-DD, defaults to yesterday")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day = args.date or (date.today() - timedelta(days=1))

    username = os.environ.get("EPIAS_USERNAME")
    password = os.environ.get("EPIAS_PASSWORD")
    if not username or not password:
        logger.error("EPIAS_USERNAME/EPIAS_PASSWORD environment variables are required.")
        return 1

    tgt = get_tgt(username, password)

    for label, fetch_fn in [
        ("Sistem Marjinal Fiyatı (SMF)", fetch_system_marginal_price_for_date),
        ("Dengesizlik Tutarı", fetch_imbalance_amount_for_date),
    ]:
        logger.info("=== %s (%s) ===", label, day)
        try:
            rows = fetch_fn(tgt, day)
        except EpiasError as exc:
            logger.error("%s", exc)
            continue

        if not rows:
            logger.info("Boş yanıt (0 satır).")
            continue

        logger.info("İlk satır: %s", rows[0])
        logger.info("Alan adları: %s", sorted(rows[0].keys()))
        logger.info("Toplam %d satır.", len(rows))

    return 0


if __name__ == "__main__":
    sys.exit(main())
