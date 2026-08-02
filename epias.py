"""
EPİAŞ Transparency Platform client: authentication + generation data fetch.

Mirrors the API layer of the sibling DataPull_EPIAS/app.py project so this
project can build training datasets without depending on that project (and
its Streamlit UI) at runtime.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Callable, Optional

import pandas as pd

from shared import ApiError, request_with_retries

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
GENERATION_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/"
    "realtime-generation-bulk"
)

# Sistem Marjinal Fiyatı (SMF) - Şeffaflık Platformu API dokümantasyonu
# madde 5.113. URL/HTTP method/TGT auth/tarih formatı (ISO-8601) VE response
# alan adları CANLI ÇAĞRIYLA DOĞRULANDI (2026-08-02, check_epias_endpoints.py
# ile): {"date": "...", "hour": "...", "systemMarginalPrice": <float>}.
SMF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/bpm/data/system-marginal-price"

# NOT: EPİAŞ Şeffaflık Platformu'nun "Dengesizlik Tutarı" servisi (madde
# 5.183) bilinçli olarak KULLANILMIYOR - katılımcı/santral bazlı (Karapınar'a
# özel) veri halka açık verilmiyor, muhtemelen sistem/bölge toplamını
# dönüyor (canlı çağrıda 2026-08-02'de iki farklı tarihte 0 satır). Santrale
# özel dengesizlik tutarı sadece EPİAŞ'ın kapalı PYS/YS (Piyasa Yönetim
# Sistemi) hesabında görünür. imbalance_cost.py zaten buna ihtiyaç duymuyor:
# dengesizlik maliyeti = SMF (sistem geneli, aşağıda) × santralin KENDİ
# sapması (predict.py'ın tahmini vs. data/'daki gerçekleşen üretim).

# SMF alan adları canlı çağrıyla doğrulandı (systemMarginalPrice ilk
# denemede tuttu) - hiçbiri tutmazsa (_first_matching_field) sessizce
# yanlış/boş sonuç üretmek yerine gerçek response'taki alan adlarını
# gösteren net bir EpiasError fırlatılır.
_PRICE_FIELD_CANDIDATES = ["systemMarginalPrice", "price", "smp", "smpValue", "value"]

REQUEST_TIMEOUT_LOGIN = 10
REQUEST_TIMEOUT_DATA = 15
MAX_WORKERS = 2


class EpiasError(ApiError):
    """A user-facing error while talking to the EPİAŞ API."""


class TokenExpiredError(EpiasError):
    """Raised when the API rejects the current TGT; caller should re-authenticate."""


def get_tgt(username: str, password: str) -> str:
    """Authenticate with EPİAŞ and return a TGT session ticket.

    The EPİAŞ account password is exchanged for a short-lived ticket (TGT)
    on every login; EPİAŞ does not issue a static API key.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
    }
    payload = {"username": username, "password": password}

    response = request_with_retries(
        "POST",
        CAS_URL,
        timeout=REQUEST_TIMEOUT_LOGIN,
        error_context="Could not reach the EPİAŞ login server",
        headers=headers,
        data=payload,
    )

    if response.status_code == 201:
        return response.text.strip()
    if response.status_code == 401:
        raise EpiasError(
            "Invalid Token: EPİAŞ rejected the supplied username/password."
        )
    raise EpiasError(f"EPİAŞ login failed (HTTP {response.status_code}).")


def fetch_generation_for_date(tgt: str, plant_id: int, day: date) -> list[dict[str, Any]]:
    """Fetch one day of hourly generation data for a single power plant."""
    headers = {
        "TGT": tgt,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "date": f"{day.isoformat()}T00:00:00+03:00",
        "powerPlantIds": [plant_id],
    }

    response = request_with_retries(
        "POST",
        GENERATION_URL,
        timeout=REQUEST_TIMEOUT_DATA,
        error_context=f"Connection error while fetching data for {day}",
        json=payload,
        headers=headers,
    )

    if response.status_code in (401, 403):
        raise TokenExpiredError(
            "Invalid Token: your session token was rejected or has expired."
        )
    if response.status_code == 404:
        raise EpiasError(f"Santral Code not found: no power plant with ID {plant_id}.")
    if not response.ok:
        raise EpiasError(
            f"EPİAŞ API returned an error (HTTP {response.status_code}) for {day}."
        )

    body = response.json()
    rows = body.get("items", body) if isinstance(body, dict) else body
    return rows or []


def _first_matching_field(row: dict[str, Any], candidates: list[str]) -> str:
    """İlk satırda candidates listesindeki alan adlarından hangisi varsa onu döner.

    Hiçbiri yoksa (DTO alan adları tahminimizden farklıysa) sessizce yanlış/
    boş sonuç üretmek yerine gerçek alan adlarını gösteren bir hata fırlatır -
    bu hatadaki isimlerle SMF_URL'nin candidate listesini güncellemek, tek
    satırlık bir düzeltmedir.
    """
    for name in candidates:
        if name in row:
            return name
    raise EpiasError(
        "EPİAŞ yanıtındaki alan adları tanınmadı - tahmin edilen "
        f"{candidates} listesinde hiçbiri yok. Gerçek alanlar: {sorted(row.keys())}. "
        "epias.py'deki _PRICE_FIELD_CANDIDATES listesini bu gerçek alan "
        "adlarıyla güncelleyin."
    )


def fetch_system_marginal_price_for_date(tgt: str, day: date) -> list[dict[str, Any]]:
    """
    Bir günün saatlik Sistem Marjinal Fiyatı (SMF) verisini çeker (madde
    5.113 - URL/method/auth/tarih formatı (ISO-8601 startDate/endDate) VE
    response alan adları CANLI ÇAĞRIYLA DOĞRULANDI, bkz. yukarısı).
    """
    headers = {"TGT": tgt, "Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "startDate": f"{day.isoformat()}T00:00:00+03:00",
        "endDate": f"{day.isoformat()}T23:59:59+03:00",
    }

    response = request_with_retries(
        "POST",
        SMF_URL,
        timeout=REQUEST_TIMEOUT_DATA,
        error_context=f"Connection error while fetching system marginal price for {day}",
        json=payload,
        headers=headers,
    )

    if response.status_code in (401, 403):
        raise TokenExpiredError(
            "Invalid Token: your session token was rejected or has expired."
        )
    if not response.ok:
        raise EpiasError(
            f"EPİAŞ API returned an error (HTTP {response.status_code}) for {day}."
        )

    body = response.json()
    rows = body.get("items", body) if isinstance(body, dict) else body
    rows = rows or []
    if rows:
        _first_matching_field(rows[0], _PRICE_FIELD_CANDIDATES)  # erken doğrula, yanlışsa hemen patlasın
    return rows


def fetch_imbalance_price_for_date(tgt: str, day: date) -> list[dict[str, Any]]:
    """imbalance_cost.py'nin beklediği isim - fetch_system_marginal_price_for_date'e ince bir sarmalayıcı."""
    return fetch_system_marginal_price_for_date(tgt, day)


def fetch_generation_range(
    tgt: str,
    plant_id: int,
    start: date,
    end: date,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Fetch generation data for every day in [start, end] and combine into a DataFrame.

    Days are fetched concurrently (capped at MAX_WORKERS) since each day is an
    independent request; a fixed worker cap keeps this from hammering the API
    on wide date ranges.
    """
    total_days = (end - start).days + 1
    days = [start + timedelta(days=i) for i in range(total_days)]
    all_rows: list[dict[str, Any]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_days)) as executor:
        future_to_day = {
            executor.submit(fetch_generation_for_date, tgt, plant_id, day): day
            for day in days
        }
        try:
            for future in as_completed(future_to_day):
                rows = future.result()  # re-raises any error from that day's request
                for row in rows:
                    raw_date = row.get("date", "")
                    all_rows.append(
                        {
                            "Date": raw_date.split("T")[0] if "T" in raw_date else raw_date,
                            "Hour": row.get("hour", "00:00"),
                            "Generation (MWh)": row.get("sun", row.get("total", 0)),
                        }
                    )
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_days)
        finally:
            # Best-effort: stop any not-yet-started requests once one day fails.
            for pending_future in future_to_day:
                pending_future.cancel()

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    return df
