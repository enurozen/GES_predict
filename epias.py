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

# Dengesizlik Tutarı - madde 5.183. URL/method/auth teyit edildi; canlı çağrı
# (check_epias_endpoints.py, 2026-08-02, iki farklı tarih) her ikisinde de 0
# satır döndü. Sebebi muhtemelen eksik bir parametre DEĞİL, kavramsal bir
# uyumsuzluk: Şeffaflık Platformu bu servisten katılımcı/santral BAZLI (yani
# doğrudan Karapınar'a özel) dengesizlik cezasını halka açık vermiyor - bu
# servis muhtemelen sistemin/bölgenin toplam finansal dengesizlik hacmini
# dönüyor. Şirketlerin kendi mali uzlaştırma verisi sadece kendi kapalı
# PYS/YS (Piyasa Yönetim Sistemi) hesabında görünür, Şeffaflık Platformu'nda
# değil. Dolu dönse bile bu servis PLANT-ÖZEL veri vermeyeceği için
# imbalance_cost.py'nin ihtiyacı olan şey bu değil - bkz. aşağıdaki
# fetch_imbalance_amount_for_date'in docstring'i.
IMBALANCE_AMOUNT_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/imbalance/data/imbalance-amount"

# SMF alan adları yukarıda doğrulandı (systemMarginalPrice ilk denemede
# tuttu). ImbalanceAmountResponseDto'nunkiler hâlâ tahmin - hiçbiri tutmazsa
# (_first_matching_field) sessizce yanlış/boş sonuç üretmek yerine gerçek
# response'taki alan adlarını gösteren net bir EpiasError fırlatılır.
_PRICE_FIELD_CANDIDATES = ["systemMarginalPrice", "price", "smp", "smpValue", "value"]
_AMOUNT_FIELD_CANDIDATES = ["imbalanceAmount", "amount", "netAmount", "value"]
_DATE_FIELD_CANDIDATES = ["date", "time", "hour"]

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
    bu hatadaki isimlerle SMF_URL/IMBALANCE_AMOUNT_URL'nin candidate
    listelerini güncellemek, tek satırlık bir düzeltmedir.
    """
    for name in candidates:
        if name in row:
            return name
    raise EpiasError(
        "EPİAŞ yanıtındaki alan adları tanınmadı - tahmin edilen "
        f"{candidates} listesinde hiçbiri yok. Gerçek alanlar: {sorted(row.keys())}. "
        "epias.py'deki _PRICE_FIELD_CANDIDATES/_AMOUNT_FIELD_CANDIDATES/"
        "_DATE_FIELD_CANDIDATES listelerini bu gerçek alan adlarıyla güncelleyin."
    )


def _fetch_market_data_for_date(tgt: str, url: str, day: date, error_label: str) -> list[dict[str, Any]]:
    """SMF ve dengesizlik tutarı endpoint'lerinin ortak isteği - ikisi de aynı
    POST + TGT header + startDate/endDate (ISO-8601) desenini kullanıyor
    (EPİAŞ Şeffaflık Platformu API dokümantasyonu, "İstemci Oluşturmak"
    bölümü)."""
    headers = {"TGT": tgt, "Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "startDate": f"{day.isoformat()}T00:00:00+03:00",
        "endDate": f"{day.isoformat()}T23:59:59+03:00",
    }

    response = request_with_retries(
        "POST",
        url,
        timeout=REQUEST_TIMEOUT_DATA,
        error_context=f"Connection error while fetching {error_label} for {day}",
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
    return rows or []


def fetch_system_marginal_price_for_date(tgt: str, day: date) -> list[dict[str, Any]]:
    """
    Bir günün saatlik Sistem Marjinal Fiyatı (SMF) verisini çeker (madde
    5.113 - URL/method/auth teyit edildi, response alan adları
    _PRICE_FIELD_CANDIDATES ile tahmin ediliyor, bkz. yukarısı).
    """
    rows = _fetch_market_data_for_date(tgt, SMF_URL, day, "system marginal price")
    if rows:
        _first_matching_field(rows[0], _PRICE_FIELD_CANDIDATES)  # erken doğrula, yanlışsa hemen patlasın
    return rows


def fetch_imbalance_amount_for_date(tgt: str, day: date) -> list[dict[str, Any]]:
    """
    Bir günün saatlik Dengesizlik Tutarı verisini çeker (madde 5.183 -
    URL/method/auth teyit edildi, response alan adları
    _AMOUNT_FIELD_CANDIDATES ile tahmin ediliyor, bkz. yukarısı).

    KULLANIM UYARISI: bu servis Şeffaflık Platformu'nda katılımcı/santral
    BAZLI (Karapınar'a özel) dengesizlik verisini halka açık vermiyor -
    muhtemelen sistemin/bölgenin toplam finansal hacmini dönüyor (bu yüzden
    2026-08-02'de iki farklı tarihte 0 satır döndü). Santrale özel dengesizlik
    tutarı sadece EPİAŞ'ın kapalı PYS/YS (Piyasa Yönetim Sistemi) hesabında
    görünür, bu API'den ASLA gelmeyecek.

    imbalance_cost.py'nin ihtiyacı olan şey zaten bu değil: dengesizlik
    maliyeti = SMF (fetch_system_marginal_price_for_date - sistem geneli,
    CANLI DOĞRULANDI) * santralin KENDİ sapması (predict.py'ın tahmini vs.
    data/'daki gerçekleşen üretim - ikisi de zaten elimizde, EPİAŞ'tan
    gelmesi gerekmiyor). Yani bu fonksiyon pratikte KULLANILMIYOR - sadece
    referans/gelecekte bir bölgesel-hacim analizi ihtiyacı olursa diye
    tutuluyor.
    """
    rows = _fetch_market_data_for_date(tgt, IMBALANCE_AMOUNT_URL, day, "imbalance amount")
    if rows:
        _first_matching_field(rows[0], _AMOUNT_FIELD_CANDIDATES)
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
