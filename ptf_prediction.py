from __future__ import annotations
import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
PTF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
CONSUMPTION_URL = ("https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/load-estimation-plan")
KGUP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/dpp"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
METEO_TIMEZONE = "Europe/Istanbul"
METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "apparent_temperature",
]
METEO_HOURLY = ",".join(METEO_VARIABLES)
METEO_RENAME = {
    "temperature_2m": "sicaklik_C",
    "relative_humidity_2m": "nem_pct",
    "wind_speed_10m": "ruzgar_kmh",
    "shortwave_radiation": "gunes_radyasyon_Wm2",
    "cloud_cover": "bulutluluk_pct",
    "precipitation": "yagis_mm",
    "apparent_temperature": "hissedilen_C",
}
METEO_FEATURE_COLS = list(METEO_RENAME.values())

BASE_LAG_MAP = {
    "price": [1, 2, 3, 24, 48, 72, 168, 336],
    "consumption": [1, 24, 168],
    "ruzgar": [24],
    "gunes": [24],
    "dogalgaz": [24, 168],
}
DROP_COLUMNS = ["hour", "priceUsd", "priceEur", "time_x", "time_y", "toplam"]

GEN_COLS_YENILENEBILIR = ["ruzgar", "gunes", "akarsu", "barajli", "biokutle", "jeotermal"]
GEN_COLS_TERMIK        = ["dogalgaz", "linyit", "tasKomur", "ithalKomur"]
GEN_COLS_TUMÜ          = GEN_COLS_YENILENEBILIR + GEN_COLS_TERMIK + ["fuelOil", "nafta", "diger"]

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _desktop_path() -> Path:
    return Path(r"C:\Users\User\Desktop")

def load_env_optional() -> None:
    if load_dotenv is None:
        return
    env_path = _project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path)

def load_credentials() -> tuple[str, str]:
    user = os.environ.get("EPIAS_USERNAME", "").strip()
    pwd = os.environ.get("EPIAS_PASSWORD", "").strip()
    if not user or not pwd:
        raise RuntimeError(
            "EPIAS kimlik bilgisi eksik.\n"
            "  EPIAS_USERNAME\n"
            "  EPIAS_PASSWORD\n"
        )
    return user, pwd

def parse_args() -> argparse.Namespace:
    desktop = Path(r"C:\Users\User\Desktop")
    default_forecast = desktop / "forecast.xlsx"
    default_out = desktop / "ptf_tahmin.xlsx"

    p = argparse.ArgumentParser(
        description="Saatlik PTF tahmini (EPIAS + LightGBM + Excel tahmin)."
    )
    p.add_argument("--start-date", default=os.environ.get("PTF_START_DATE", "2025-01-01"))
    p.add_argument("--end-date",   default=os.environ.get("PTF_END_DATE",   "2026-05-03"))
    p.add_argument("--train-before", default=os.environ.get("PTF_TRAIN_BEFORE", "2026-01-01"))
    p.add_argument("--forecast-input",  type=Path, default=Path(os.environ.get("PTF_FORECAST_INPUT",  str(default_forecast))))
    p.add_argument("--output",          type=Path, default=Path(os.environ.get("PTF_FORECAST_OUTPUT", str(default_out))))
    p.add_argument("--lgbm-device", choices=["cpu", "gpu", "auto"], default=os.environ.get("LGBM_DEVICE", "cpu"))
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--no-meteo",    action="store_true")
    p.add_argument("--meteo-lat",   type=float, default=float(os.environ.get("METEO_LAT", "41.0082")))
    p.add_argument("--meteo-lon",   type=float, default=float(os.environ.get("METEO_LON", "28.9784")))
    p.add_argument("--debug-save",  action="store_true")
    p.add_argument("--debug-dir",   type=Path, default=None)
    p.add_argument(
        "--log-target",
        action="store_true",
        default=True,
        help="Fiyatı log(price+1) ile dönüştürerek eğit (spike tahminini düzeltir)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Eğitim setinin early stopping için ayrılacak validation oranı (varsayılan 0.10)",
    )
    return p.parse_args()


def to_iso_tr(d: str) -> str:
    return f"{d}T00:00:00+03:00"


def get_tgt(username: str, password: str) -> str:
    r = requests.post(
        CAS_URL,
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        timeout=40,
    )
    if r.status_code != 201:
        raise SystemExit(f"TGT alinamadi: {r.status_code}-{r.text}")
    return r.headers.get("Location", "").rstrip("/").split("/")[-1]


def fetch_epias_data(url: str, start_date: str, end_date: str, tgt: str, label: str) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    part_start = start
    all_data = []

    while part_start <= end:
        part_end = (part_start + relativedelta(months=3)) - relativedelta(days=1)
        if part_end > end:
            part_end = end
        body = {
            "startDate": to_iso_tr(part_start.strftime("%Y-%m-%d")),
            "endDate":   to_iso_tr(part_end.strftime("%Y-%m-%d")),
        }
        r = requests.post(
            url, json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json", "TGT": tgt},
            timeout=60,
        )
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                all_data.extend(items)
                print(f"{label}: {part_start.date()} - {part_end.date()} arasi {len(items)} kayit.")
            else:
                print(f"{label}: {part_start.date()} - {part_end.date()} veri yok.")
        else:
            print(f"{label} Hata ({r.status_code}) -> {r.text}")
        part_start = part_end + relativedelta(days=1)

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def fetch_kgup(start_date: str, end_date: str, tgt: str) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    part_start = start
    all_data = []

    while part_start <= end:
        part_end = (part_start + relativedelta(months=3)) - relativedelta(days=1)
        if part_end > end:
            part_end = end
        body = {
            "startDate": to_iso_tr(part_start.strftime("%Y-%m-%d")),
            "endDate":   to_iso_tr(part_end.strftime("%Y-%m-%d")),
            "region": "TR1",
        }
        r = requests.post(
            KGUP_URL, json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json", "TGT": tgt},
            timeout=60,
        )
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                all_data.extend(items)
                print(f"KGUP: {part_start.date()} - {part_end.date()} arasi {len(items)} kayit.")
            else:
                print(f"KGUP: {part_start.date()} - {part_end.date()} veri yok.")
        else:
            print(f"KGUP Hata ({r.status_code}) -> {r.text}")
        part_start = part_end + relativedelta(days=1)

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def validate_merge_frames(df_ptf: pd.DataFrame, df_consumption: pd.DataFrame, df_kgup: pd.DataFrame) -> None:
    if df_ptf.empty:
        raise ValueError("PTF verisi bos.")
    if "date" not in df_ptf.columns:
        raise ValueError(f"PTF 'date' kolonu yok. Kolonlar: {list(df_ptf.columns)}")
    if not df_consumption.empty and "date" not in df_consumption.columns:
        raise ValueError(f"Tuketim 'date' kolonu yok. Kolonlar: {list(df_consumption.columns)}")
    if not df_kgup.empty and "date" not in df_kgup.columns:
        raise ValueError(f"KGUP 'date' kolonu yok. Kolonlar: {list(df_kgup.columns)}")


def safe_drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    return df.drop(columns=cols) if cols else df


def normalize_consumption_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "consumption" in df.columns:
        return df
    if "lep" in df.columns:
        df["consumption"] = pd.to_numeric(df["lep"], errors="coerce")
    return df

def build_lag_map(df: pd.DataFrame) -> dict:
    lag_map: dict = {}
    for col, lags in BASE_LAG_MAP.items():
        if col in df.columns:
            lag_map[col] = lags
        else:
            print(f"Uyari: '{col}' kolonu bulunamadi, lag atlanıyor.")
    return lag_map


def _meteo_day(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def align_hourly_stamp(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, errors="coerce")
    try:
        if t.dt.tz is not None:
            t = t.dt.tz_convert(ZoneInfo(METEO_TIMEZONE)).dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return t.dt.floor("h")


def fetch_meteo_historical(start: str, end: str, latitude: float, longitude: float) -> pd.DataFrame:
    r = requests.get(
        OPEN_METEO_ARCHIVE,
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start, "end_date": end,
            "hourly": METEO_HOURLY, "timezone": METEO_TIMEZONE,
        },
        timeout=60,
    )
    r.raise_for_status()
    hourly = r.json().get("hourly")
    if not hourly or not hourly.get("time"):
        return pd.DataFrame()
    return pd.DataFrame(hourly)


def fetch_meteo_forecast_short(filter_end_date: str, latitude: float, longitude: float) -> pd.DataFrame:
    r = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": latitude, "longitude": longitude,
            "hourly": METEO_HOURLY, "timezone": METEO_TIMEZONE,
            "forecast_days": 2,
        },
        timeout=30,
    )
    r.raise_for_status()
    hourly = r.json().get("hourly")
    if not hourly or not hourly.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df[df["time"].dt.strftime("%Y-%m-%d") <= filter_end_date]


def fetch_meteo_forecast_range(start_date: str, end_date: str, latitude: float, longitude: float) -> pd.DataFrame:
    r = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "hourly": METEO_HOURLY, "timezone": METEO_TIMEZONE,
        },
        timeout=120,
    )
    if r.status_code != 200:
        print(f"Hava (forecast aralik) HTTP {r.status_code}: {r.text[:400]}")
        return pd.DataFrame()
    hourly = r.json().get("hourly")
    if not hourly or not hourly.get("time"):
        return pd.DataFrame()
    return pd.DataFrame(hourly)


def meteo_raw_to_merge_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.rename(columns=METEO_RENAME)
    out["date"] = out["time"]
    out = out.drop(columns=["time"])
    for c in METEO_FEATURE_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["date"] = align_hourly_stamp(out["date"])
    out = out.drop_duplicates(subset=["date"], keep="first")
    return out.sort_values("date").reset_index(drop=True)


def get_meteo_for_epias_merged(epias_start: str, epias_end: str, latitude: float, longitude: float) -> pd.DataFrame:
    tz    = ZoneInfo(METEO_TIMEZONE)
    bitis = (datetime.now(tz).date() - timedelta(days=1)).strftime("%Y-%m-%d")
    yarin = (datetime.now(tz).date() + timedelta(days=1)).strftime("%Y-%m-%d")

    e0 = _meteo_day(epias_start)
    e1 = _meteo_day(epias_end)
    d_bitis  = _meteo_day(bitis)
    hist_end = min(e1, d_bitis)

    parts: list[pd.DataFrame] = []
    if e0 <= hist_end:
        print(f"Hava archive: {e0.isoformat()} -> {hist_end.isoformat()}")
        parts.append(fetch_meteo_historical(e0.isoformat(), hist_end.isoformat(), latitude, longitude))

    print(f"Hava forecast (2 gun, <= {yarin}): birlestiriliyor")
    parts.append(fetch_meteo_forecast_short(yarin, latitude, longitude))

    raw = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    if raw.empty or "time" not in raw.columns:
        return pd.DataFrame()
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)
    out = meteo_raw_to_merge_df(raw)
    print(f"Hava toplam: {len(out)} saat | {out['date'].min()} -> {out['date'].max()}")
    return out


def get_meteo_for_forecast_excel(d_min: pd.Timestamp, d_max: pd.Timestamp, latitude: float, longitude: float) -> pd.DataFrame:
    if pd.isna(d_min) or pd.isna(d_max):
        return pd.DataFrame()
    d0 = align_hourly_stamp(pd.Series([d_min])).iloc[0]
    d1 = align_hourly_stamp(pd.Series([d_max])).iloc[0]
    if d0 > d1:
        return pd.DataFrame()

    tz    = ZoneInfo(METEO_TIMEZONE)
    today = datetime.now(tz).date()
    yday  = today - timedelta(days=1)

    start_d = d0.date()
    end_d   = d1.date()
    chunks: list[pd.DataFrame] = []

    if start_d <= yday:
        h_end = min(end_d, yday)
        if start_d <= h_end:
            print(f"Hava (Excel archive): {start_d} -> {h_end}")
            raw_h = fetch_meteo_historical(start_d.isoformat(), h_end.isoformat(), latitude, longitude)
            if not raw_h.empty:
                chunks.append(raw_h)

    f0 = max(start_d, today)
    if f0 <= end_d:
        print(f"Hava (Excel forecast): {f0} -> {end_d}")
        raw_f = fetch_meteo_forecast_range(f0.isoformat(), end_d.isoformat(), latitude, longitude)
        if not raw_f.empty:
            chunks.append(raw_f)
    elif start_d > yday:
        print(f"Hava (Excel sadece forecast): {start_d} -> {end_d}")
        raw_f = fetch_meteo_forecast_range(start_d.isoformat(), end_d.isoformat(), latitude, longitude)
        if not raw_f.empty:
            chunks.append(raw_f)

    if not chunks:
        return pd.DataFrame()
    raw = pd.concat(chunks, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)
    out  = meteo_raw_to_merge_df(raw)
    mask = (out["date"] >= d0) & (out["date"] <= d1)
    out  = out.loc[mask].reset_index(drop=True)
    print(f"Hava Excel: {len(out)} saat | {out['date'].min()} -> {out['date'].max()}")
    return out


def merge_meteo_hourly(left: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    left = left.copy()
    left["date"] = align_hourly_stamp(left["date"])
    for c in METEO_FEATURE_COLS:
        if c in left.columns:
            left = left.drop(columns=[c])
    if meteo is None or meteo.empty:
        print("Uyari: hava verisi bos.")
        return left
    m = meteo.copy()
    m["date"] = align_hourly_stamp(m["date"])
    m = m.drop_duplicates(subset=["date"], keep="first")
    return left.merge(m, on="date", how="left")


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ay"]              = df["date"].dt.month
    df["gün"]             = df["date"].dt.day
    df["saat"]            = df["date"].dt.hour
    df["haftanıngünleri"] = df["date"].dt.dayofweek
    df["yilin_haftasi"]   = df["date"].dt.isocalendar().week.astype(int)
    df["ceyrek"]          = df["date"].dt.quarter
    df["is_weekend"]           = (df["haftanıngünleri"] >= 5).astype(int)
    df["weekend_x_saat"]       = df["is_weekend"] * df["saat"]
    return df

def add_generation_mix(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    mevcut_yenilenebilir = [c for c in GEN_COLS_YENILENEBILIR if c in df.columns]
    mevcut_termik        = [c for c in GEN_COLS_TERMIK        if c in df.columns]
    mevcut_tumu          = [c for c in GEN_COLS_TUMÜ          if c in df.columns]

    if mevcut_tumu:
        df["toplam_uretim"] = df[mevcut_tumu].sum(axis=1)
        denom = df["toplam_uretim"].clip(lower=1)

        if mevcut_yenilenebilir:
            df["yenilenebilir_oran"] = df[mevcut_yenilenebilir].sum(axis=1) / denom
        if mevcut_termik:
            df["termik_oran"] = df[mevcut_termik].sum(axis=1) / denom
        if "consumption" in df.columns:
            df["net_ithalat"] = df["consumption"] - df["toplam_uretim"]

    return df

def add_weather_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hissedilen_C" in df.columns:
        df["hissedilen_C2"] = df["hissedilen_C"] ** 2
    if "sicaklik_C" in df.columns:
        df["sicaklik_lag24_delta"] = df["sicaklik_C"] - df["sicaklik_C"].shift(24)
    if "gunes_radyasyon_Wm2" in df.columns and "bulutluluk_pct" in df.columns:
        df["gunes_efektif"] = (
            df["gunes_radyasyon_Wm2"] * (1 - df["bulutluluk_pct"] / 100)
        )
    return df

def add_holiday_proximity(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """Tatil öncesi ve sonrası günleri işaretler — fiyat profili çok farklı."""
    df = df.copy()
    tatil_oncesi = {d - timedelta(days=1) for d in holiday_dates}
    tatil_sonrasi = {d + timedelta(days=1) for d in holiday_dates}
    dates = pd.to_datetime(df["date"]).dt.date
    df["tatil_oncesi_gun"]  = dates.isin(tatil_oncesi).astype(int)
    df["tatil_sonrasi_gun"] = dates.isin(tatil_sonrasi).astype(int)
    return df


def add_lags(df: pd.DataFrame, lag_map: dict) -> pd.DataFrame:
    df = df.copy()
    for col, lags in lag_map.items():
        if col not in df.columns:
            print(f"Uyari: lag için '{col}' kolonu yok, atlanıyor.")
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_shifted = df["price"].shift(1)

    for window in [6, 24, 168]:
        df[f"price_rolling_mean_{window}h"] = price_shifted.rolling(window, min_periods=1).mean()
        df[f"price_rolling_std_{window}h"]  = price_shifted.rolling(window, min_periods=2).std()

    df["price_rolling_max_24h"] = price_shifted.rolling(24, min_periods=1).max()
    df["price_rolling_min_24h"] = price_shifted.rolling(24, min_periods=1).min()

    df["price_return_1h"]  = df["price"].shift(1).pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["price_return_24h"] = df["price"].shift(1).pct_change(24).replace([np.inf, -np.inf], np.nan)

    if "consumption" in df.columns:
        cons_shifted = df["consumption"].shift(1)
        df["consumption_rolling_mean_24h"] = cons_shifted.rolling(24, min_periods=1).mean()

    return df


def fill_price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tahmin döngüsünde NaN kalan return feature'larını güvenli doldurur."""
    df = df.copy()
    eps = 1e-6

    if "price_return_1h" in df.columns:
        prev_1 = df["price"].shift(1)
        prev_2 = df["price"].shift(2)
        safe_base_1h = prev_2.where(prev_2.abs() > eps, np.nan)
        df["price_return_1h"] = ((prev_1 / safe_base_1h) - 1).replace([np.inf, -np.inf], np.nan)

    if "price_return_24h" in df.columns:
        prev_1 = df["price"].shift(1)
        prev_25 = df["price"].shift(25)
        safe_base_24h = prev_25.where(prev_25.abs() > eps, np.nan)
        df["price_return_24h"] = ((prev_1 / safe_base_24h) - 1).replace([np.inf, -np.inf], np.nan)

    # Gelecek saatlerde zorunlu feature boş kalırsa nötr etki için 0'a çek.
    for c in ("price_return_1h", "price_return_24h"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


def add_ewm(df: pd.DataFrame, col: str, spans: list[int]) -> pd.DataFrame:
    df = df.copy()
    for s in spans:
        df[f"{col}_ewm_{s}"] = df[col].ewm(span=s, adjust=False).mean()
    return df


def resolve_lgbm_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        try:
            test = lgb.LGBMRegressor(device="gpu", n_estimators=1, verbose=-1)
            test.fit([[0.0], [1.0]], [0.0, 1.0])
        except Exception as e:
            print(f"Warning: GPU kullanılamıyor ({e!r}), CPU kullanılıyor.")
            return "cpu"
        return "gpu"
    try:
        test = lgb.LGBMRegressor(device="gpu", n_estimators=1, verbose=-1)
        test.fit([[0.0], [1.0]], [0.0, 1.0])
        return "gpu"
    except Exception:
        return "cpu"

def build_lgbm_params(device: str) -> dict:
    return {
        "colsample_bytree":  0.8,
        "learning_rate":     0.03,
        "max_depth":         6,
        "min_child_samples": 50,
        "num_leaves":        63,
        "subsample":         0.8,
        "subsample_freq":    1,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      42,
        "device":            device,
        "n_estimators":      10000,
        "verbose":           -1,
    }

def main() -> int:
    load_env_optional()
    args = parse_args()

    if plt is None and not args.no_plots:
        print("matplotlib yok; grafikler atlanıyor.", file=sys.stderr)
        args.no_plots = True

    username, password = load_credentials()
    tgt = get_tgt(username, password)

    df_ptf        = fetch_epias_data(PTF_URL,          args.start_date, args.end_date, tgt, "PTF")
    df_consumption = fetch_epias_data(CONSUMPTION_URL, args.start_date, args.end_date, tgt, "Tüketim")
    df_kgup       = fetch_kgup(args.start_date, args.end_date, tgt)

    validate_merge_frames(df_ptf, df_consumption, df_kgup)

    df_merged = df_ptf.merge(df_consumption, on="date", how="left").merge(df_kgup, on="date", how="left")
    df_merged = safe_drop_columns(df_merged, DROP_COLUMNS)
    df_merged = normalize_consumption_column(df_merged)
    df_merged = safe_drop_columns(df_merged, ["lep"])

    df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")
    if df_merged["date"].dt.tz is not None:
        df_merged["date"] = df_merged["date"].dt.tz_convert(METEO_TIMEZONE).dt.tz_localize(None)

    if not args.no_meteo:
        meteo_hist = get_meteo_for_epias_merged(args.start_date, args.end_date, args.meteo_lat, args.meteo_lon)
        df_merged  = merge_meteo_hourly(df_merged, meteo_hist)

    df_merged = create_date_features(df_merged)

    years_holiday = sorted(pd.to_datetime(df_merged["date"]).dt.year.dropna().unique().tolist())
    tr_holidays   = holidays.Turkey(years=years_holiday or [datetime.now().year])
    holiday_dates = set(tr_holidays.keys())

    df_merged["resmi_tatil"] = np.where(
        pd.to_datetime(df_merged["date"]).dt.date.isin(holiday_dates), 1, 0
    )

    df_merged = add_holiday_proximity(df_merged, holiday_dates)

    df_merged = safe_drop_columns(df_merged, ["haftanınGünü"])

    df_merged = add_generation_mix(df_merged)

    df_merged = add_weather_derived(df_merged)

    lag_map   = build_lag_map(df_merged)
    df_merged = add_lags(df_merged, lag_map)

    df_merged = add_rolling_features(df_merged)

    lag_and_rolling_cols = [c for c in df_merged.columns if "_lag_" in c or "_rolling_" in c]
    df_merged = df_merged.dropna(subset=lag_and_rolling_cols).reset_index(drop=True)

    df_merged = add_ewm(df_merged, "price", spans=[24, 168])

    df_merged["sin_hour"]         = np.sin(2 * np.pi * df_merged["saat"] / 24)
    df_merged["cos_hour"]         = np.cos(2 * np.pi * df_merged["saat"] / 24)
    df_merged["sin_month"]        = np.sin(2 * np.pi * df_merged["ay"] / 12)
    df_merged["cos_month"]        = np.cos(2 * np.pi * df_merged["ay"] / 12)
    df_merged["sin_week"]         = np.sin(2 * np.pi * df_merged["haftanıngünleri"] / 7)
    df_merged["cos_week"]         = np.cos(2 * np.pi * df_merged["haftanıngünleri"] / 7)
    df_merged["sin_day_of_month"] = np.sin(2 * np.pi * df_merged["gün"] / 31)
    df_merged["cos_day_of_month"] = np.cos(2 * np.pi * df_merged["gün"] / 31)

    if args.log_target:
        df_merged["price"] = np.log1p(df_merged["price"])
        print("Log transform uygulandı: price = log(price + 1)")

    train_df = df_merged[df_merged["date"] < args.train_before]
    test_df  = df_merged[df_merged["date"] >= args.train_before]

    if train_df.empty:
        raise ValueError("Egitim kumesi bos: --train-before ve tarih araligini kontrol edin.")
    if test_df.empty:
        print("Uyari: test kumesi bos.")

    cols   = [col for col in train_df.columns if col not in ["date", "price"]]
    Y_train = train_df["price"]
    X_train = train_df[cols]
    Y_test  = test_df["price"]
    X_test  = test_df[cols]

    val_size = max(24, int(len(train_df) * args.val_ratio))
    X_val = X_train.iloc[-val_size:]
    Y_val = Y_train.iloc[-val_size:]
    X_tr  = X_train.iloc[:-val_size]
    Y_tr  = Y_train.iloc[:-val_size]
    print(f"Egitim: {len(X_tr)} satir | Validation: {len(X_val)} satir | Test: {len(X_test)} satir")

    lgbm_device = resolve_lgbm_device(args.lgbm_device)
    print(f"LightGBM device: {lgbm_device}")

    best_params = build_lgbm_params(lgbm_device)
    model = lgb.LGBMRegressor(**best_params)

    model.fit(
        X_tr, Y_tr,
        eval_set=[(X_val, Y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=500),
        ],
    )
    print(f"En iyi ağaç sayısı: {model.best_iteration_}")

    y_pred_raw = model.predict(X_test)

    if args.log_target:
        y_pred = np.expm1(y_pred_raw)
        Y_test_orig = np.expm1(Y_test)
    else:
        y_pred      = y_pred_raw
        Y_test_orig = Y_test

    mae  = mean_absolute_error(Y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test_orig, y_pred))
    print("TEST MAE:", mae)
    print("TEST RMSE:", rmse)

    df_test = test_df[["date"]].copy()
    df_test["date"]   = pd.to_datetime(df_test["date"])
    df_test["actual"] = Y_test_orig.values
    df_test["pred"]   = y_pred

    if not args.no_plots:
        daily = df_test.set_index("date")[["actual", "pred"]].resample("D").mean()
        plt.figure(figsize=(12, 5))
        plt.plot(daily.index, daily["actual"], label="Actual",    linewidth=2, color="black")
        plt.plot(daily.index, daily["pred"],   label="Predicted", linewidth=2, color="tab:blue")
        plt.title("Daily Average Price: Actual vs Predicted")
        plt.ylabel("Price (TL)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(df_test["actual"], df_test["pred"], alpha=0.25)
        plt.plot(
            [df_test["actual"].min(), df_test["actual"].max()],
            [df_test["actual"].min(), df_test["actual"].max()],
            linestyle="--", color="black",
        )
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Predicted vs Actual")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        df_test_plot        = df_test.copy()
        df_test_plot["hour"]      = df_test_plot["date"].dt.hour
        df_test_plot["abs_error"] = (df_test_plot["actual"] - df_test_plot["pred"]).abs()
        hourly_mae = df_test_plot.groupby("hour")["abs_error"].mean()
        plt.figure(figsize=(10, 4))
        plt.bar(hourly_mae.index, hourly_mae.values)
        plt.xlabel("Hour of Day")
        plt.ylabel("MAE")
        plt.title("Mean Absolute Error by Hour")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    mape = (
        np.mean(
            np.abs((df_test["actual"] - df_test["pred"]) / np.clip(np.abs(df_test["actual"]), 1e-6, None))
        ) * 100
    )
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE (%)"],
        "Value":  [mae, rmse, mape],
    })
    print(metrics_df)

    feature_importance = (
        pd.DataFrame({"feature": X_tr.columns, "importance": model.feature_importances_})
        .sort_values(by="importance", ascending=False)
    )
    if not args.no_plots:
        top_n = 20
        plt.figure(figsize=(8, 7))
        plt.barh(
            feature_importance["feature"][:top_n][::-1],
            feature_importance["importance"][:top_n][::-1],
        )
        plt.title("Top Feature Importances (LightGBM)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    forecast_path = args.forecast_input.expanduser().resolve()
    if not forecast_path.is_file():
        raise FileNotFoundError(f"Tahmin girdisi bulunamadi: {forecast_path}")

    df_future = pd.read_excel(forecast_path)
    rename_map = {
        "Tarih": "date",
        "Doğalgaz": "dogalgaz", "Rüzgar": "ruzgar", "Linyit": "linyit",
        "Taş Kömür": "tasKomur", "İthal Kömür": "ithalKomur",
        "Fueloil": "fuelOil", "Jeotermal": "jeotermal",
        "Barajlı": "barajli", "Nafta": "nafta", "Biyokütle": "biokutle",
        "Akarsu": "akarsu", "Gunes": "gunes", "Güneş": "gunes",
        "Diğer": "diger", "Diger": "diger",
        "Tüketim Miktarı(MWh)": "consumption", "Tüketim Miktar(MWh)": "consumption",
        "Tuketim Miktari(MWh)": "consumption", "Tuketim Miktar(MWh)": "consumption",
    }
    df_future = df_future.rename(columns=rename_map)
    df_future["date"]  = pd.to_datetime(df_future["date"], dayfirst=True, errors="coerce")
    df_future["price"] = np.nan

    df_merged = df_merged.copy()
    df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")
    if df_merged["date"].dt.tz is not None:
        df_merged["date"] = df_merged["date"].dt.tz_convert(METEO_TIMEZONE).dt.tz_localize(None)
    if df_future["date"].dt.tz is not None:
        df_future["date"] = df_future["date"].dt.tz_convert(METEO_TIMEZONE).dt.tz_localize(None)

    if not args.no_meteo:
        d_min = df_future["date"].min()
        d_max = df_future["date"].max()
        if pd.notna(d_min) and pd.notna(d_max):
            meteo_f   = get_meteo_for_forecast_excel(d_min, d_max, args.meteo_lat, args.meteo_lon)
            df_future = merge_meteo_hourly(df_future, meteo_f)

    df_future = df_future.reindex(columns=df_merged.columns, fill_value=np.nan)

    df_all = (
        pd.concat([df_merged, df_future], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    dt = df_all["date"]
    df_all["ay"]              = dt.dt.month
    df_all["gün"]             = dt.dt.day
    df_all["saat"]            = dt.dt.hour
    df_all["haftanıngünleri"] = dt.dt.dayofweek
    df_all["yilin_haftasi"]   = dt.dt.isocalendar().week.astype(int)
    df_all["ceyrek"]          = dt.dt.quarter
    df_all["is_weekend"]      = (df_all["haftanıngünleri"] >= 5).astype(int)
    df_all["weekend_x_saat"]  = df_all["is_weekend"] * df_all["saat"]

    years     = sorted(dt.dt.year.dropna().unique().tolist())
    tr_h      = holidays.Turkey(years=years)
    hol_dates = set(tr_h.keys())
    df_all["resmi_tatil"]      = dt.dt.date.isin(hol_dates).astype(int)
    df_all = add_holiday_proximity(df_all, hol_dates)

    df_all["sin_hour"]         = np.sin(2 * np.pi * df_all["saat"] / 24)
    df_all["cos_hour"]         = np.cos(2 * np.pi * df_all["saat"] / 24)
    df_all["sin_month"]        = np.sin(2 * np.pi * df_all["ay"] / 12)
    df_all["cos_month"]        = np.cos(2 * np.pi * df_all["ay"] / 12)
    df_all["sin_week"]         = np.sin(2 * np.pi * df_all["haftanıngünleri"] / 7)
    df_all["cos_week"]         = np.cos(2 * np.pi * df_all["haftanıngünleri"] / 7)
    df_all["sin_day_of_month"] = np.sin(2 * np.pi * df_all["gün"] / 31)
    df_all["cos_day_of_month"] = np.cos(2 * np.pi * df_all["gün"] / 31)

    df_all = add_generation_mix(df_all)
    df_all = add_weather_derived(df_all)

    future_idx = df_all.index[df_all["price"].isna()].tolist()

    for i in future_idx:
        df_all = add_lags(df_all, lag_map)
        df_all = add_rolling_features(df_all)
        df_all = fill_price_return_features(df_all)
        df_all = add_ewm(df_all, "price", spans=[24, 168])

        X_row = df_all.loc[[i], cols]

        if X_row.isna().any().any():
            missing = X_row.columns[X_row.isna().any()].tolist()
            raise ValueError(f"{df_all.loc[i, 'date']} icin eksik feature: {missing}")

        pred_raw = float(model.predict(X_row)[0])

        if args.log_target:
            pred = np.expm1(pred_raw)
        else:
            pred = pred_raw

        pred = float(np.clip(pred, 0, 4500))
        df_all.at[i, "price"] = np.log1p(pred) if args.log_target else pred

    forecast_df = df_all.loc[future_idx, ["date", "price"]].copy()
    if args.log_target:
        forecast_df["price"] = np.expm1(forecast_df["price"])
    forecast_df = forecast_df.rename(columns={"price": "ptf_tahmin"})

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_excel(out_path, index=False)
    print("Tahmin dosyasi kaydedildi:", out_path)

    if args.debug_save:
        if args.debug_dir is not None:
            dbg = args.debug_dir.expanduser().resolve()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dbg = _desktop_path() / f"PTF_DEBUG_{stamp}"
        dbg.mkdir(parents=True, exist_ok=True)
        df_merged.to_excel(dbg / "df_merged.xlsx", index=False)
        df_future.to_excel(dbg / "df_future.xlsx", index=False)
        df_all.to_excel(dbg / "df_all.xlsx", index=False)
        forecast_df.to_excel(dbg / "forecast_df.xlsx", index=False)
        print("Debug dosyalari:", dbg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
