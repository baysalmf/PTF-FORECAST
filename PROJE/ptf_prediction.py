import pandas as pd
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta

username = "muhammetfurkan.baysal@zorlu.com"
password = "Zorlu.2025"

start_date = "2025-01-01"
end_date = "2025-12-24"

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
PTF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
CONSUMPTION_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"
KGUP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/dpp"


def to_iso_tr(d):
    return f"{d}T00:00:00+03:00"


def get_tgt():
    r = requests.post(
        CAS_URL,
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        timeout=40
    )
    if r.status_code != 201:
        raise SystemExit(f"TGT alınamadı: {r.status_code}-{r.text}")
    return r.headers.get("Location", "").rstrip("/").split("/")[-1]

def fetch_epias_data(url, start_date, end_date, tgt, label):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    part_start = start
    all_data = []

    while part_start <= end:
        part_end = (part_start + relativedelta(months=3)) - relativedelta(days=1)
        if part_end > end:
            part_end = end

        body = {
            "startDate": to_iso_tr(part_start.strftime("%Y-%m-%d")),
            "endDate": to_iso_tr(part_end.strftime("%Y-%m-%d"))
        }

        r = requests.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json", "TGT": tgt},
            timeout=60
        )

        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                all_data.extend(items)
                print(f"{label}: {part_start.date()} - {part_end.date()} arası {len(items)} kayıt çekildi.")
            else:
                print(f"{label}: {part_start.date()} - {part_end.date()} arası veri yok.")
        else:
            print(f"{label} Hata ({r.status_code}) → {r.text}")

        part_start = part_end + relativedelta(days=1)

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def fetch_kgup(start_date, end_date, tgt):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    part_start = start
    all_data = []

    while part_start <= end:
        part_end = (part_start + relativedelta(months=3)) - relativedelta(days=1)
        if part_end > end:
            part_end = end

        body = {
            "startDate": to_iso_tr(part_start.strftime("%Y-%m-%d")),
            "endDate": to_iso_tr(part_end.strftime("%Y-%m-%d")),
            "region": "TR1"
        }

        r = requests.post(
            KGUP_URL,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json", "TGT": tgt},
            timeout=60
        )

        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                all_data.extend(items)
                print(f"KGÜP: {part_start.date()} - {part_end.date()} arası {len(items)} kayıt çekildi.")
            else:
                print(f"KGÜP: {part_start.date()} - {part_end.date()} arası veri yok.")
        else:
            print(f"KGÜP Hata ({r.status_code}) → {r.text}")

        part_start = part_end + relativedelta(days=1)

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

tgt = get_tgt()

df_ptf = fetch_epias_data(PTF_URL, start_date, end_date, tgt, "PTF")
df_consumption = fetch_epias_data(CONSUMPTION_URL, start_date, end_date, tgt, "Tüketim")
df_kgup = fetch_kgup(start_date, end_date, tgt)

"""

df_ghı = pd.read_csv(r"C:\Users\muhammetfb\OneDrive\Masaüstü\POWER_Point_Hourly_GHI_2025.csv",
    skiprows=lambda x: x < 9,
    sep=",")

df_temp = pd.read_csv(r"C:\Users\muhammetfb\OneDrive\Masaüstü\POWER_Point_Hourly.csv",
    skiprows=lambda x: x < 9,
    sep=",")

"""

df_merged = df_ptf.merge(df_consumption, on="date", how="left").merge(df_kgup, on="date", how="left")

df_merged.drop(['hour', 'priceUsd', 'priceEur', 'time_x', 'time_y', 'toplam'], axis=1, inplace=True)

df_merged['date'] = pd.to_datetime(df_merged['date'])

def create_date_features(dataframe):
    dataframe['ay'] = dataframe['date'].dt.month
    dataframe['gün'] = dataframe['date'].dt.day
    dataframe['saat'] = dataframe['date'].dt.hour
    dataframe['haftanıngünleri'] = dataframe['date'].dt.dayofweek
    dataframe['haftanınGünü'] = dataframe['date'].dt.day_name()
    return dataframe

create_date_features(df_merged)

import holidays
import numpy as np

tr_holidays = holidays.Turkey(years=[2025])

holiday_dates = set(tr_holidays.keys())

df_merged['resmi_tatil'] = np.where(
    pd.to_datetime(df_merged['date']).dt.date.isin(holiday_dates),
    1,
    0)

df_merged.drop('haftanınGünü', axis=1, inplace=True)

# Model
def add_lags(df, lag_map):

    df = df.copy()
    for col, lags in lag_map.items():
        if col not in df.columns:
            raise KeyError(f"Kolon bulunamadı: {col}. Mevcut kolonlar: {list(df.columns)}")
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

lag_map = {
    "price": [1, 2, 3, 24, 48, 72, 168],
    "consumption": [24, 168],
    "ruzgar": [24],
    "gunes": [24],
    "dogalgaz": [24, 168],
}

df_merged = add_lags(df_merged, lag_map)

lag_cols = [c for c in df_merged.columns if "_lag_" in c]
df_merged = df_merged.dropna(subset=lag_cols).reset_index(drop=True)

def add_ewm(df, col, spans):
    df = df.copy()
    for s in spans:
        df[f"{col}_ewm_{s}"] = df[col].ewm(span=s, adjust=False).mean()
    return df

df_merged = add_ewm(df_merged, "price", spans=[24, 168])

df_merged["sin_hour"] = np.sin(2 * np.pi * df_merged["saat"] / 24)
df_merged["cos_hour"] = np.cos(2 * np.pi * df_merged["saat"] / 24)

df_merged["sin_month"] = np.sin(2 * np.pi * df_merged["ay"] / 12)
df_merged["cos_month"] = np.cos(2 * np.pi * df_merged["ay"] / 12)

df_merged["sin_week"] = np.sin(2 * np.pi * df_merged["haftanıngünleri"] / 7)
df_merged["cos_week"] = np.cos(2 * np.pi * df_merged["haftanıngünleri"] / 7)

df_merged["sin_day_of_month"] = np.sin(2 * np.pi * df_merged["gün"] / 31)
df_merged["cos_day_of_month"] = np.cos(2 * np.pi * df_merged["gün"] / 31)

train_df = df_merged[df_merged["date"] < "2025-10-01"]
test_df = df_merged[df_merged["date"] >= "2025-10-01"]

cols = [col for col in train_df.columns if col not in ['date', 'price']]

Y_train = train_df['price']
X_train = train_df[cols]

Y_test = test_df['price']
X_test = test_df[cols]

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

lgb_regressor = lgb.LGBMRegressor(device="gpu", random_state=42)

param_grid = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [-1, 8, 12],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "subsample": [0.8, 0.9, 1.0],
    "min_child_samples": [20, 50, 100],
}

tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    estimator=lgb_regressor,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)
print("Best CV MAE:", -grid_search.best_score_)

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

print("TEST MAE:", mae)
print("TEST RMSE:", rmse)

best_params = {
    "colsample_bytree": 0.9,
    "learning_rate": 0.1,
    "max_depth": 8,
    "min_child_samples": 100,
    "num_leaves": 127,
    "subsample": 0.8,
    "random_state": 42,
    "device": "gpu",
    "n_estimators": 5000
}

import matplotlib.pyplot as plt

df_test = test_df[["date"]].copy()
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["actual"] = Y_test.values
df_test["pred"] = y_pred

daily = (
    df_test
    .set_index("date")[["actual", "pred"]]
    .resample("D")
    .mean()
)

plt.figure(figsize=(12,5))
plt.plot(daily.index, daily["actual"], label="Actual", linewidth=2, color="black")
plt.plot(daily.index, daily["pred"], label="Predicted", linewidth=2, color="tab:blue")
plt.title("Daily Average Price: Actual vs Predicted")
plt.ylabel("Price")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(df_test["actual"], df_test["pred"], alpha=0.25)
plt.plot(
    [df_test["actual"].min(), df_test["actual"].max()],
    [df_test["actual"].min(), df_test["actual"].max()],
    linestyle="--",
    color="black"
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Prices")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

df_test["hour"] = df_test["date"].dt.hour
df_test["abs_error"] = (df_test["actual"] - df_test["pred"]).abs()

hourly_mae = df_test.groupby("hour")["abs_error"].mean()

plt.figure(figsize=(10,4))
plt.bar(hourly_mae.index, hourly_mae.values)
plt.xlabel("Hour of Day")
plt.ylabel("MAE")
plt.title("Mean Absolute Error by Hour")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

mae = mean_absolute_error(df_test["actual"], df_test["pred"])
rmse = np.sqrt(mean_squared_error(df_test["actual"], df_test["pred"]))

mape = np.mean(
    np.abs((df_test["actual"] - df_test["pred"]) /
           np.clip(np.abs(df_test["actual"]), 1e-6, None))
) * 100

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE (%)"],
    "Value": [mae, rmse, mape]
})

print(metrics_df)

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance.head(15)
top_n = 15

plt.figure(figsize=(8,6))
plt.barh(
    feature_importance["feature"][:top_n][::-1],
    feature_importance["importance"][:top_n][::-1]
)
plt.title("Top Feature Importances (LightGBM)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Gelecek Tahmin

feature_cols = ['consumption', 'dogalgaz', 'ruzgar', 'linyit', 'tasKomur', 'ithalKomur', 'fuelOil',
                'jeotermal', 'barajli', 'nafta', 'biokutle', 'akarsu', 'gunes', 'diger', 'ay', 'gün',
                'saat', 'haftanıngünleri', 'resmi_tatil', 'price_lag_1', 'price_lag_2', 'price_lag_3',
                'price_lag_24', 'price_lag_48', 'price_lag_72', 'price_lag_168', 'consumption_lag_24',
                'consumption_lag_168', 'ruzgar_lag_24', 'gunes_lag_24', 'dogalgaz_lag_24',
                'dogalgaz_lag_168', 'price_ewm_24', 'price_ewm_168', 'sin_hour', 'cos_hour', 'sin_month',
                'cos_month', 'sin_week', 'cos_week', 'sin_day_of_month', 'cos_day_of_month']

price_lags = [1, 2, 3, 24, 48, 72, 168]

df_future = pd.read_excel(r"C:\Users\muhammetfb\OneDrive\Masaüstü\tahmin.xlsx")

df_future = df_future.drop(columns=["Toplam(MWh)"], errors="ignore")

rename_map = {
    "Tarih": "date",
    "Doğalgaz": "dogalgaz",
    "Rüzgar": "ruzgar",
    "Linyit": "linyit",
    "Taş Kömür": "tasKomur",
    "İthal Kömür": "ithalKomur",
    "Fueloil": "fuelOil",
    "Jeotermal": "jeotermal",
    "Barajlı": "barajli",
    "Nafta": "nafta",
    "Biyokütle": "biokutle",
    "Akarsu": "akarsu",
    "Gunes": "gunes",
    "Diğer": "diger",
    "Tüketim Miktarı(MWh)": "consumption",
}
df_future = df_future.rename(columns=rename_map)

df_future["date"] = pd.to_datetime(df_future["date"])
df_future["price"] = np.nan

df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")
if df_merged["date"].dt.tz is not None:
    df_merged["date"] = df_merged["date"].dt.tz_localize(None)

df_future = df_future.drop(columns=["Saat", "Toplam(MWh)"], errors="ignore")

df_future["date"] = pd.to_datetime(df_future["date"], errors="coerce")

if "price" not in df_future.columns:
    df_future["price"] = np.nan

df_future = df_future.reindex(columns=df_merged.columns, fill_value=np.nan)

df_all = (pd.concat([df_merged, df_future], ignore_index=True).sort_values("date").reset_index(drop=True)


