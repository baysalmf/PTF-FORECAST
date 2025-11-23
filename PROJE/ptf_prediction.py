import pandas as pd
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)

username = ""
password = ""

start_date = "2024-10-28"
end_date = "2025-11-22"

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

df_merged = df_ptf.merge(df_consumption, on="date", how="left").merge(df_kgup, on="date", how="left")

df_merged.drop(['hour', 'priceUsd', 'priceEur', 'time_x', 'time_y', 'toplam'], axis=1, inplace=True)

df_merged['date'] = pd.to_datetime(df_merged['date'])

def create_date_features(dataframe):
    dataframe['ay'] = dataframe['date'].dt.month
    dataframe['gün'] = dataframe['date'].dt.day
    dataframe['saat'] = dataframe['date'].dt.hour
    dataframe['haftanıngünleri'] = dataframe['date'].dt.dayofweek
    dataframe['mevsim'] = dataframe['date'].dt.quarter
    dataframe['haftanınGünü'] = dataframe['date'].dt.day_name()
    return dataframe

create_date_features(df_merged)

cal = Turkey()
all_holidays = []

for year in range(2022, 2025):
    holidays = cal.holidays(year)
    for date, name in holidays:
        all_holidays.append(date.isoformat())

all_holidays = pd.to_datetime(all_holidays).date

df['resmi_tatil'] = np.where(pd.to_datetime(df['date']).dt.date.isin(all_holidays), 0, 1)

df.drop('haftanınGünü', axis=1, inplace=True)

plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['price'], linewidth=0.6)
plt.title('Saatlik PTF Grafiği')
plt.xlabel('Tarih')
plt.ylabel('Piyasa Takas Fiyatı (PTF)')
plt.grid(False)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16, 6)
sns.boxplot(x='saat', y='price', data=df, ax=ax)
plt.title('Saatlik Periyotta PTF')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches((16, 6))
sns.boxplot(x='ay', y='price', data=df, ax=ax)
plt.title('Aylık Periyotta PTF')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches((16, 6))
sns.boxplot(x='haftanıngünleri', y='price', data=df, ax=ax)
plt.title('Haftalık periyotta PTF')
plt.tight_layout()
plt.show()

sns.set(rc={'figure.figsize': (30, 20)})
sns.lineplot(x=df.index, y='price', hue=df.yıl, data=df, color='black').set_title('PTF')
plt.plot(df.price.rolling(24 * 30).mean(), alpha=1, color='red', label='Hareketli Ortalama 1-aylık')
plt.plot(df.price.rolling(24 * 30 * 3).mean(), alpha=1, color='green', label='Hareketli Ortalama 3-aylık')
plt.legend()
plt.show()

df2 = df.pivot_table(index=df['saat'], columns='haftanıngünleri', values='price', aggfunc='mean')
df2.plot(figsize=(16, 6), title='Günlük Ortalama PTF')
plt.tight_layout()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

result = seasonal_decompose(df['price'], period=48)
rcParams['figure.figsize'] = 12, 8
fig = result.plot()
plt.show()

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF

rcParams['figure.figsize'] = 12, 4
plot_acf(df['price'], lags=24)
plt.tick_params(axis='both', labelsize=12)
plt.show()

#PCF

rcParams['figure.figsize'] = 12, 4
plot_pacf(df['price'], lags=24)
plt.tick_params(axis='both', labelsize=12)
plt.show()

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

y = df['price']
# Dickey-Fuller test:
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=24).mean()
    rolstd = timeseries.rolling(window=24).std()

    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, color='blue', label='Original', linewidth=0.8)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=0.8)
    plt.plot(rolstd, color='black', label='Rolling Std', linewidth=0.8)
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.show()


test_stationarity(y)


def is_stationary(y):
    p_value = sm.tsa.stattools.adfuller(y)[1]

    if p_value < 0.05:
        print(F'Result : Stationary (H0: non-stationary, p_value {p_value:.10f})')

    else:
        print(F'Result: Non-Stationary (H0: non-stationary, p_value {p_value:.10f})')


is_stationary(y)


# Model

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

df.drop(['yıl', 'saat', 'mevsim', 'yılıngünü'], axis=1, inplace=True)

train_size = int(len(df) * 0.95)
train, test = df[:train_size], df[train_size:]

exogenous_vars_train = train[['ay', 'gün', 'haftanıngünleri', 'resmi_tatil']]
exogenous_vars_test = test[['ay', 'gün', 'haftanıngünleri', 'resmi_tatil']]

'''
SARIMAX_model = pm.auto_arima(train['price'],
                              exogenous=exogenous_vars_train,
                              start_p=1,
                              start_q=1,
                              test='adf',
                              max_p=3,
                              max_q=3,
                              start_P=0,
                              start_Q=0,
                              max_P=3,
                              max_Q=3,
                              m=24,
                              seasonal=True,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

print(SARIMAX_model.summary())
'''

model = SARIMAX(train['price'],
                exog=exogenous_vars_train,
                order=(1, 0, 0),
                seasonal_order=(2, 0, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)
print(model_fit.summary())

predictions = model_fit.predict(start=test.index[0], end=test.index[-1], exog=exogenous_vars_test)
predictions = np.where(predictions < 0, 0, predictions)
predictions = np.where(predictions > 3000, 3000, predictions)

true_values = test['price']

mae = mean_absolute_error(true_values, predictions)
mse = mean_squared_error(true_values, predictions)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

comparison_df = pd.DataFrame({'Gerçek Değerler': true_values, 'Tahminler': predictions})
print(comparison_df.tail(24))

plt.figure(figsize=(10, 6))
plt.plot(test.index, true_values, label='Gerçek Değerler', color='blue')
plt.plot(test.index, predictions, label='Tahminler', color='red', linestyle='--')

plt.title('Gerçek Değerler ve Tahminler Karşılaştırması')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()




