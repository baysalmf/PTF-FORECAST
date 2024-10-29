import json
import requests
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
from workalendar.europe import Turkey

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)

username = ''# *** username is required
password = ''# *** password is required
start_date = '2023-10-28'#*** start date is required
end_date = '2024-10-28'#*** end date is required
url1 = f"https://giris.epias.com.tr/cas/v1/tickets?username={username}&password={password}"
url2 = f"https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp?username={username}&password={password}"


def date_converter(date_string):
    new_string = date_string + "T00:00:00+03:00"
    return new_string


def data_tgt(username, password):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'text/plain'
    }
    response = requests.request("POST", url1, headers=headers, timeout=40)

    if response.status_code == 201:
        tgt_key = response.text
        return tgt_key


def data_mcp(start_date, end_date, tgt_key, username, password):
    sd = date_converter(start_date)
    ed = date_converter(end_date)

    payload = json.dumps({
        'endDate': ed,
        'startDate': sd
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'TGT': tgt_key
    }
    response = requests.post(url2, headers=headers, data=payload)

    if response.status_code == 200:
        df = pd.json_normalize(response.json()['items'])
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        return response.text


tgt = data_tgt(username, password)

df = data_mcp(start_date, end_date, tgt, username, password)

df.drop(['hour', 'priceUsd', 'priceEur'], axis=1, inplace=True)


def create_date_features(dataframe):
    dataframe['yıl'] = dataframe['date'].dt.year
    dataframe['ay'] = dataframe['date'].dt.month
    dataframe['gün'] = dataframe['date'].dt.day
    dataframe['saat'] = dataframe['date'].dt.hour
    dataframe['haftanıngünleri'] = dataframe['date'].dt.dayofweek
    dataframe['mevsim'] = dataframe['date'].dt.quarter
    dataframe['haftanınGünü'] = dataframe['date'].dt.day_name()
    dataframe['yılıngünü'] = dataframe['date'].dt.dayofyear
    return dataframe


create_date_features(df)


cal = Turkey()
all_holidays = []

for year in range(2022, 2024):
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





