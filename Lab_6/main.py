import pandas as pd
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.graphics.tsaplots as tsaplots
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import numpy as np

if __name__=="__main__":
    pd.set_option('display.max_columns', None)

    data=pd.read_csv("data/rozrobka-softu.csv",index_col=['Дата'], parse_dates=['Дата'],encoding="utf-8")
    print(data)

    data.drop(columns=["дельта"], inplace=True)
    data.rename_axis("date", inplace=True)
    data.rename(columns={"Популярні мови програмування згідно кількості вакансій в Україні":"vacancies"}, inplace=True)

    print(data)
    print(data.describe())

    print(f"\nCount of missing values: {data['vacancies'].isna().sum()}")

    fig, ax = plt.subplots(figsize=(15, 10))
    data.plot(ax=ax)
    plt.legend(["Vacancies"])
    ax.grid()
    plt.show() #zeros detected

    for index, row in data.iterrows():
        if row['vacancies'] == 0:
            start_idx = max(data.index.get_loc(index) - 20, 0)
            rolling_avg = data.loc[data.index[start_idx]:index, 'vacancies'].mean()
            data.at[index, 'vacancies'] = rolling_avg

    fig, ax = plt.subplots(figsize=(15, 10))
    data.plot(ax=ax)
    plt.legend(["Vacancies"])
    ax.grid()
    plt.show() #better

    # decompose to basic layers
    decomp = smt.seasonal_decompose(data, model='additive')
    decomp.plot()
    plt.show()

    seasonal_substracted = decomp.trend+decomp.resid
    seasonal_substracted.plot()
    plt.show()

    #creating arima

    time_difference = data.index - data.index.min()
    threshold=(data.index.max() - data.index.min()).days*0.8

    data_train=data[time_difference.days < threshold]
    data_test = data[time_difference.days >= threshold]

    #picking parameters

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(data["vacancies"])
    ax1.set_title('Original Series')
    ax1.axes.xaxis.set_visible(False)

    ax2.plot(data["vacancies"].diff())
    ax2.set_title('1st Order Differencing')
    ax2.axes.xaxis.set_visible(False)
    plt.show()#becomes stationary - d = 1

    adfuller_test = adfuller(data["vacancies"].diff().dropna())
    print('ADF Statistic: %f' % adfuller_test[0])
    print('p-value: %f' % adfuller_test[1])
    print('Critical Values:')
    for key, value in adfuller_test[4].items():
        print('\t%s: %.3f' % (key, value))
    if adfuller_test[0] > adfuller_test[4]['5%']:
        print('Series is non-stationary')
    else:
        print('Series is stationary')

    plot_acf(data["vacancies"])
    plt.show()  # nonstationary

    plot_acf(data["vacancies"].diff().dropna())
    plt.show()  # better, but too many sequential lags oout of critical area

    plot_acf(data["vacancies"].diff().diff().dropna())
    plt.show() # 4 lags have strong autocorrelation - q = 4
    plot_pacf(data["vacancies"])
    plt.show()# third is the closest to limit - so p = 1 (first lag ignored)"""


    model = ARIMA(data_train["vacancies"], order=(1, 1, 4))
    arima_result = model.fit()



    #evaluating quality
    fig, ax = plt.subplots(figsize=(10, 6))
    data_train.plot(ax=ax)
    tsaplots.plot_predict(arima_result, ax=ax,
                          start=data_test.index.min(),
                          end=data_test.index.max())
    data_test.plot(ax=ax)
    plt.show()
    future_data_arima = arima_result.forecast(steps=7)

    #predicting data for next week
    fig, ax = plt.subplots(figsize=(10, 6))
    data.tail(60).plot(ax=ax)
    tsaplots.plot_predict(arima_result, ax=ax,
                          start=data.index.max(),
                          end=data.index.max()+timedelta(days=7))
    plt.show()

    #going with regression
    lag_orders = 2

    regr_data=pd.DataFrame(data)

    for i in range(1, lag_orders + 1):
        regr_data[f'vacancies_lag_{i}'] = regr_data['vacancies'].shift(i)

    regr_data = regr_data.dropna()

    X = regr_data[['vacancies_lag_1', 'vacancies_lag_2']]
    y = regr_data['vacancies']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")

    y_pred_series = pd.Series(y_pred, index=y_test.index)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_train.plot(ax=ax, label='Training Data')
    y_pred_series.plot(ax=ax, label='Predicted Data')
    y_test.plot(ax=ax, label='Testing Data')
    plt.legend()
    plt.show()

    last_date = data.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)
    future_data_regr = pd.DataFrame({'date': future_dates, 'vacancies': np.nan})
    future_data_regr.set_index('date', inplace=True)

    y_tmp=pd.DataFrame(y, index=y.index)

    for i in range(7):
        last_observation = y_tmp.iloc[-1]['vacancies']
        lag_1 = y_tmp.iloc[-1]['vacancies']
        lag_2 = y_tmp.iloc[-2]['vacancies']

        prediction = model.predict([[lag_1, lag_2]])
        future_data_regr['vacancies'].iloc[i] = prediction[0]
        y_tmp.loc[last_date + pd.DateOffset(days=i + 1)]=prediction[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    y.tail(60).plot(ax=ax, label="vacancies")
    future_data_regr['vacancies'].plot(ax=ax, label="predicted")
    plt.legend()
    plt.show()

    #correlations
    correlation = np.corrcoef(future_data_arima.values, future_data_regr['vacancies'].values)[0, 1]
    print(f"Correlation of two prediction sets: {correlation}")

    plt.scatter(future_data_arima, future_data_regr['vacancies'])
    plt.xlabel('Regression')
    plt.ylabel('Arima')
    plt.show()