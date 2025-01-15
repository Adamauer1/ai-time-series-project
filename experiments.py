import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from random_forest.random_forest import RandomForest
from exponential_smoothing.triple_exponential_smoothing import TripleExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

climate_data_path = "./datasets/DailyDelhiClimateTrain.csv"
microsoft_stock_data_path = "./datasets/Microsoft_Stock.csv"

def run_exponential_smoothing(dataset_path, target_name, date_name, seasonality_value, alpha, beta, gamma, prebuild_flag = False):
    df = pd.read_csv(dataset_path)
    data = df[target_name].dropna().to_numpy()
    #dates = df[date_name]
    y_test = data[int(len(data) * .8):]
    start_time = time.time()
    if prebuild_flag:
        model = ExponentialSmoothing(data[:int(len(data) * .8)], trend="add", damped_trend=True, seasonal="add", seasonal_periods=seasonality_value)
        # smoothed_values = model.fit(data[:int(len(data) * .8)-1], seasonality_value)
        smoothed_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
        forecasts = smoothed_model.forecast(len(y_test))
        end_time = time.time()
        # test_dates = dates.iloc[-len(y_test):]
        rmse = np.sqrt(mean_squared_error(y_test, forecasts))
        return rmse, (end_time - start_time)
    model = TripleExponentialSmoothing(alpha, beta, gamma)
    smoothed_values = model.fit(data[:int(len(data) * .8)], seasonality_value)
    forecasts = model.predict(len(y_test))
    end_time = time.time()
    #test_dates = dates.iloc[-len(y_test):]
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))
    plt.figure(figsize=(12, 6))
    plt.plot(forecasts, label="Triple", linestyle="--", marker=None)
    plt.plot(y_test, label="Real", marker=None)
    # plt.plot(test_dates, y_test, label='Actual', color='blue', marker='o')
    # plt.plot(test_dates, forecasts, label='Predicted', color='red', linestyle='--', marker='x')
    plt.legend(fontsize=14)
    plt.xlabel("Time")
    plt.ylabel(target_name)
    plt.show()
    return rmse, (end_time - start_time)

def run_random_forest(dataset_path, target_name, feature_names, date_name, n_lags, n_trees, max_depth, min_split_amount, diff_flag = False, prebuilt_flag = False):

    data = pd.read_csv(dataset_path)

    df = pd.DataFrame(data)

    if diff_flag:
        for feature in feature_names:
            df[feature] = df[feature].diff(1)

    for lag in range(1, n_lags + 1):
        for feature in feature_names:
            df[f'lag_{lag}_{feature}'] = df[feature].shift(lag)
    df.dropna(inplace=True)
    x = df.drop(columns=feature_names+[date_name]).iloc[:,:].values
    y = np.array([[item] for item in df[target_name]])
    #dates = data[date_name]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41, shuffle=False)
    start_time = time.time()
    if prebuilt_flag:
        random_forest = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, min_samples_split=min_split_amount, bootstrap=True)
        random_forest.fit(x_train, y_train.ravel())
        y_predictions = random_forest.predict(x_test)
        end_time = time.time()
        rmse = np.sqrt(mean_squared_error(y_test, y_predictions))
        return rmse, (end_time - start_time)
    random_forest = RandomForest(n_trees, max_depth, min_split_amount)
    random_forest.fit(x_train, y_train)
    y_predictions = random_forest.predict(x_test)
    end_time = time.time()
    rmse = np.sqrt(mean_squared_error(y_test, y_predictions))
    #test_dates = dates.iloc[-len(y_test):]
    plt.figure(figsize=(12, 6))
    plt.plot(y_predictions, label="Random Forest", linestyle="--", marker=None)
    plt.plot(y_test, label="Real", marker=None)
    # plt.plot(test_dates, y_test, label='Actual', color='blue', marker='o')
    # plt.plot(test_dates, forecasts, label='Predicted', color='red', linestyle='--', marker='x')
    plt.legend(fontsize=14)
    plt.xlabel("Time")
    plt.ylabel(target_name)
    plt.show()
    return rmse, (end_time - start_time)


def run_experiments(prebuilt_flag = False):
    climate_tes_rmses = 0
    climate_rf_rmses = 0
    climate_tes_times = 0
    climate_rf_times = 0
    stock_tes_rmses = 0
    stock_rf_rmses = 0
    stock_tes_times = 0
    stock_rf_times = 0
    for _ in range(5):
        climate_tes_rmse, climate_tes_time = run_exponential_smoothing(climate_data_path, "meantemp", "date", 363, 0.2, 0.5,0.1, prebuild_flag=prebuilt_flag)
        climate_rf_rmse, climate_rf_time = run_random_forest(climate_data_path, "meantemp", ['meantemp', 'humidity', 'wind_speed', 'meanpressure'], "date", 7, 50, 3,  27, diff_flag=False, prebuilt_flag=prebuilt_flag)
        climate_tes_rmses += climate_tes_rmse
        climate_rf_rmses += climate_rf_rmse
        climate_tes_times += climate_tes_time
        climate_rf_times += climate_rf_time

        stock_tes_rmse, stock_tes_time =run_exponential_smoothing(microsoft_stock_data_path, "Close", "Date", 251, 0.1, 0.1, 0.3, prebuild_flag=prebuilt_flag)
        stock_rf_rmse, stock_rf_time = run_random_forest(microsoft_stock_data_path, "Close", ["Open", "High", "Low", "Volume", "Close"], "Date", 5, 50, 4, 18, diff_flag=True, prebuilt_flag=prebuilt_flag)
        stock_tes_rmses += stock_tes_rmse
        stock_rf_rmses += stock_rf_rmse
        stock_tes_times += stock_tes_time
        stock_rf_times += stock_rf_time

    print("Climate Test")
    print(climate_tes_rmses / 5)
    print(climate_rf_rmses / 5)
    print(climate_tes_times / 5)
    print(climate_rf_times / 5)

    print("Stock Test")
    print(stock_tes_rmses / 5)
    print(stock_rf_rmses / 5)
    print(stock_tes_times / 5)
    print(stock_rf_times / 5)


# run_experiments()
run_experiments(prebuilt_flag=True)
