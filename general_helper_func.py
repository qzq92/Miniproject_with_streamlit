"""
Script containing useful helper functions

"""
import matplotlib.pyplot as plt
import pandas as pd

def extract_info_from_model_name(stats_model_filename):
    """Function that extracts ARIMA/SARIMA parameters captured in the input file name representing arima/sarima models as identified with the following format <arima/sarima>_model-<arima order parameters>-<sarima seasonal order parameters>.pkl
    
    E.g arima_model-1_1_1.pkl ;sarima_model_model-1_1_1-3_1_3_52.pkl
    
    Args:
        stats_model_filename (string): Model file name in string with extension
    Returns:
        order_info (tuple): Extracted ARIMA component parameters
        seasonal_order_info (tuple): Extracted SARIMAX component seasonal parameters
    Raises:
        ValueError when empty input is passed.
    """
    if not stats_model_filename:
        raise ValueError('Empty string detected, please check')

    temp_name = stats_model_filename.replace(".pkl","")
    if temp_name.startswith('a'):
        temp_name_list = temp_name.split("-")
        order_info = tuple([int(i) for i in temp_name_list[1].split("_")])
        seasonal_order_info = None
    else:
        temp_name_list = temp_name.split("-")
        order_info = tuple([int(i) for i in temp_name_list[1].split("_")])
        seasonal_order_info = tuple([int(i) for i in temp_name_list[2].split("_")])
    
    return order_info, seasonal_order_info


def plot_forecast(test, forecast):
    """Function that plots the content of test series and forecasted values into single matplotlib object 

    Args:
        test (Pandas Series): Test dataset
        forecast (numpy array): Array of forecasted values 
    Returns:
        None.
    Raises:
        None.
    """
    # Plot the actual values and forecasted values for each model
    plt.figure(figsize=(20, 3))
    plt.plot(test.index, test.values, label='Test Data')
    plt.plot(test.index, forecast, label='Manual ARIMA Forecast')
    plt.title('Actual vs. Forecasted Values')
    plt.xlabel('Week')
    plt.ylabel('Number of Sales Orders')
    plt.legend()
    plt.show()
    return None

def resample_num_orders_weekly(df):
    # Load the data using the pipeline function
    # Resample the DataFrame by week and sum the 'num_orders' column

    # Ensures setting of dataframe
    if 'date' in df.columns:
        df = df.set_index('date')
    ts = df['num_orders'].resample('W-MON').sum()

    return ts

def slice_time_series(ts, n_periods=10):
    """Function that slices an input timeseries data into simple train and test portions indexed by time based on input number of periods, where last n_periods of the time series data is used as test data, and the remaining as training data.

    Args:
        ts (Pandas Series): 
            Time series data
        n_periods (int):
            Number of periods to use for test set
    Returns:
        train (Pandas Series):
            Training dataset
        test (Pandas Series):
            Testing dataset
    Raises:
        None.
    """
    # Split the last 10 records as test series
    train = ts[:-n_periods]
    test = ts[-n_periods:]

    return train, test


def simple_feature_engineer(df):

    # Combine features
    df['meal_id-cuisine-category'] = df['meal_id'].astype(str) + '-' + df['cuisine']+'-'+ df['category']

    df['center_type_short'] = df['center_type'].map(lambda x: x[-1])
    df['center_id_type'] = df['center_id'].astype(str) + '-' + df['center_type_short']

    df['total_base_price'] = df['base_price'] * df['num_orders']
    df['total_checkout_price'] = df['checkout_price'] * df['num_orders']

    df = df.drop('center_type_short', axis=1)

    return df

def create_test_data_date_info(test_df):
    # convert weeks to time stamp
    # Calculate the date corresponding to the start of the first week since test data week value is a continuation of train dataset
    start_date = pd.to_datetime('2021-11-8')

    # Calculate the date for each week based on the start date and the week number
    test_df['date'] = start_date + pd.to_timedelta(test_df['week'] - 146, unit='W')

    # Set the 'date' column as the index of the DataFrame
    test_df.set_index('date', inplace=True)

    return test_df