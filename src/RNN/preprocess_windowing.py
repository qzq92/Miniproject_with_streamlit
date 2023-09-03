# Creating end to end pipeline from preprocessing to model evaluation
import pandas as pd
import numpy as np 
# from src.datapipeline import run_datapipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def load_data(train_percent, test_percent):
    # Load the data using the pipeline function
    df1 = pd.read_csv('../time_series/data/train_GzS76OK/train.csv') #'./data/train_GzS76OK/train.csv'
    df2 = pd.read_csv('../time_series/data/train_GzS76OK/fulfilment_center_info.csv') #'./data/train_GzS76OK/fulfilment_center_info.csv'
    df3 = pd.read_csv('../time_series/data/train_GzS76OK/meal_info.csv') # './data/train_GzS76OK/meal_info.csv'

    # 1. Merge all df 
    df = pd.merge(df1, df2, on='center_id',how='left')
    df = pd.merge(df,df3,on='meal_id',how='left')

    # convert weeks to time stamp
    # Calculate the date corresponding to the start of the first week
    start_date = pd.to_datetime('2019-01-28')

    # Calculate the date for each week based on the start date and the week number
    df['date'] = start_date + pd.to_timedelta(df['week'] - 1, unit='W')

    # Set the 'date' column as the index of the DataFrame
    df.set_index('date', inplace=True)
    
    # Resample the DataFrame by week and sum the 'num_orders' column
    df = pd.DataFrame(df['num_orders'].resample('W').sum())

    # # Split the last 29 records as test series
    total_len = len(df)
    train_len = int(train_percent*total_len)
    test_len = int(test_percent*total_len)

    df_test = df[-test_len:] # 29 weeks in advance (to 145 week) (20%)
    df_train = df[:train_len] # Train on 116 weeks past data. (80%)
    return df_train, df_test

def normalization(df_train, df_test, scaler): #can change scaler
    scaler = scaler
    # scaler = StandardScaler()
    scaler.fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    return train , test , scaler #return scalar for inverse transform later

def create_dataset(dataset, lookback, lookforward, shift=1):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
        lookforward: Number of time steps to predict into the future
        shift: Number of time steps to shift the target
    """
    X, y = [], []
    for i in range(len(dataset)-lookback-lookforward+1):
        feature = dataset[i:i+lookback]
        target = dataset[i+lookforward:i+lookforward+shift]
        X.append(feature)
        y.append(target)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)