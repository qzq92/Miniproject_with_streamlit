import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

def plot (scaler, df_train, df_test, trained_model, X_train, y_train, X_test, y_test):
    # Make predictions on the test set
    with torch.no_grad():
        y_pred_test = trained_model(X_test)
        y_pred_test = scaler.inverse_transform(y_pred_test[:, -1, :].numpy())

    # Make predictions on the training set
    with torch.no_grad():
        y_pred_train = trained_model(X_train)
        y_pred_train = scaler.inverse_transform(y_pred_train[:, -1, :].numpy())

    # Convert the targets to numpy arrays
    y_test = scaler.inverse_transform((y_test.reshape(-1, 1)).numpy())
    y_train = scaler.inverse_transform((y_train.reshape(-1, 1)).numpy())

    # Create a DataFrame to store the predicted values
    pred_df_test = pd.DataFrame({
        'Actual Value_test': y_pred_test.flatten(),
        'Forecast_test': y_test.flatten()
    })

    pred_df_train = pd.DataFrame({
        'Actual Value_train': y_pred_train.flatten(),
        'Forecast_train': y_train.flatten()
    })

    # Set index for plotting
    pred_df_train.index = df_train[len(df_train) - len(pred_df_train):].index
    pred_df_test.index = df_test[-len(pred_df_test):].index

    # Plotting
    print(pred_df_train) # predict 10 weeks in advance
    pred_df_train.plot(figsize=(15, 10))
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()

    print(pred_df_test) # predict 10 weeks in advance
    pred_df_test.plot(figsize=(15, 10))
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()

    # Final RMSE for test and train:
    rmse_test = np.sqrt(mean_squared_error(pred_df_test['Actual Value_test'], pred_df_test['Forecast_test']))
    rmse_train = np.sqrt(mean_squared_error(pred_df_train['Actual Value_train'], pred_df_train['Forecast_train']))

    print(f"Unnormalized test RMSE: ", rmse_test)
    print(f"Unnormalized train RMSE: ", rmse_train)
