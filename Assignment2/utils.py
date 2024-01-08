# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:42:12 2023

@author: Errasti Domínguez, Nuria and Levenfeld Sabau, Gabriela
"""

# Data import and manipulation
# =============================
import pandas as pd

# Visualization
# ==============
import matplotlib.pyplot as plt

# Models and metrics
# ===================
from sklearn import metrics


def load_data(file_path):
    return pd.read_csv(file_path, compression="gzip")

def split_data(data, years_train_train):
    """ Split the train data into train_train (80%) and train_val (20%) dataset. """
    
    # For train_train we take 2005, 2006, 2007 and 2008.
    train_train = data[data['year'].isin(years_train_train)]
    X_train = train_train.drop(columns='energy')
    y_train = train_train['energy'].values

    # For train_validation we take 2009.
    train_val = data[~data['year'].isin(years_train_train)]
    X_val = train_val.drop(columns='energy')
    y_val = train_val['energy'].values
    
    X_train = X_train.drop(['year'], axis=1)
    X_val = X_val.drop(['year'], axis=1)

    return X_train, y_train, X_val, y_val

def eval_metrics_model(y_val, y_val_pred):
    """ Function to compute and print performance metrics: MAE, RMSE and R-squared. """
    
    mae = metrics.mean_absolute_error(y_val, y_val_pred)
    rmse = metrics.mean_squared_error(y_val, y_val_pred, squared=False)
    r2 = metrics.r2_score(y_val, y_val_pred)
    
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared: {r2}')
    
    return {'MAE': mae, 'RMSE': rmse, 'R^2': r2}

def plot_predictions(y_val, y_val_pred, model_name):
    """ Plot real and predicted values for a given model. """
    
    x_lab = [i for i in range(len(y_val))]
    plt.figure(figsize=(16, 4))
    plt.plot(x_lab, y_val, label='Real values', marker='o')
    plt.plot(x_lab, y_val_pred, label='Predicted values', marker='x')
    plt.title(f'Predictions for {model_name}')
    plt.legend()
    plt.show()