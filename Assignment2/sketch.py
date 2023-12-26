# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:42:12 2023

@author: Errasti Domínguez, Nuria and Levenfeld Sabau, Gabriela
"""

# Data import and manipulation
# =============================
import pandas as pd
import numpy as np

# Visualization
# ==============
import matplotlib.pyplot as plt

# Data processing
# ================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Models and metrics
# ===================
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import metrics



def load_data(file_path):
    return pd.read_csv(file_path, compression="gzip")

def split_data(data, years_train_train):
    # Split the data into train_train (80%) and train_val (20%) dataset.
        # For train_train we take 2005, 2006, 2007 and 2008.
    train_train = data[data['year'].isin(years_train_train)]
    X_train = train_train.drop(columns='energy')
    y_train = train_train['energy'].values

        # For train_validation we take 2009.
    train_val = data[~data['year'].isin(years_train_train)]
    X_val = train_val.drop(columns='energy')
    y_val = train_val['energy'].values

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    wind_ava = load_data('data/wind_available.csv.gzip')
    
    #----------------------------------------------------------
    # Question 3: SPLIT into TRAIN and TEST
    
    # In order to decide how to split the dataset, we take a look on how data is distribute
    year_counts = wind_ava.groupby('year').size()
    
    # Selected years for train_train + Our final split percentage
    years_train_train = [2005, 2006, 2007, 2008]
    total_data_count = year_counts.sum()
    percentage_train_train = year_counts[years_train_train].sum() / total_data_count * 100
    percentage_train_val = 100 - percentage_train_train
    
    X_train, y_train, X_val, y_val = split_data(wind_ava, years_train_train)
    
    #----------------------------------------------------------
    # Step 3.5: Ouliers -> los manejamos (?)
    
    
    #----------------------------------------------------------
    # Question 4: Default hyper-parameters: Trees and KNN
    
    # Step: Handling missing values (KNN and Trees can not deal with NA)
    # No se puede hacer antes porque sino fuga de valores
    
    # For KNN, prepocessing pipeline
    imputer = SimpleImputer(strategy='median')
    imputer = KNNImputer()
    scaler = StandardScaler()
    scaler = RobustScaler() #311
    scaler = MinMaxScaler() #347.15
    knn = KNeighborsRegressor()
    
    reg_knn = Pipeline([
        ('imputation', imputer),
        ('standarization', scaler),
        ('knn', knn)
        ])
    
    reg_knn.fit(X_train, y_train)
    y_val_pred = reg_knn.predict(X_val)
    
    mae = metrics.mean_absolute_error(y_val, y_val_pred)
    print(f'Mean Absolute Error: {mae}')
    rmse = metrics.mean_squared_error(y_val, y_val_pred)
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    r2 = metrics.r2_score(y_val, y_val_pred)
    print(f'R-squared: {r2}')
    
    
    # Plot for the results
    plt.scatter(y_val, y_val_pred)
    plt.plot(y_val, y_val, color='red')
    plt.xlabel('Actual Energy output')
    plt.ylabel('Predicted Energy output')
    plt.title('Actual vs Predicted values')
    plt.show()
    
    
    # Tree model
    SEED = 100507449
    np.random.seed(SEED)
    
    reg_tree = tree.DecisionTreeRegressor(random_state=SEED)
    reg_tree.fit(X_train, y_train)
    y_val_pred = reg_tree.predict(X_val)
    print(y_val_pred)
    print(y_val)
    print(metrics.accuracy_score(y_val, y_val_pred))
    
    