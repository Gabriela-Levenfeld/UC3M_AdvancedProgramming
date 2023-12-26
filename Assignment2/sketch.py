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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression




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
    # Our data is non i.i.d. since it's temporally ordered
    
    # Selected years for train_train + Our final split percentage
    years_train_train = [2005, 2006, 2007, 2008]
    total_data_count = year_counts.sum()
    percentage_train_train = year_counts[years_train_train].sum() / total_data_count * 100
    percentage_train_val = 100 - percentage_train_train
    
    X_train, y_train, X_val, y_val = split_data(wind_ava, years_train_train)
    
    # The data we will use to train the final model:
    X = wind_ava.drop(columns='energy')
    y = wind_ava['energy'].values
    
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
    
    # We define the type of training method (nothing happens yet)
    reg_tree = tree.DecisionTreeRegressor(random_state=SEED)
    # Now, we train (fit) the method on the train dataset
    reg_tree.fit(X_train, y_train)
    #We use the model to predict on the validate set 
    y_val_pred = reg_tree.predict(X_val)
    
    mae = metrics.mean_absolute_error(y_val, y_val_pred)
    print(f'Mean Absolute Error: {mae}')
    rmse = metrics.mean_squared_error(y_val, y_val_pred)
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    r2 = metrics.r2_score(y_val, y_val_pred)
    print(f'R-squared: {r2}')
    
    #Plot for the results
    plt.scatter(y_val, y_val_pred)
    plt.plot(y_val, y_val, color='red')
    plt.xlabel('Actual Energy output')
    plt.ylabel('Predicted Energy output')
    plt.title('Actual vs Predicted values')
    plt.show()
    
    #----------------------------------------------------------
    # Question 5: Hyper-parameter tuning: Trees and KNN
    
    #KNN
    # Preprocessing the pipeline
    imputer = SimpleImputer(strategy='median')
    imputer = KNNImputer()
    scaler = StandardScaler()
    scaler = RobustScaler() #311
    scaler = MinMaxScaler() #347.15
    knn = KNeighborsRegressor()
    selector = SelectKBest(f_regression)
    
    #Defining the pipeline
    reg_knn = Pipeline([
        ('imputation', imputer),
        ('standarization', scaler),
        ('knn', knn),
        ('select', selector),
        ])
    # Defining the hyperparameter space
    param_grid = {'select__k': [2,3,4],
                  'knn__n_neighbors': [1,3,5]}
    
    # Defining a 5 fold crossvalidation grid search
    inner_cv= KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    reg_knn_grid = GridSearchCV(reg_knn,
                            param_grid,
                            scoring = 'neg_mean_squared_error', 
                            cv=inner_cv, n_jobs=1, verbose=1)
    # Now, we train (fit) the method on the train dataset
    reg_knn_grid.fit(X_train , y_train)
    y_val_pred = reg_knn_grid.predict(X_val)
    
    # The best hyper parameter values (and their scores) can be accessed
    reg_knn_grid.best_params_
    reg_knn_grid.best_score_
    
    
    # Plot for the results
    plt.scatter(y_val, y_val_pred)
    plt.plot(y_val, y_val, color='red')
    plt.xlabel('Actual Energy output')
    plt.ylabel('Predicted Energy output')
    plt.title('Actual vs Predicted values')
    plt.show()




    # Trees
    # Defining the method
    reg_tree = tree.DecisionTreeRegressor(random_state=SEED)
    # Defining the Search space
    param_grid = {'max_depth': range(2,16,2),
                  'min_samples_split': range(2,34,2)}
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED)
    reg_grid = GridSearchCV(reg_tree,
                            param_grid,
                            scoring='neg_mean_squared_error',
                            cv=cv,
                            n_jobs= 1 , 
                            verbose=1)
    # Fit does hyper parameter tuning, followed by training the model 
    # with the best hyper parameters found
    reg_grid.fit(X_train, y_train)
    # Making predictions on the validating partition
    y_val_pred = reg_grid.predict(X_val)
    # And finally computing the test accuracy (estimation of future
    # performance or outer evaluation)
    print(metrics.accuracy_score(y_val_pred, y_val))
            
            
    
    
    
    
    
    
    
    
    
    