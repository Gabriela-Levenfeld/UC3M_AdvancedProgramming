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
from scipy.stats import randint as sp_randint

# Models and metrics
# ===================
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit


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

def eval_metrics_model(y_val, y_val_pred):
    """ Function to compute and print performance metrics: MAE, RMSE and R-squared. """
    
    mae = metrics.mean_absolute_error(y_val, y_val_pred)
    rmse = metrics.mean_squared_error(y_val, y_val_pred)
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



if __name__ == "__main__":
    wind_ava = load_data('data/wind_available.csv.gzip')
    
    #----------------------------------------------------------
    # Question 3. SPLIT into TRAIN and TEST
    
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
    # Question 4. Default hyper-parameters: Trees and KNN
    
    # Step: Handling missing values (KNN and Trees can not deal with NA)
    # No se puede hacer antes porque sino fuga de valores
    
    # KNN model
    imputer_knn = KNNImputer() # Imputation transformer for completing missing values
    knn = KNeighborsRegressor()
    
    # Pipeline for KNN with Standard Scaler
    reg_knn_std = Pipeline([
        ('imputation', imputer_knn),
        ('standarization', StandardScaler()),
        ('knn', knn)
        ])
    
    # Pipeline for KNN with Robust Scaler
    reg_knn_robust = Pipeline([
        ('imputation', imputer_knn),
        ('standarization', RobustScaler()),
        ('knn', knn)
        ])
        
    # Pipeline for KNN with MinMax Scaler
    reg_knn_minmax = Pipeline([
        ('imputation', imputer_knn),
        ('standarization', MinMaxScaler()),
        ('knn', knn)
        ])
    
    # Fit and evaluate KNN with Standard Scaler
    reg_knn_std.fit(X_train, y_train)
    y_val_pred_std = reg_knn_std.predict(X_val)  
    metrics_knn_std = eval_metrics_model(y_val, y_val_pred_std)
    
    # Fit and evaluate KNN with Robust Scaler
    reg_knn_robust.fit(X_train, y_train)
    y_val_pred_robust = reg_knn_robust.predict(X_val)    
    metrics_knn_robust = eval_metrics_model(y_val, y_val_pred_robust)  
    
    # Fit and evaluate KNN with MinMax Scaler
    reg_knn_minmax.fit(X_train, y_train)
    y_val_pred_minmax = reg_knn_minmax.predict(X_val)  
    metrics_knn_minmax = eval_metrics_model(y_val, y_val_pred_minmax)
    
    # Plot for the results - Standard Scaler (which performance better)
    plt.scatter(y_val, y_val_pred_std)
    plt.plot(y_val, y_val, color='red')
    plt.xlabel('Actual Energy output')
    plt.ylabel('Predicted Energy output')
    plt.title('Actual vs Predicted values (Standard Scaler)')
    plt.show()
    
    plot_predictions(y_val, y_val_pred_std, 'KNN (Standard Scaler)')
    
    # Tree model
    SEED = 100507449
    np.random.seed(SEED)
    
    # We define the type of training method (nothing happens yet)
    tree_model = tree.DecisionTreeRegressor(random_state=SEED)
    imputer_tree = SimpleImputer(strategy='mean') # Imputation transformer for completing missing values
    
    # Pipeline for Tree 
    reg_tree = Pipeline([
        ('imputation', imputer_tree),
        ('tree', tree_model)
        ])
    
    # Now, we train (fit) the method on the train dataset
    reg_tree.fit(X_train, y_train)
    # We use the model to predict on the validate set 
    y_val_pred_tree = reg_tree.predict(X_val)
    # Evaluate the model
    metrics_tree = eval_metrics_model(y_val, y_val_pred_tree)
    
    # Plot for the results
    plot_predictions(y_val, y_val_pred_tree, 'Tree')
    
    #----------------------------------------------------------
    # Question 5. Hyper-parameter tuning: Trees and KNN
    
    # CODE Used for both models: KNN & Tree
    # Defining a fixed train/validation grid search
    # -1 means training, 0 means validation
    validation_indices = np.zeros(X_train.shape[0]) 
    validation_indices[:round(2/3*X_train.shape[0])] = -1
    tr_val_partition = PredefinedSplit (validation_indices)
    
    # KNN model
    # Defining the method (KNN) with pipeline
    reg_knn_hpo = Pipeline([
        ('imputation', imputer_knn),
        ('standarization', StandardScaler()),
        ('knn', knn)
        ])
    
    # Defining the Search space
    param_grid_knn = {'knn__n_neighbors': sp_randint(2,16,2),
                      'knn__weights': ['uniform', 'distance'],
                      'knn__leaf_size': sp_randint(10,50,2)}
    
    reg_knn_grid = RandomizedSearchCV(reg_knn_hpo,
                                param_distributions=param_grid_knn,
                                n_iter=10,
                                scoring='neg_mean_absolute_error',
                                cv=tr_val_partition,
                                n_jobs=1, verbose=1)
    
    reg_knn_grid.fit(X_train, y_train) # Now, we train (fit) the method on the train dataset
    y_val_pred_hpo_knn = reg_knn_grid.predict(X_val) # We use the model to predict on the validate set 
    metrics_hpo_knn = eval_metrics_model(y_val, y_val_pred_hpo_knn) # Evaluate the model
    
    # Plot for the results
    plot_predictions(y_val, y_val_pred_hpo_knn, 'HPO KNN')
    
    # The best hyper parameter values (and their scores) can be accessed
    print(f'Best hyper-parameters: {reg_knn_grid.best_params_} and inner evaluation: {reg_knn_grid.best_score_}')
    
    
    # Tree model
    # Defining the method
    reg_tree_hpo = Pipeline([
        ('imputation', imputer_tree),
        ('tree', tree_model)
        ])
    
    # Defining the Search space
    param_grid_tree = {'tree__max_depth': sp_randint(2,16,2),
                       'tree__min_samples_split': sp_randint(2,34),
                       'tree__min_samples_leaf': sp_randint(1,30,5)}
    
    reg_tree_grid = RandomizedSearchCV(reg_tree_hpo,
                                       param_distributions=param_grid_tree,
                                       n_iter=10,
                                       scoring='neg_mean_absolute_error',
                                       cv=tr_val_partition,
                                       n_jobs=1, verbose=1)

    reg_tree_grid.fit(X_train, y_train) # Training the model with the grid-search
    y_val_pred_hpo_tree = reg_tree_grid.predict(X_val) # Making predictions on the validate set 
    metrics_hpo_tree = eval_metrics_model(y_val, y_val_pred_hpo_tree) # Evaluate the model
    
    # Plot for the results
    plot_predictions(y_val, y_val_pred_hpo_tree, 'HPO Tree')
    
    # The best hyper parameter values (and their scores) can be accessed
    print(f'Best hyper-parameters: {reg_tree_grid.best_params_} and inner evaluation: {reg_tree_grid.best_score_}')
    
    
    # Report a summary with results
    summary_models = pd.DataFrame({
        'Model': ['KNN', 'KNN', 'KNN', 'Tree', 'KNN (HPO)', 'Tree (HPO)'],
        'Standarization': ['Standard Scaler', 'Robust Scaler', 'MinMax Scaler', 'None', 'Standard Scaler', 'None'],
        'Best Hyperparameters': ['Default', 'Default', 'Default', 'Default', reg_knn_grid.best_params_, reg_tree_grid.best_params_],
        'Validation MAE': [metrics_knn_std['MAE'], metrics_knn_robust['MAE'], metrics_knn_minmax['MAE'], metrics_tree['MAE'], metrics_hpo_knn['MAE'], metrics_hpo_tree['MAE']],
        'Validation R2': [metrics_knn_std['R^2'], metrics_knn_robust['R^2'], metrics_knn_minmax['R^2'], metrics_tree['R^2'], metrics_hpo_knn['R^2'], metrics_hpo_tree['R^2']],
        'Validation RMSE': [metrics_knn_std['RMSE'], metrics_knn_robust['RMSE'], metrics_knn_minmax['RMSE'], metrics_tree['RMSE'], metrics_hpo_knn['RMSE'], metrics_hpo_tree['RMSE']]
    })
    summary_df = pd.DataFrame(summary_models)
    summary_df.to_csv('results/models_summary.csv', index=False)
    print("Summary has been saved to 'models_summary.csv'")
    