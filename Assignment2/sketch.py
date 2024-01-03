# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:42:12 2023

@author: Errasti Domínguez, Nuria and Levenfeld Sabau, Gabriela
"""

# Data import and manipulation
# =============================
import pandas as pd
import numpy as np

# Data processing
# ================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import randint as sp_randint

# Models and metrics
# ===================
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree, set_config
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Bayesian search
# ================
import optuna

# Other library needed
# =====================
from joblib import dump
import time

# Own functions
# ==============
from utils import load_data, split_data, eval_metrics_model, plot_predictions
from knn_bayesian_search import param_search_bo, write_trials_to_csv



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
    
    # Plot for the results - Standard Scaler (with better performance)    
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
    
    reg_tree.fit(X_train, y_train) # Now, we train (fit) the method on the train dataset
    y_val_pred_tree = reg_tree.predict(X_val) # We use the model to predict on the validate set 
    metrics_tree = eval_metrics_model(y_val, y_val_pred_tree) # Evaluate the model
    
    # Plot for the results
    plot_predictions(y_val, y_val_pred_tree, 'Tree')
    
    #----------------------------------------------------------
    # Question 5 (Part 1). Hyper-parameter tuning with random-search: Trees and KNN
    
    # CODE Used for both models: KNN & Tree
    # Defining a fixed train/validation grid search
    # -1 means training, 0 means validation
    validation_indices = np.zeros(X_train.shape[0]) 
    validation_indices[:round(2/3*X_train.shape[0])] = -1
    tr_val_partition = PredefinedSplit(validation_indices)
    
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
                                n_iter=20,
                                scoring='neg_mean_absolute_error',
                                cv=tr_val_partition,
                                n_jobs=4, verbose=1)
    
    reg_knn_grid.fit(X_train, y_train) # Training the model with the grid search
    y_val_pred_hpo_knn = reg_knn_grid.predict(X_val) # Making predictions on the validate set
    metrics_hpo_knn = eval_metrics_model(y_val, y_val_pred_hpo_knn) # Evaluate the model
    
    # Plot for the results
    plot_predictions(y_val, y_val_pred_hpo_knn, 'HPO KNN')
    
    # The best hyper parameter values (and their scores) can be accessed
    print(f'Best hyper-parameters: {reg_knn_grid.best_params_} and inner evaluation: {reg_knn_grid.best_score_}')
    
    
    # Tree model
    # Defining the method with pipeline
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
                                       n_iter=20,
                                       scoring='neg_mean_absolute_error',
                                       cv=tr_val_partition,
                                       n_jobs=3, verbose=1)

    reg_tree_grid.fit(X_train, y_train) # Training the model with the grid search
    y_val_pred_hpo_tree = reg_tree_grid.predict(X_val) # Making predictions on the validate set 
    metrics_hpo_tree = eval_metrics_model(y_val, y_val_pred_hpo_tree) # Evaluate the model
    
    # Plot for the results
    plot_predictions(y_val, y_val_pred_hpo_tree, 'HPO Tree')
    
    # The best hyper parameter values (and their scores) can be accessed
    print(f'Best hyper-parameters: {reg_tree_grid.best_params_} and inner evaluation: {reg_tree_grid.best_score_}')
    
    #----------------------------------------------------------
    # Step 5.5. Hyper-parameter tuning: KNN with bayesian-search
    
    # Create Optuna study
    study_knn = optuna.create_study(study_name='AP_Task2',
                                    direction='minimize',
                                    storage = 'sqlite:///KNNPredictBO.db',
                                    load_if_exists=True)
    # Bayesian search
    n_trials=60
    best_params_bo = param_search_bo((X_train, y_train), (X_val, y_val), study_knn, n_trials)
    best_mae_bo = study_knn.best_value
    print(f'Best hyper-parameters: {study_knn.best_params}')
     
    # Extra info with all trials compute during the bayesian search
    trials = study_knn.get_trials()
    trial_bo_csv_path = 'results/trials_info.csv'
    write_trials_to_csv(trials, trial_bo_csv_path)
    print(f"All trials information with Bayesian-search has been saved in '{trial_bo_csv_path}'")

    #----------------------------------------------------------
    # Question 5 (Part 2). Report a summary with results
    name_bo_params = {key: value for key, value in best_params_bo.items() if key != 'scalers'}

    summary_models = pd.DataFrame({
        'Model': ['KNN', 'KNN', 'KNN', 'Tree', 'KNN', 'Tree', 'KNN'],
        'Search': ['None', 'None', 'None', 'None', 'Random-search', 'Random-search', 'Bayesian-search'],
        'Standarization': ['standard', 'robust', 'MinMax', 'None', 'standard', 'None', best_params_bo['scalers']],
        'Best Hyperparameters': ['Default', 'Default', 'Default', 'Default', reg_knn_grid.best_params_, reg_tree_grid.best_params_, name_bo_params],
        'Validation MAE': [metrics_knn_std['MAE'], metrics_knn_robust['MAE'], metrics_knn_minmax['MAE'], metrics_tree['MAE'], metrics_hpo_knn['MAE'], metrics_hpo_tree['MAE'], best_mae_bo],
        'Validation R2': [metrics_knn_std['R^2'], metrics_knn_robust['R^2'], metrics_knn_minmax['R^2'], metrics_tree['R^2'], metrics_hpo_knn['R^2'], metrics_hpo_tree['R^2'], 'NaN'],
        'Validation RMSE': [metrics_knn_std['RMSE'], metrics_knn_robust['RMSE'], metrics_knn_minmax['RMSE'], metrics_tree['RMSE'], metrics_hpo_knn['RMSE'], metrics_hpo_tree['RMSE'], 'NaN']
    })
    summary_df = pd.DataFrame(summary_models)
    summary_csv_path = 'results/models_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary has been saved in '{summary_csv_path}'")
    
    #----------------------------------------------------------
    # Question 6. Only with the Best method
    
    # Look for the best model
    df = pd.read_csv('results/models_summary.csv')
    best_model = df.loc[df['Validation MAE'].idxmin()]
    print(f'The best model is {best_model["Model"]} with {best_model["Search"]}.')
    # The best model is KNN with Bayesian-search
    
    # a -> Estimation of the error at the competition
    print(f'The estimation of the error that the model might get at the competition is: {best_mae_bo}')
    
    # b -> Final model
    # best_params_bo has already been defined and it carries out the parameters of the HPO with bayesian search
    scalers = best_params_bo['scalers']
    if scalers == "minmax":
        best_scaler = MinMaxScaler()
    elif scalers == "standard":
        best_scaler = StandardScaler()
    else:
        best_scaler = RobustScaler()
        
    final_model = Pipeline([
        ('imputation', imputer_knn),
        ('standarization', best_scaler),
        ('knn', KNeighborsRegressor(n_neighbors=best_params_bo['n_neighbors'],
                                    weights=best_params_bo['weights'],
                                    algorithm=best_params_bo['algorithm'],
                                    leaf_size=best_params_bo['leaf_size']))
        ])
    
    start_time = time.time()
    # In order to get the final model we need to train using the entire dataset (X,y)
    final_model.fit(X, y)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Now, we can use the final_model to make predictions on new data
    wind_comp = load_data('data/wind_competition.csv.gzip') # Load new data
    pred_new = final_model.predict(wind_comp)
    
    wind_comp['Predictions'] = pred_new
    # Save predictions for the competition dataset
    wind_comp.to_csv('results/predictions_wind_competition.csv', index=False)
    
    # Save the final_model
    dump(final_model, 'results/final_model.joblib')

    #----------------------------------------------------------
    # Question 7. Feature selection for KNN
    
    # Set the global configuration to keep DataFrame structure in transformers' output
    set_config(display='diagram', transform_output='pandas')
    
    # Create the pipeline for defining the method KNN
    # best_scaler; param getting for the bayesian-search
    selector = SelectKBest()
    knn_fs = KNeighborsRegressor(n_neighbors=best_params_bo['n_neighbors'])
    reg_knn_fs = Pipeline([
        ('imputation', imputer_knn),
        ('standarization', best_scaler),
        ('select', selector),
        ('knn', knn_fs)
        ])
    
    # Defining hyper-parameter space
    param_grid_fs = {'select__k': list(range(4, 80, 1)),
                     'select__score_func': [f_regression, mutual_info_regression]}
    
    reg_fs_grid = GridSearchCV(reg_knn_fs,
                               param_grid=param_grid_fs,
                               n_iter=20,
                               scoring='neg_mean_absolute_error',
                               cv=tr_val_partition,
                               n_jobs=4, verbose=1)
    
    reg_fs_grid.fit(X_train, y_train) # Training the model with the grid search
    y_val_pred_fs = reg_fs_grid.predict(X_val) # Making predictions on the validation set
    metrics_fs = eval_metrics_model(y_val, y_val_pred_fs) # Evaluate the model
    
    # Get the best score
    best_score_fs = reg_fs_grid.best_score_
    best_params_fs = reg_fs_grid.best_params_
    print(f'Best score (negative mean absolute error): {best_score_fs}')
    print(f'Best params: {best_params_fs}')

    # Get the selected feature scores and names
    selected_feature_mask = reg_fs_grid.best_estimator_.named_steps['select'].get_support()
    feature_scores = reg_fs_grid.best_estimator_.named_steps['select'].scores_[selected_feature_mask]
    selected_feature_names = X_train.columns[selected_feature_mask]
    print("Selected features and scores:")
    for elem in zip(selected_feature_names, feature_scores):
        print(elem)
        
    # Sort the selected features based on scores
    sorted_features = sorted(zip(selected_feature_names, feature_scores), key=lambda x: x[1], reverse=True)
    print("Selected features and scores sorted")
    for elem in sorted_features:
        print(elem)