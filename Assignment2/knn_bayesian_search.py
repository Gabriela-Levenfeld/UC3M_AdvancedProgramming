# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:42:12 2023

@author: Errasti Domínguez, Nuria and Levenfeld Sabau, Gabriela
"""

# Data processing and KNN with Optuna
# ====================================
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from optuna.trial import TrialState
import csv


def create_objective(train_dataset, validation_dataset):
    def objective(trial):
        (X_train, y_train) = train_dataset
        (X_val, y_val) = validation_dataset
        
        # Parameter for the pre-processing
        scalers = trial.suggest_categorical("scalers", ['minmax', 'standard', 'robust'])
        if scalers == "minmax":
            scaler = MinMaxScaler()
        elif scalers == "standard":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        
        # Parameters of KNN method    
        n_neighbors = trial.suggest_int('n_neighbors', 1, 25)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = trial.suggest_int('leaf_size', 10, 50)
        
        # Generate the model
        knn_bo = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size)

        # Create KNN pipeline
        reg_knn_bo = Pipeline([
            ('imputation', KNNImputer()),
            ('standarization', scaler),
            ('knn', knn_bo)
        ])
        
        reg_knn_bo.fit(X_train, y_train) # Training the model
        y_val_pred_knn_bo = reg_knn_bo.predict(X_val) # Making predictions on the validate set 
        mae = metrics.mean_absolute_error(y_val, y_val_pred_knn_bo) # Evaluate the model
        return mae
    return objective

def param_search_bo(train, validation, study, n_trials):
    objective = create_objective(train, validation)
    trials = [trial for trial in study.get_trials() if trial.state == TrialState.COMPLETE]
    n_trials = max(0, n_trials-len(trials))
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, catch=(Exception, ))
    best_params = load_best_params(study)
    return best_params

def load_best_params(study):
    try:
        return study.best_params
    except Exception as e:
        print('Study does not exist')
        raise e

def write_trials_to_csv(trials, csv_file_path):
    """ Write all trials information to a CSV file """
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
    
        # Write header
        csv_writer.writerow(['trial_number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state', 'params'])
    
        # Write data for each trial
        for trial in trials:
            csv_writer.writerow([
                trial.number,
                trial.value,
                trial.datetime_start,
                trial.datetime_complete,
                trial.duration,
                trial.state,
                trial.params
            ])