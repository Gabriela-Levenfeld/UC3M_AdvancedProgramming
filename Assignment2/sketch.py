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

# Models and metrics
# ===================

# Data processing: aquí irían las librerías de pipeline, transformación de datos,...
# ================



wind_ava = pd.read_csv('data/wind_available.csv.gzip', compression="gzip") # Read train data

#----------------------------------------------------------
# Question 3: SPLIT into TRAIN and TEST

# TODO -> Preprocessing: Handling missing values


# Split dataset
SEED = 100507449
np.random.seed(SEED)

# In order to decide how to split the dataset, we take a look on how data is distribute
year_counts = wind_ava.groupby('year').size()

# Selected years for train_train + Our final split percentage
years_train = [2005, 2006, 2007, 2008]
total_data_count = year_counts.sum()
percentage_train = year_counts[years_train].sum() / total_data_count * 100

# We split the data into train_train (80%) and train_val (20%) dataset.
    # For train_train we take 2005, 2006, 2007 and 2008.
train_train = wind_ava[wind_ava['year'].isin(years_train)]
X_train = train_train.drop(columns='energy')
y_train = train_train['energy'].values
    # For train_validation we take 2009.
train_val = wind_ava[~wind_ava['year'].isin(years_train)]
X_val = train_val.drop(columns='energy')
y_val = train_val['energy'].values
