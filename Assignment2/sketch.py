# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:42:12 2023

@author: Errasti Domínguez, Nuria and Levenfeld Sabau, Gabriela (SEED = 100507449)
"""

# Data import and manipulation
# ===================================
import pandas as pd
import numpy as np

# Visualization
# =============
import seaborn as sns


wind_ava = pd.read_csv('data/wind_available.csv.gzip', compression="gzip") # Read train data


""""
Question 2: SIMPLIFIED EDA
"""

# How many features and instances are?
wind_ava.shape # Tamaño del dataset: (4748, 555)


wind_ava.head()
wind_ava.columns
wind_ava.dtypes

wind_ava.info() # The goal is to get a quick and general summary

#.head(); vemos que tenemos [5 rows x 555 columns], no todos los datos aparecen porque hay demasiados.
#columns; la idea es ver qué features tenemos, pero como hay 555 no aparecen todas.


# Which vars are categorical/numeric? -> Parece que todas son numéricas
numeric_vars = wind_ava.select_dtypes(include=[np.number])
numeric_vars.columns

categorical_features = wind_ava.select_dtypes(include=[object])
# Output: [4748 rows x 0 columns] -> There is no categorical features

# 13 -> Location of Sotavento
for column in wind_ava.columns:
    if column[-2:] == '13':
        print(column, wind_ava[column].dtypes)


# Identificamos el número de missing values en cada columna
wind_ava.isnull().sum()

for column in wind_ava.columns:
    if column[-2:] == '13':
        print(column, wind_ava[column].isnull().sum())

# Identificamos los missing values visualmente
sns.heatmap(wind_ava.isnull(), cbar=False)
