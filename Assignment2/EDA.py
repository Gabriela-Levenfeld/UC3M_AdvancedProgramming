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
import seaborn as sns
import matplotlib.pyplot as plt



wind_ava = pd.read_csv('data/wind_available.csv.gzip', compression="gzip") # Read train data

#----------------------------------------------------------
# Question 2: SIMPLIFIED EDA

# How many features and instances are?
wind_ava.shape # Tamaño del dataset: (4748, 555)
wind_ava.info() # The goal is to get a quick and general summary

print('Number of features:', wind_ava.columns.size)
print('Number of instances:', len(wind_ava.iloc[:,0]))

# Different function that maybe useful if we had less observations
# Quitarlo?
wind_ava.head()
wind_ava.columns
wind_ava.dtypes
#.head(); vemos que tenemos [5 rows x 555 columns], no todos los datos aparecen porque hay demasiados.
#columns; la idea es ver qué features tenemos, pero como hay 555 no aparecen todas.


# Which vars are categorical/numeric? -> Parece que todas son numéricas
numeric_vars = wind_ava.select_dtypes(include=[np.number])
numeric_vars.columns

categorical_features = wind_ava.select_dtypes(include=[object])
categorical_features
# Output: Empty DataFrame [4748 rows x 0 columns] -> There is no categorical features

# Data type for just one point on the grid: point 13 -> Location of Sotavento
for column in wind_ava.columns:
    if column[-2:] == '13' or column in ['energy', 'year', 'month', 'day', 'hour']:
        print(column, wind_ava[column].dtypes)


# Identificamos el número de missing values en cada columna
wind_ava.isnull().sum()

# Total NA on the dataset: 326132
total_NA = wind_ava.isnull().sum().sum()
total_NA/(555*4748)
# Missing values for just Sotavento point
for column in wind_ava.columns:
    if column[-2:] == '13' or column in ['energy', 'year', 'month', 'day', 'hour']:
        print(column, wind_ava[column].isnull().sum())

# We can also identify missing values in a graphycally way
# Missing values for Sotavento point
    # Filtering Sotavento point data
sotavento_columns = [column for column in wind_ava.columns if column[-2:] == '13' or column in ['energy', 'year', 'month', 'day', 'hour']]
sotavento_data = wind_ava[sotavento_columns]
    # Plot heatmap
plt.figure()
sns.heatmap(sotavento_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values heatmap for Sotavento point')
plt.show()
# Missing values for all dataset
sns.heatmap(wind_ava.isnull(), cbar=False)


# Check if there are duplicate data -> No duplicate data
exist_duplicates = wind_ava.duplicated().any()
if exist_duplicates:
    print("There are duplicate data in the dataset.")
else:
    print("There are no duplicate data in the dataset.")
    


# Data visualisation: Identify any trends in energy production
df = wind_ava.copy()
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df = df.sort_values('datetime')

# Plotting energy over time
plt.plot(df['datetime'], df['energy'])
plt.title('Wind Energy Production Over Time')
plt.xlabel('Time')
plt.ylabel('Energy Production')
plt.xticks(rotation=45)
plt.show()

# We observed 2008 year present weird results. However this makes sense because
#during this year just have 178 record over the 1200 aprox. from the other years

# Now, we plot energy just for 2008 year
df_2008 = df[df['year'] == 2008]
df_2008 = df_2008.sort_values('datetime')

# Plotting energy over time
plt.plot(df_2008['datetime'], df_2008['energy'])
plt.title('2008 - Wind Energy Production')
plt.xlabel('Time')
plt.ylabel('Energy Production')
plt.xticks(rotation=45)
plt.show()