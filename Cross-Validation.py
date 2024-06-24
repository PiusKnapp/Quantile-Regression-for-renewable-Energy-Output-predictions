import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Read the CSV file into Pandas DataFrames
merged_data = pd.read_csv('merged_daily_data.csv')
merged_data['day'] = pd.to_datetime(merged_data['day'])

# Clean data: Remove NaNs
merged_data = merged_data.replace([np.inf, -np.inf], np.nan).dropna()

# Split the data into training and validation sets
train_data = merged_data[merged_data['day'].dt.year != 2022]

# All features: 'cdir', 'z', 'msl', 'blh', 'tcc', 'u10', 'v10', 't2m', 'ssr', 'tsr', 'sund', 'tp', 'fsr', 'u100',
# 'v100', 'wind_speed_avg'

# Define features and target variable for training
features = ['msl', 'blh', 'u10', 'v10', 't2m', 'tp', 'u100', 'v100', 'wind_speed_avg']
target = 'Wind_Total'

# Cross-Validation with Feature Subsets
feature_combinations = combinations(features, 5)
best_score = float('inf')
best_combination = None

for comb in feature_combinations:
    comb = list(comb)
    model = LinearRegression()
    scores = cross_val_score(model, train_data[comb], train_data[target], cv=5, scoring='neg_mean_absolute_error')
    score = -scores.mean()

    if score < best_score:
        best_score = score
        best_combination = comb

print(f'Best combination (Cross-Validation with Feature Subsets): {best_combination} with MAE: {best_score}')
