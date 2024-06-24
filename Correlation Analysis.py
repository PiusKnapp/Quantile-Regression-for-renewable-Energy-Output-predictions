import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into Pandas DataFrames
merged_data = pd.read_csv('merged_daily_data.csv')
merged_data['day'] = pd.to_datetime(merged_data['day'])

# Only use the training data from 2019 to 2021
train_data = merged_data[merged_data['day'].dt.year != 2022]

# All features: 'cdir', 'z', 'msl', 'blh', 'tcc', 'u10', 'v10', 't2m', 'ssr', 'tsr', 'sund', 'tp',
# 'fsr', 'u100', 'v100', 'wind_speed_avg'

# Define features and target variable for training
forecast_columns = ['z', 'msl', 'blh', 'u10', 'v10', 't2m', 'tp',
                    'fsr', 'u100', 'v100', 'wind_speed_avg']
features = forecast_columns
target = 'Wind_Total'

# Compute the correlation matrix
corr_matrix = train_data[features + [target]].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Display the correlation of each feature with the target variable
print(corr_matrix[target].sort_values(ascending=False))
