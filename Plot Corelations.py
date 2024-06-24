import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.utils import resample

# Read the CSV file into Pandas DataFrame
merged_data = pd.read_csv('merged_hourly_data.csv')
merged_data['hour'] = pd.to_datetime(merged_data['hour'])

# Define the weather variable and target variable
weather_variable = 'v100'
target = 'Wind_Total'

# Clean data: Remove NaNs
data = merged_data[[weather_variable, target]].dropna()

# Subsample the data to reduce density (keep 0.5% of the data)
subsample_fraction = 0.005
data_sampled = resample(data, n_samples=int(len(data) * subsample_fraction), random_state=1)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_sampled[weather_variable], y=data_sampled[target], s=10)
plt.title(f'{weather_variable} vs {target}')
plt.xlabel(weather_variable)
plt.ylabel(target)

# Fit and plot the linear regression trend line
x = data_sampled[[weather_variable]].values
y = data_sampled[target].values
reg = LinearRegression().fit(x, y)
trend_line = reg.predict(x)

# Sorting once for plotting
sorted_idx = np.argsort(x.flatten())
sorted_x = x[sorted_idx]

plt.plot(sorted_x, trend_line[sorted_idx], color='red', linestyle='--', linewidth=2, label='Linear Regression')

# Fit and plot the quantile regression lines
quantiles = [0.1, 0.5, 0.9]
colors = ['blue', 'green', 'yellow']
labels = [f'Quantile {q}' for q in quantiles]

for q, color, label in zip(quantiles, colors, labels):
    qr = QuantileRegressor(quantile=q, solver='highs', alpha=0).fit(x, y)
    quantile_line = qr.predict(sorted_x)
    plt.plot(sorted_x, quantile_line, color=color, linestyle='-', linewidth=2, label=label)

# Add legend
plt.legend()

# Show plot
plt.show()
