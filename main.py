import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from datetime import date, datetime

# Read the CSV file into Pandas DataFrames
merged_data = pd.read_csv('merged_hourly_data.csv')
merged_data['hour'] = pd.to_datetime(merged_data['hour'])

# Split the data into training and validation sets
train_data = merged_data[merged_data['hour'].dt.year != 2022]
validation_data = merged_data[merged_data['hour'].dt.year == 2022]

# Subsample the training data to reduce compute time (keep 10% of the data)
subsample_fraction = 0.001
train_data = resample(train_data, n_samples=int(len(train_data) * subsample_fraction), random_state=1)

# Define forecast columns
forecast_columns = ['cdir', 'tcc', 't2m', 'ssr', 'sund']

# Define target variable
target = 'Photovoltaic [MW]'

# Remove any rows with missing or infinite values
train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()
validation_data = validation_data.replace([np.inf, -np.inf], np.nan).dropna()

# Scale the features
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data[forecast_columns])
validation_data_scaled = scaler.transform(validation_data[forecast_columns])

# Scale the target
target_scaler = StandardScaler()
train_target_scaled = target_scaler.fit_transform(train_data[[target]]).flatten()
validation_target_scaled = target_scaler.transform(validation_data[[target]]).flatten()

# Remove any rows with missing or infinite values
"""train_data_scaled = train_data_scaled[:, ~np.isnan(train_data_scaled).any(axis=0)]
train_target_scaled = train_target_scaled[:, ~np.isnan(train_target_scaled).any(axis=0)]
validation_data_scaled = validation_data_scaled[:, ~np.isnan(validation_data_scaled).any(axis=0)]
validation_target_scaled = validation_target_scaled[:, ~np.isnan(validation_target_scaled).any(axis=0)]
"""
# Define the quantile levels for quantile regression and their corresponding alpha values
quantile_alpha_map = {
    0.1: 0.001,
    0.5: 0.001,
    0.9: 0.001
}

# Train quantile regression models with different alphas
models = {}
for quantile, alpha in quantile_alpha_map.items():
    qr = QuantileRegressor(quantile=quantile, alpha=alpha, solver='highs')
    qr.fit(train_data_scaled, train_target_scaled)
    models[quantile] = qr

# Validate the models using the 2022 validation data
predictions = {}
for quantile, model in models.items():
    scaled_predictions = model.predict(validation_data_scaled)
    predictions[quantile] = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()

# Combine predictions into a DataFrame
predictions_df = pd.DataFrame(predictions)
predictions_df['Actual'] = validation_data[target].values
predictions_df['Date'] = validation_data['hour'].values

# Aggregate predictions and actual values by day
predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
aggregated_predictions = predictions_df.groupby(predictions_df['Date'].dt.date).mean()
aggregated_actual = predictions_df.groupby(predictions_df['Date'].dt.date)['Actual'].mean()

# Calculate Mean Absolute Error and Mean Pinball Loss for each quantile
mae = {}
pinball_losses = {}
for quantile in quantile_alpha_map.keys():
    mae[quantile] = mean_absolute_error(aggregated_actual, aggregated_predictions[quantile])
    pinball_losses[quantile] = mean_pinball_loss(aggregated_actual, aggregated_predictions[quantile], alpha=quantile)

print("Mean Absolute Error for each quantile:")
for quantile in quantile_alpha_map.keys():
    print(f"Quantile {quantile}: {mae[quantile]}")

print("\nMean Pinball Loss for each quantile:")
for quantile in quantile_alpha_map.keys():
    print(f"Quantile {quantile}: {pinball_losses[quantile]}")

# Plot the aggregated predictions vs actual values for all quantiles in one plot
plt.figure(figsize=(14, 7))
plt.plot(aggregated_predictions.index, aggregated_actual, label='Actual', color='black', alpha=0.7)

# Plot predictions for each quantile
for quantile in quantile_alpha_map.keys():
    plt.plot(aggregated_predictions.index, aggregated_predictions[quantile], label=f'Quantile {quantile}', linestyle='--', alpha=0.8)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Photovoltaic [MW]')
plt.title('Quantile Regression Prediction vs Actual for Multiple Quantiles')

# Format the x-axis to show months
date_fmt = DateFormatter("%b")
month_locator = MonthLocator()
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.gca().xaxis.set_major_locator(month_locator)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Filter the data for the first week of January 2022
start_date = date(2022, 1, 1)
end_date = date(2022, 1, 7)

first_week_predictions = aggregated_predictions.loc[start_date:end_date]
first_week_actual = aggregated_actual.loc[start_date:end_date]

# Plot the predictions vs actual values for the first week
plt.figure(figsize=(14, 7))
plt.plot(first_week_predictions.index, first_week_actual, label='Actual', color='black', alpha=0.7, marker='o')

# Plot predictions for each quantile
for quantile in quantile_alpha_map.keys():
    plt.plot(first_week_predictions.index, first_week_predictions[quantile], label=f'Quantile {quantile}', linestyle='--', alpha=0.8, marker='o')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Photovoltaic [MW]')
plt.title('Quantile Regression Prediction vs Actual for the First Week of January 2022')

# Format the x-axis to show dates properly
plt.gca().xaxis.set_major_formatter(DateFormatter("%b %d"))
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot for January 1st, 2022
# Filter the data for January 1st, 2022
jan_first_date = datetime(2022, 1, 1)

jan_first_predictions = predictions_df[predictions_df['Date'].dt.date == jan_first_date.date()]

# Group by hour and average to get one line per quantile
hourly_predictions = jan_first_predictions.groupby(jan_first_predictions['Date'].dt.hour).mean()

# Plot the predictions vs actual values for January 1st
plt.figure(figsize=(14, 7))
plt.plot(hourly_predictions.index, hourly_predictions['Actual'], label='Actual', color='black', alpha=0.7, marker='o')

# Plot predictions for each quantile
for quantile in quantile_alpha_map.keys():
    plt.plot(hourly_predictions.index, hourly_predictions[quantile], label=f'Quantile {quantile}', linestyle='--', alpha=0.8, marker='o')
plt.legend()
plt.xlabel('Hour')
plt.ylabel('Photovoltaic [MW]')
plt.title('Quantile Regression Prediction vs Actual for January 1st, 2022')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):02d}:00'))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
