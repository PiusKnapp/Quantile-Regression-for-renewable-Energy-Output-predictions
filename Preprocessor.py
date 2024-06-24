import pandas as pd
import numpy as np

"""
This Preprocessor preprocesses all necessary data for later ML tasks. 
It gets rid of unnecessary data, handles the german daylight savings and stores all necessary forecast and 
supply data in one hourly csv file.
"""

# Load the CSV files into Pandas DataFrames
weather = pd.read_csv('Weather_Data_Germany.csv')
weather_2022 = pd.read_csv('Weather_Data_Germany_2022.csv')
realised_supply = pd.read_csv('Realised_Supply_Germany.csv', sep=';')

# Strip any extra whitespace from column names
realised_supply.columns = realised_supply.columns.str.strip()

# Drop unneeded columns from realised_supply
realised_supply = realised_supply.drop(['Biomass [MW]', 'Hydro Power [MW]', 'Other Renewable [MW]',
                                        'Nuclear Power [MW]', 'Lignite [MW]', 'Coal [MW]',
                                        'Natural Gas [MW]', 'Pumped Storage [MW]', 'Other Conventional [MW]'], axis=1)

# Concatenate weather data for all years
weather = pd.concat([weather, weather_2022], ignore_index=True)

# Convert date and time columns to datetime format
weather['forecast_origin'] = pd.to_datetime(weather['forecast_origin'])
weather['time'] = pd.to_datetime(weather['time'])

# Create new date range covering the weather data period at 15-minute intervals
date_rng = pd.date_range(start='2019-01-01 00:00', end='2022-12-31 23:45', freq='15min')

# Replace Date_from in realised_supply with this new date range
realised_supply['Date_from'] = date_rng
realised_supply['Date_from'] = pd.to_datetime(realised_supply['Date_from'])

# Replace , with . and cast to float
realised_supply['Wind Offshore [MW]'] = realised_supply['Wind Offshore [MW]'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
realised_supply['Wind Onshore [MW]'] = realised_supply['Wind Onshore [MW]'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
realised_supply['Photovoltaic [MW]'] = realised_supply['Photovoltaic [MW]'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)

# Calculate Wind_Total
realised_supply['Wind_Total'] = realised_supply['Wind Offshore [MW]'] + realised_supply['Wind Onshore [MW]']

# Set Date_from as index for resampling
realised_supply.set_index('Date_from', inplace=True)

# Resample supply data to hourly intervals, summing the 15-minute data
resampled_supply = realised_supply.resample('H').sum()

# Align supply data to the start of each hour
resampled_supply.reset_index(inplace=True)
resampled_supply['hour'] = resampled_supply['Date_from'].dt.floor('H')

# Create hourly forecast data
weather['hour'] = weather['time'].dt.floor('H')
hourly_forecast = weather.groupby(['longitude', 'latitude', 'hour']).mean().reset_index()

# Merge the hourly forecast with hourly supply
merged_data_list = []

for (longitude, latitude), group in hourly_forecast.groupby(['longitude', 'latitude']):
    group = group.copy()
    merged = pd.merge(group, resampled_supply, on='hour', how='inner')
    if merged.empty:
        print(f"No matching hours for longitude {longitude} and latitude {latitude}")
    else:
        merged['longitude'] = longitude
        merged['latitude'] = latitude
        merged_data_list.append(merged)

# Concatenate all the merged data
merged_data = pd.concat(merged_data_list, ignore_index=True)

# Get rid of Date_from
merged_data = merged_data.drop(['Date_from'], axis=1)

# Calculate wind speed and direction from u10 and v10
merged_data['wind_speed_10'] = np.sqrt(merged_data['u10']**2 + merged_data['v10']**2)
merged_data['wind_direction_10'] = np.arctan2(merged_data['v10'], merged_data['u10'])

# Calculate wind speed and direction from u100 and v100
merged_data['wind_speed_100'] = np.sqrt(merged_data['u100']**2 + merged_data['v100']**2)
merged_data['wind_direction_100'] = np.arctan2(merged_data['v100'], merged_data['u100'])

# Combine wind speed at 10 and 100 meters
merged_data['wind_speed_avg'] = (merged_data['wind_speed_10'] + merged_data['wind_speed_100']) / 2

# Save the merged data to CSV
merged_data.to_csv('merged_hourly_data.csv', index=False)

# Add a 'day' column based on the 'hour' column
merged_data['day'] = merged_data['hour'].dt.date

# Group by 'day', 'longitude', 'latitude' and sum the values for each group
daily_data = merged_data.groupby(['day', 'longitude', 'latitude']).sum().reset_index()

# Save the daily data to CSV
daily_data.to_csv('merged_daily_data.csv', index=False)


