import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
import seaborn as sns

# Load the data
dropped_data = os.path.join("dataset", "Cleaned_flight_data.csv")
dropped_data = pd.read_csv(dropped_data)

# Checking total NA
# data.isna().sum()

# data.describe()

# Adjust delay calculations for edge cases
def calculate_departure_delay(departure_time, scheduled_departure):
    # If DEPARTURE_TIME is greater than SCHEDULED_DEPARTURE but it's still the day before
    if departure_time > 1300 and scheduled_departure < 300:
        return (departure_time - 1440) - scheduled_departure  # Adjust by subtracting a full day (1440 minutes)
    else:
        return departure_time - scheduled_departure

def calculate_arrival_delay(arrival_time, scheduled_arrival):
    # If ARRIVAL_TIME is after midnight but the scheduled arrival was late the day before
    if arrival_time < 300 and scheduled_arrival > 1300:
        return arrival_time + 1440 - scheduled_arrival  # Adjust by adding a full day (1440 minutes)
    else:
        return arrival_time - scheduled_arrival

# Feature Engineering
# AIR_TIME = (WHEELS_OFF - WHEELS_ON) + (DESTINATION_TZ - SOURCE_TZ) * 60
dropped_data.loc[:, 'AIR_TIME'] = (dropped_data['WHEELS_ON'] - dropped_data['WHEELS_OFF']) + ((dropped_data['ORIGIN_AIRPORT_TZ'] - dropped_data['DESTINATION_AIRPORT_TZ']) * 60)

# ELAPSED_TIME = AIR_TIME + TAXI_OUT + TAXI_IN
dropped_data.loc[:, 'ELAPSED_TIME'] = dropped_data['AIR_TIME'] + dropped_data['TAXI_OUT'] + dropped_data['TAXI_IN']

# Calculate DEPARTURE_DELAY using the custom function to handle edge cases
dropped_data.loc[:, 'DEPARTURE_DELAY'] = dropped_data.apply(
    lambda row: calculate_departure_delay(row['DEPARTURE_TIME'], row['SCHEDULED_DEPARTURE']),
    axis=1
)

# Calculate ARRIVAL_DELAY using the custom function to handle edge cases
dropped_data.loc[:, 'ARRIVAL_DELAY'] = dropped_data.apply(
    lambda row: calculate_arrival_delay(row['ARRIVAL_TIME'], row['SCHEDULED_ARRIVAL']),
    axis=1
)

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY_OF_WEEK", 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
                  "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE", "TAXI_IN", "SCHEDULED_ARRIVAL"]

# Drop unnecessary columns (if needed)
dropped_data = dropped_data[categorical_cols + numerical_cols + ["ARRIVAL_DELAY"]]

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(dropped_data[numerical_cols]), columns=numerical_cols)

# Apply OneHotEncoder to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded = pd.DataFrame(encoder.fit_transform(dropped_data[categorical_cols]),
                                   columns=encoder.get_feature_names_out(categorical_cols))

# Combine scaled numerical data, encoded categorical data, and target (ARRIVAL_DELAY)
final_data = pd.concat([numerical_scaled, categorical_encoded, dropped_data["ARRIVAL_DELAY"].reset_index(drop=True)], axis=1)

# Now `final_data` is ready for modeling
final_data.describe()

# Optionally, you can save the processed data to a new CSV file
# final_data.to_csv("Processed_flight_data.csv", index=False)

# Display the first few rows of the processed dataset
print(final_data.head())
