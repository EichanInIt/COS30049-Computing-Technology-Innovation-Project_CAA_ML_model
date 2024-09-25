import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.model_selection import train_test_split
from lazypredict import LazyRegressor
import time
import pickle

# Load the data
dropped_data = os.path.join("dataset", "2015-Cleaned_flight_data.csv")
dropped_data = pd.read_csv(dropped_data)

# # Measure start time
# start_time = time.time()

# # Adjust delay calculations for edge cases
# def calculate_departure_delay(departure_time, scheduled_departure):
#     # If DEPARTURE_TIME is greater than SCHEDULED_DEPARTURE but it's still the day before
#     if departure_time > 1300 and scheduled_departure < 300:
#         return (departure_time - 1440) - scheduled_departure  # Adjust by subtracting a full day (1440 minutes)
#     else:
#         return departure_time - scheduled_departure

# def calculate_arrival_delay(arrival_time, scheduled_arrival):
#     # If ARRIVAL_TIME is after midnight but the scheduled arrival was late the day before
#     if arrival_time < 300 and scheduled_arrival > 1300:
#         return arrival_time + 1440 - scheduled_arrival  # Adjust by adding a full day (1440 minutes)
#     else:
#         return arrival_time - scheduled_arrival

# Feature Engineering
# AIR_TIME = (WHEELS_OFF - WHEELS_ON) + (DESTINATION_TZ - SOURCE_TZ) * 60
#dropped_data.loc[:, 'AIR_TIME'] = (dropped_data['WHEELS_ON'] - dropped_data['WHEELS_OFF']) + ((dropped_data['ORIGIN_AIRPORT_TZ'] - dropped_data['DESTINATION_AIRPORT_TZ']) * 60)

# # Calculate DEPARTURE_DELAY using the custom function to handle edge cases
# dropped_data.loc[:, 'DEPARTURE_DELAY'] = dropped_data.apply(
#     lambda row: calculate_departure_delay(row['DEPARTURE_TIME'], row['SCHEDULED_DEPARTURE']),
#     axis=1
# )

# # Calculate ARRIVAL_DELAY using the custom function to handle edge cases
# dropped_data.loc[:, 'ARRIVAL_DELAY'] = dropped_data.apply(
#     lambda row: calculate_arrival_delay(row['ARRIVAL_TIME'], row['SCHEDULED_ARRIVAL']),
#     axis=1
# )

# # Measure end time
# end_time = time.time()
# running_time = end_time - start_time
# print(f"Total Feature engineering: {running_time:.2f} seconds")


# Measure start time
start_time = time.time()

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY"]

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(dropped_data[numerical_cols]), columns=numerical_cols)

# Apply OneHotEncoder to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded = pd.DataFrame(encoder.fit_transform(dropped_data[categorical_cols]),
                                   columns=encoder.get_feature_names_out(categorical_cols))

# Combine scaled numerical data, encoded categorical data, and target (ARRIVAL_DELAY)
final_data = pd.concat([numerical_scaled, categorical_encoded, dropped_data["ARRIVAL_DELAY"].reset_index(drop=True)], axis=1)

# Measure end time
end_time = time.time()
running_time = end_time - start_time
print(f"Total encoding: {running_time:.2f} seconds")


# #Optionally, you can save the processed data to a new CSV file
# final_data.to_csv("Processed_flight_data.csv", index=False)

# #Display the first few rows of the processed dataset
# print(final_data.head())

# Split data into features and target
x = final_data.drop(columns=["ARRIVAL_DELAY"])  # Features
y = final_data["ARRIVAL_DELAY"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=108)

# Initialize LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Measure start time
start_time = time.time()

# Fit and evaluate multiple regression models
print("Start trainning")
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Measure end time
end_time = time.time()

# Calculate running time
running_time = end_time - start_time
print(f"Total running time: {running_time:.2f} seconds")


# Save the models and predictions to a pickle file
with open("model/2015-Flight_lazypredict_models.pkl", "wb") as f:
    pickle.dump((models, predictions), f)

# Print the performance of the models directly to the terminal
print(models)
