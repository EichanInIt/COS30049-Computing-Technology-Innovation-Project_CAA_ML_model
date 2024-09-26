import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import pickle


# Load the data
dropped_data = os.path.join("dataset", "2015-Cleaned_flight_data.csv")
dropped_data = pd.read_csv(dropped_data)

# Measure start time
start_time = time.time()

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_ARRIVAL"]

Target_cols = [ "TAXI_OUT", "WHEELS_OFF", "AIR_TIME", "WHEELS_ON", "TAXI_IN"]

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(dropped_data[numerical_cols]), columns=numerical_cols)

# Apply OneHotEncoder to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded = pd.DataFrame(encoder.fit_transform(dropped_data[categorical_cols]),
                                   columns=encoder.get_feature_names_out(categorical_cols))

# Combine scaled numerical data, encoded categorical data, and target (ARRIVAL_DELAY)
final_data = pd.concat([numerical_scaled, categorical_encoded, dropped_data[Target_cols].reset_index(drop=True)], axis=1)

# Measure end time
end_time = time.time()
running_time = end_time - start_time
print(f"Total encoding: {running_time:.2f} seconds")

# #Optionally, saving the processed data to a new CSV file
# final_data.to_csv("Processed_flight_data.csv", index=False)

# #Display the first few rows of the processed dataset
# print(final_data.head())

# Split data into features and target
x = final_data.drop(columns=Target_cols)  # Features
y = final_data[Target_cols]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=108)

