import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import pickle
import numpy as np

# Load the data
dropped_data = os.path.join("dataset", "2015-Cleaned_flight_data.csv")
dropped_data = pd.read_csv(dropped_data)

# Measure start time
start_time = time.time()

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN", "SCHEDULED_ARRIVAL"]

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

# #Optionally, saving the processed data to a new CSV file
# final_data.to_csv("Processed_flight_data.csv", index=False)

# #Display the first few rows of the processed dataset
# print(final_data.head())

# Split data into features and target
x = final_data.drop(columns=["ARRIVAL_DELAY"])  # Features
y = final_data["ARRIVAL_DELAY"]  # Target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=108)

# Initialize the Linear Regression model
model = LinearRegression()

# Measure start time for training
train_start_time = time.time()

# Train the Linear Regression model
model.fit(X_train, y_train)

# Measure end time for training
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"Total training time: {train_time:.2f} seconds")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Output the results
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Save the trained model to a pickle file
with open("model/2015-LinearRegression_FlightDelay.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as '2015-LinearRegression_FlightDelay.pkl'")

# Total encoding: 1.53 seconds
# Total training time: 10.47 seconds
# R-squared (R²): 0.98
# Mean Absolute Error (MAE): 2.11
# Mean Squared Error (MSE): 14.78
# Model saved as '2015-LinearRegression_FlightDelay.pkl'
