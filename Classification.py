import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
from ydata_profiling import ProfileReport

# Load the data
dropped_data_path = os.path.join("dataset", "2015-Cleaned_flight_data_with_delay_rating.csv")
dropped_data = pd.read_csv(dropped_data_path)

# # Define a function to classify delay
# def classify_delay(delay):
#     if delay <= 0:
#         return 'On Time'
#     else:
#         return 'Delayed'

# # Apply the function to create a new column 'DELAY_RATING'
# dropped_data['DELAY_RATING'] = dropped_data['ARRIVAL_DELAY'].apply(classify_delay)

# # Save the updated DataFrame with the new 'DELAY_RATING' column to a CSV file
# dropped_data.to_csv("2015-Cleaned_flight_data_with_delay_rating.csv", index=False)

# # Generate a profiling report with ydata-profiling
# profile = ProfileReport(dropped_data, title="Flight Data Profiling Report", explorative=True)
# profile.to_file("Visualization/clf_flight_data_profile_report.html")

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY","AIR_TIME", "DISTANCE","SCHEDULED_ARRIVAL", "ARRIVAL_DELAY", "WHEELS_ON", "TAXI_IN", "TAXI_OUT", "WHEELS_OFF"]

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(dropped_data[numerical_cols]), columns=numerical_cols)

# Apply LabelEncoder to categorical columns
label_encoders = {}
categorical_encoded = pd.DataFrame()
for col in categorical_cols:
    le = LabelEncoder()
    categorical_encoded[col] = le.fit_transform(dropped_data[col])
    label_encoders[col] = le

# Combine scaled numerical data and encoded categorical data
final_data = pd.concat([numerical_scaled, categorical_encoded, dropped_data["DELAY_RATING"].reset_index(drop=True)], axis=1)

# Label encode the 'DELAY_RATING' column
label_encoder = LabelEncoder()
final_data['DELAY_RATING_ENCODED'] = label_encoder.fit_transform(final_data['DELAY_RATING'])

# Drop the 'DELAY_RATING' column (original) and use the encoded one
X = final_data.drop(columns=["DELAY_RATING", "DELAY_RATING_ENCODED", "ARRIVAL_DELAY"])  # Features
Y = final_data["DELAY_RATING_ENCODED"]  # Encoded target

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=108)

# Check the original class distribution
print(f"Original class distribution: {Counter(Y_train)}")

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=108)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# # Check the new class distribution after resampling
# print(f"Resampled class distribution: {Counter(Y_train_resampled)}")
# # Apply undersampling to reduce the "On Time" class (class 1) to match class 0's size
# undersample = RandomUnderSampler(sampling_strategy={1: 101688}, random_state=108)
# X_train_resampled, Y_train_resampled = undersample.fit_resample(X_train, Y_train)

# Check the new class distribution after resampling
print(f"Resampled class distribution: {Counter(Y_train_resampled)}")

# Implement RandomForestClassifier with optimized parameters
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced', 
    random_state=108,
    n_jobs = -1
)
# Perform cross-validation to evaluate the model
cv_scores = cross_val_score(rf_model, X_train_resampled, Y_train_resampled, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")

# Fit the model with resampled data
rf_model.fit(X_train_resampled, Y_train_resampled)

# Predict on the test data
Y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report (Precision, Recall, F1-score)
print(classification_report(Y_test, Y_pred, target_names=label_encoder.classes_))

# Save the model using pickle
model_filename = 'model/rfclassifier_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(rf_model, model_file)
print(f"Model saved as {model_filename}")

# Predict on the test data
Y_pred = rf_model.predict(X_test)

# Original class distribution: Counter({1: 139632, 0: 101688})
# Resampled class distribution: Counter({0: 101688, 1: 101688})
# Accuracy: 0.8479
#               precision    recall  f1-score   support

#      Delayed       0.84      0.79      0.81     25493
#      On Time       0.85      0.89      0.87     34838

#     accuracy                           0.85     60331
#    macro avg       0.85      0.84      0.84     60331
# weighted avg       0.85      0.85      0.85     60331

# Model saved as rfclassifier_model.pkl
# Accuracy: 0.8479

# #GridSearchCV
# # Initialize the RandomForestClassifier
# rf = RandomForestClassifier(random_state=108)

# # Use GridSearchCV to find the best parameters
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
#                            cv=3, n_jobs=-1, verbose=2, scoring='recall')

# # Fit the grid search to the resampled training data
# grid_search.fit(X_train_resampled, Y_train_resampled)

# # Output the best parameters found by GridSearchCV
# print(f"Best Parameters: {grid_search.best_params_}")

# # Predict using the best model
# best_rf_model = grid_search.best_estimator_
# Y_pred = best_rf_model.predict(X_test)

# # Evaluate the model's performance
# accuracy = accuracy_score(Y_test, Y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# # Detailed classification report (Precision, Recall, F1-score)
# print(classification_report(Y_test, Y_pred, target_names=label_encoder.classes_))

# # Check if grid_search attributes are correctly set
# print(f"Best Parameters Combination: {grid_search.best_params_}")

# Fitting 3 folds for each of 81 candidates, totalling 243 fits
# Best Parameters: {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

