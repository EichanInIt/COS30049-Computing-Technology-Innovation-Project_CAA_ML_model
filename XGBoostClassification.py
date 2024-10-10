import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
from xgboost import XGBClassifier
from ydata_profiling import ProfileReport

# Load the data
dropped_data_path = os.path.join("dataset", "2015-Cleaned_flight_data_with_delay_rating.csv")
dropped_data = pd.read_csv(dropped_data_path)

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "AIR_TIME", "DISTANCE", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY"]

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

# Use the 'DELAY_RATING' column directly as the target
X = final_data.drop(columns=["DELAY_RATING", "ARRIVAL_DELAY"])  # Features
Y = final_data["DELAY_RATING"].astype(int)  # Target (ensure it's integer for classification)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=108)

# Check the original class distribution
print(f"Original class distribution: {Counter(Y_train)}")

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=108)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Check the new class distribution after resampling
print(f"Resampled class distribution: {Counter(Y_train_resampled)}")

# Calculate the scale_pos_weight for handling imbalance in XGBoost
scale_pos_weight = Counter(Y_train_resampled)[0] / Counter(Y_train_resampled)[1]

# Implement the tuned XGBoost Classifier with native categorical support
xgb_model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=108,
    eval_metric='logloss',
    n_jobs=-1,
    tree_method='hist',  
    enable_categorical=True,  
    colsample_bytree=1.0,  
    learning_rate=0.2,
    max_depth=15,
    min_child_weight=1,
    n_estimators=200,
    subsample=0.8
)

# Fit the model on the resampled training data
xgb_model.fit(X_train_resampled, Y_train_resampled)

# Predict on the test data
Y_pred = xgb_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report (Precision, Recall, F1-score)
print(classification_report(Y_test, Y_pred))

# Save the tuned model using pickle
model_filename = 'model/xgboost_tuned_model_with_categorical.pkl'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
with open(model_filename, 'wb') as model_file:
    pickle.dump(xgb_model, model_file)
print(f"Tuned model with categorical support saved as {model_filename}")

#Use GridsearchCV
# Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}
# Accuracy: 0.8445
#               precision    recall  f1-score   support

#            0       0.85      0.91      0.88     36310
#            1       0.82      0.73      0.77     20830

#     accuracy                           0.84     57140
#    macro avg       0.84      0.82      0.83     57140
# weighted avg       0.84      0.84      0.84     57140

# Tuned model saved as model/xgboost_tuned_model.pkl

#Result
# Original class distribution: Counter({0: 144891, 1: 83669})
# Resampled class distribution: Counter({0: 144891, 1: 144891})
# Accuracy: 0.8445
#               precision    recall  f1-score   support

#            0       0.85      0.91      0.88     36310
#            1       0.82      0.73      0.77     20830

#     accuracy                           0.84     57140
#    macro avg       0.84      0.82      0.83     57140
# weighted avg       0.84      0.84      0.84     57140

# Tuned model with categorical support saved as model/xgboost_tuned_model_with_categorical.pkl
