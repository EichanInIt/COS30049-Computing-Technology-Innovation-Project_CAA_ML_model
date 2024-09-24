import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
#from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import pickle

# Load the data
data = pd.read_csv('/Users/minhnguyen/Desktop/Innovation_project/datasets/Clean_Dataset.csv')
data = data.drop(data.columns[0], axis=1)
data['price_aud'] = data['price_rupee'] / 57 

print("1")
# Preprocess the data
le = LabelEncoder()
columns_to_labelencode = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

for column in columns_to_labelencode:
    data[column] = le.fit_transform(data[column])

# Split features and target
X = data.drop(['price_aud', 'price_rupee'], axis=1)
Y = data['price_aud']
print("2")
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=108)
print(len(X_train))
print("3")

# Adjusted: Use the best parameters found from GridSearchCV
model = RandomForestRegressor(
    n_estimators=100,                
    criterion="squared_error",        
    min_samples_split=5,              
    min_samples_leaf=1,               
    random_state=108
)

print("4")
# Train the model with the best parameters
model.fit(X_train, Y_train)
print("5")

# Save the model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Make predictions
Y_pred = model.predict(X_test)
print("6")

# Calculate performance metrics
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
medae = median_absolute_error(Y_test, Y_pred)

# Perform cross-validation
cv_scores = cross_val_score(model, X, Y, cv=3, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print("Performance Metrics:")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Median Absolute Error: {medae:.2f}")
print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Residual plot
residuals = Y_test - Y_pred
plt.figure(figsize=(10, 6))
plt.scatter(Y_pred, residuals, alpha=0.5)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.close()

print("\nVisualization plots have been saved as PNG files.")

# Example prediction
sample = X_test.iloc[0].values.reshape(1, -1)  # Use iloc to select the first row correctly
predicted_price = model.predict(sample)
print(f"\nPredicted price for a sample flight: {predicted_price[0]:.2f}")


#Create the parameter grid using sklearn's ParameterGrid
# grid = ParameterGrid(param_grid)

# Function to evaluate one set of parameters
# def evaluate_params(params, X_train, X_test, Y_train, Y_test):
    # Create a new RandomForestRegressor model
    # model = RandomForestRegressor(random_state=108)
    # model.set_params(**params)
    
    # Fit the model on training data
    # model.fit(X_train, Y_train)
    
    # Predict on test data
    # Y_pred = model.predict(X_test)
    
    # Calculate the negative mean squared error (as we want to maximize the score)
    # score = -mean_squared_error(Y_test, Y_pred)
    
    # return score, params

# Use joblib's Parallel to run evaluations in parallel
# results = Parallel(n_jobs=-1)(
#     delayed(evaluate_params)(params, X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()) 
#     for params in tqdm(grid, desc="Grid Search Progress")
# )

# Find the best result
# best_score, best_params = max(results, key=lambda x: x[0])

# Output best parameters and best score
# print("Best parameters found: ", best_params)
# print("Best score: ", best_score)
#Best parameters found:  {'criterion': 'squared_error', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}


