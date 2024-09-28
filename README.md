# COS30049 - Computing Technology Innovation Project (CAA ML Model)

## Installation

To install all the dependencies, ensure that the required packages listed in `requirements.txt` are installed. Use the following command:

    pip install -r requirements.txt

Note: Ensure that you are in the correct directory when running this command to avoid any path-related issues.

## Delay Training Model

The model is trained using flight data from the `flights.csv` dataset, which is located in the `dataset` folder. Follow the steps below to preprocess the data and train the model.

### Steps:

1. Run the following command to preprocess and clean the data:

    python flights_data_transformation.py

   This script will generate a file named `2015-Cleaned_flight_data.csv`, which will be saved in the `dataset` folder.

2. To start training the model and generating the output, run the following command:

    python DelayTraining.py

   After running the script, a `.pkl` file named `2015-LinearRegression_FlightDelay.pkl` will be created in the `model` folder. The model is trained using a supervised Machine Learning algorithm called **Linear Regression**.

### Optional:
You can also run the following script to experiment with different machine learning algorithms:

    python DelayTraining_Model_Decision.py

This script will display results from multiple machine learning algorithm attempts.

## Flight Fare Model Training

The model is trained using flight fare data from the `Cleaned_Dataset.csv` dataset, which is located in the `dataset` folder.

### Important Notes:
- The dataset originally does not include the `price_aud` feature. To generate this feature, you need to uncomment the relevant code in the `RegressionForFlightFare.py` file.

Run the following command to preprocess the dataset and train the flight fare model:
    
    python RegressionForFlightFare.py

After running the script, a `.pkl` file named `rf_regressor.pkl` will be created in the `model` folder. The result is also visualized via 3 png images generated in `Visualization` named respectively `actual_vs_predicted.png`, `feature_importance.png`, `residual_plot.png`. The model is trained using a supervised Machine Learning algorithm called **RandomForestRegressor**
 
