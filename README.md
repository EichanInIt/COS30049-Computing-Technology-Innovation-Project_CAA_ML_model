# COS30049 - Computing Technology Innovation Project (CAA ML Model)

# Installation

To install all the dependencies, ensure that the required packages listed in `requirements.txt` are installed. Use the following command:

    pip install -r requirements.txt

Note: Ensure that you are in the correct directory when running this command to avoid any path-related issues.

## Delay Training Model Training

The model is trained using flight data from the `flights.csv` dataset, located in the `dataset` folder. Follow the following instruction to transform and train the model.

Run the following command to preprocess and clean the data

    python flights_data_transformation.py

After running the script, a cleaned dataset named `2015-Cleaned_flight_data.csv` will be generated and saved in the dataset folder for use in model training.

Run the following command to start training and generating model. 

    python DelayTraining.py

After running the previous command, a pkl file will be generated in `model` folder, called `2015-LinearRegression_FlightDelay.pkl` with the. The model is trained using a type of supervised Machine Learning Algorithm called Linear Regression. Optionally, the `DelayTraining_Model_Decision.py` can also be executed to display multiple ML algorithm attempt.

    python DelayTraining_Model_Decision.py

## Flight Fare Model Training

The model is trained using flight fare data from the `Cleaned_Dataset.csv` dataset, which is located in the `dataset` folder.

### Important Notes:
- The dataset originally does not include the `price_aud` feature. To generate this feature, you need to uncomment the relevant code in the `RegressionForFlightFare.py` file.

Run the following command to preprocess the dataset and train the flight fare model:
 
    python RegressionForFlightFare.py
