import pandas as pd
import numpy as np
import os

dropped_data = os.path.join("dataset", "Cleaned_flight_data.csv")
dropped_data = pd.read_csv(dropped_data)

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

