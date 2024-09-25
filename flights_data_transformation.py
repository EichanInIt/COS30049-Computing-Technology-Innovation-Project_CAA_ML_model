import pandas as pd
import os

# Load the data
airports_data = os.path.join("dataset", "airports_with_utc_offsets.csv")
airports_data = pd.read_csv(airports_data)

flights_data = os.path.join("dataset", "flights.csv")
flights_data = pd.read_csv(flights_data)

# Create a mapping from IATA_CODE to UTC_OFFSET
airport_timezone_mapping = airports_data.set_index('IATA_CODE')['UTC_OFFSET'].to_dict()

# Create SOURCE_AIRPORT_TZ and DESTINATION_AIRPORT_TZ columns by mapping IATA_CODE from ORIGIN_AIRPORT and DESTINATION_AIRPORT
flights_data['ORIGIN_AIRPORT_TZ'] = flights_data['ORIGIN_AIRPORT'].map(airport_timezone_mapping)
flights_data['DESTINATION_AIRPORT_TZ'] = flights_data['DESTINATION_AIRPORT'].map(airport_timezone_mapping)

# Dropping rows with missing values in relevant columns
dropped_flight_data = flights_data.dropna(subset=[
    'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 
    'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 
    'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', "ARRIVAL_TIME", "ARRIVAL_DELAY"
])

# Function to convert time points (SCHEDULED_DEPARTURE, DEPARTURE_TIME, etc.) to minutes from midnight
def convert_time_to_minutes(time):
    time = f"{int(time):04d}"  # Ensure the time is 4 digits
    hours, minutes = int(time[:2]), int(time[2:])
    return hours * 60 + minutes

# Convert time-related columns to minutes from midnight using direct assignment
time_columns = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
for col in time_columns:
    dropped_flight_data[col] = dropped_flight_data[col].apply(convert_time_to_minutes)

# Rearrange columns so ORIGIN_AIRPORT_TZ is next to ORIGIN_AIRPORT and DESTINATION_AIRPORT_TZ is next to DESTINATION_AIRPORT
columns = dropped_flight_data.columns.tolist()

# Move ORIGIN_AIRPORT_TZ next to ORIGIN_AIRPORT
origin_airport_index = columns.index('ORIGIN_AIRPORT')
columns.insert(origin_airport_index + 1, columns.pop(columns.index('ORIGIN_AIRPORT_TZ')))

# Move DESTINATION_AIRPORT_TZ next to DESTINATION_AIRPORT
destination_airport_index = columns.index('DESTINATION_AIRPORT')
columns.insert(destination_airport_index + 1, columns.pop(columns.index('DESTINATION_AIRPORT_TZ')))

# Reorder the DataFrame with the new column order
dropped_flight_data = dropped_flight_data[columns]

#dropping redundant columns
column_to_drop = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "YEAR", "ELAPSED_TIME", "SCHEDULED_TIME", "DEPARTURE_TIME", "ARRIVAL_TIME"]
dropped_flight_data = dropped_flight_data.drop(columns= column_to_drop)

# Randomly drop half of the rows
dropped_flight_data = dropped_flight_data.sample(frac=0.3, random_state=108).reset_index(drop=True)

# Save the updated dataframe
dropped_flight_data.to_csv("dataset/2015-Cleaned_flight_data.csv", index=False)

# Display the first few rows to verfy
print(dropped_flight_data.describe)
