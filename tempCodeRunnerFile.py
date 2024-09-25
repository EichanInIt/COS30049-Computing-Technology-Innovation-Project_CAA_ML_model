# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=108)

# # Initialize LazyRegressor
# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# # Measure start time
# start_time = time.time()

# # Fit and evaluate multiple regression models
# print("Start trainning")
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# # Measure end time
# end_time = time.time()

# # Calculate running time
# running_time = end_time - start_time
# print(f"Total running time: {running_time:.2f} seconds")


# # Save the models and predictions to a pickle file
# with open("model/2015-Flight_lazypredict_models.pkl", "wb") as f:
#     pickle.dump((models, predictions), f)

# # Print the performance of the models directly to the terminal
# print(models)
