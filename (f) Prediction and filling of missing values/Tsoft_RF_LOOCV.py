"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to filter out the non-missing portions of Tsoft and conduct training using the RF algorithm with LOOCV.
"""


import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('../(a-d) Datasets/raw/data.csv')
############################################################################
#                               LOOCV + RF                                 #
############################################################################

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import numpy as np

# Filter out the non-missing data_pred points.
non_empty_df = data.dropna(subset=['Tsoft'])

# Create an empty list to store the predictions for each sample.
all_predictions = []

# Create an empty list to store the actual values for each sample.
all_targets = []

# Use the leave-one-out cross-validation method.
for index, row in non_empty_df.iterrows():
    # Select the current sample from the dataset as the testing sample.
    test_features = row[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']].values.reshape(1, -1)
    test_target = row['Tsoft']

    # Exclude the current sample from the dataset to use it as a training sample.
    train_features = non_empty_df.drop(index)[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
    train_target = non_empty_df.drop(index)['Tsoft']

    # Create a Random Forest regression model.
    rf_model = RandomForestRegressor()

    # Fit the model.
    rf_model.fit(train_features, train_target)

    # Make predictions.
    prediction = rf_model.predict(test_features)

    # Store the predicted results and the actual values.
    all_predictions.append(prediction)
    all_targets.append(test_target)

# Convert the predicted results and actual values into NumPy arrays.
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Calculate evaluation metrics.
mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
medae = median_absolute_error(all_targets, all_predictions)
rd = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
r2 = r2_score(all_targets, all_predictions)


print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Median Absolute Error:", medae)
print("Relative Difference (%):", rd)
print("R^2 Score:", r2)
# Save the model as a pickle file.
with open('../model/individual_prediction/Tsoft_rf_model.pickle', 'wb') as file:
    pickle.dump(rf_model, file)