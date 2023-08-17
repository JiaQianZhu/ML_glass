"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to filter out the non-missing portions of density and proceed with training.
"""


import pickle
import numpy as np
import pandas as pd


data = pd.read_csv('../data/raw/data.csv')
############################################################################
#                                RF + 10_cv                                #
############################################################################

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# Filter out the non-missing data points.
non_empty_df = data.dropna(subset=['Density293K'])

# Define input features and target variables.
features = non_empty_df[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
target = non_empty_df['Density293K']

# Split the dataset into training, validation, and test sets with 70:15:15 portions.
train_features, remaining_features, train_target, remaining_target = train_test_split(features, target, test_size=0.3,
                                                                                      random_state=42)
val_features, test_features, val_target, test_target = train_test_split(remaining_features, remaining_target,
                                                                        test_size=0.5, random_state=42)

# Create a Random Forest regression model.
rf_model = RandomForestRegressor()

# Fit the model on the training set.
rf_model.fit(train_features, train_target)

# Perform cross-validation.
cv_scores = cross_val_score(rf_model, features, target, cv=10, scoring='neg_mean_squared_error')

# Convert to positive values.
cv_scores = -cv_scores

# Calculate the mean and standard deviation of cross-validation evaluation metrics.
cv_mse_mean = np.mean(cv_scores)
cv_mse_std = np.std(cv_scores)

# Make predictions on the validation set.
val_predictions = rf_model.predict(val_features)

# Calculate evaluation metrics on the validation set.
val_mse = mean_squared_error(val_target, val_predictions)
val_r2 = r2_score(val_target, val_predictions)
val_mae = mean_absolute_error(val_target, val_predictions)
val_medae = median_absolute_error(val_target, val_predictions)
val_rd = np.mean(np.abs((val_target - val_predictions) / val_target)) * 100

#  Make predictions on the test set.
test_predictions = rf_model.predict(test_features)

# Calculate evaluation metrics on the test set.
test_mse = mean_squared_error(test_target, test_predictions)
test_r2 = r2_score(test_target, test_predictions)
test_mae = mean_absolute_error(test_target, test_predictions)
test_medae = median_absolute_error(test_target, test_predictions)
test_rd = np.mean(np.abs((test_target - test_predictions) / test_target)) * 100


print("Cross-Validation Mean Squared Error:", cv_mse_mean)
print("Cross-Validation MSE Standard Deviation:", cv_mse_std)

print("\nValidation Set:")
print("Mean Squared Error:", val_mse)
print("R^2 Score:", val_r2)
print("Mean Absolute Error:", val_mae)
print("Median Absolute Error:", val_medae)
print("Relative Deviation (%):", val_rd)

print("\nTest Set:")
print("Mean Squared Error:", test_mse)
print("R^2 Score:", test_r2)
print("Mean Absolute Error:", test_mae)
print("Median Absolute Error:", test_medae)
print("Relative Deviation (%):", test_rd)


# Save the model as a pickle file.
with open('../model/individual_prediction/density_rf_model.pickle', 'wb') as file:
    pickle.dump(rf_model, file)
