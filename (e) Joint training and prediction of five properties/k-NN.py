"""
Author: JiaQian Zhu
Date: 2023-8-17
Usage: This module offers a method to train and test on the original dataset (filled) with K-Nearest Neighbors (KNN).
"""

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error

data = pd.read_csv('../(a-d) Datasets/clean/dataset_completed.csv')

# Define input features and target variables
features = data[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
targets = ['YoungModulus', 'Density293K', 'CTEbelowTg', 'Tstrain', 'Tsoft']

# Split the dataset into train, test, and validation sets with 64:20:16 portions.
train_features, test_features, train_targets, test_targets = train_test_split(features, data[targets], test_size=0.2, random_state=42)
train_features, val_features, train_targets, val_targets = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Create a K-Nearest Neighbors (KNN) regression model.
knn_model = KNeighborsRegressor(n_neighbors=5)  

# Fit the model on the training set.
knn_model.fit(train_features, train_targets)

# Predict on the validation set.
val_predictions = knn_model.predict(val_features)

# Calculate evaluation metrics on the validation set.
val_mse = mean_squared_error(val_targets, val_predictions)
val_r2 = r2_score(val_targets, val_predictions)
val_mae = mean_absolute_error(val_targets, val_predictions)
val_medae = median_absolute_error(val_targets, val_predictions)
val_rd = np.mean(np.abs((val_targets - val_predictions) / val_targets)) * 100

# Predict on the test set.
test_predictions = knn_model.predict(test_features)

# Calculate evaluation metrics on the test set.
test_mse = mean_squared_error(test_targets, test_predictions)
test_r2 = r2_score(test_targets, test_predictions)
test_mae = mean_absolute_error(test_targets, test_predictions)
test_medae = median_absolute_error(test_targets, test_predictions)
test_rd = np.mean(np.abs((test_targets - test_predictions) / test_targets)) * 100

# Print the evaluation metrics.
print("Validation Set:")
print("Mean Squared Error:", val_mse)
print("R² Score:", val_r2)
print("Mean Absolute Error:", val_mae)
print("Median Absolute Error:", val_medae)
print("Relative Deviation (%):", val_rd)

print("\nTest Set:")
print("Mean Squared Error:", test_mse)
print("R² Score:", test_r2)
print("Mean Absolute Error:", test_mae)
print("Median Absolute Error:", test_medae)
print("Relative Deviation (%):", test_rd)

# Save the model as a pickle file.
with open('../model/joint_prediction/knn_model.pickle', 'wb') as file:
    pickle.dump(knn_model, file)
