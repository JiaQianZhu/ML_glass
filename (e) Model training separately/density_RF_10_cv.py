"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to filter out the non-missing portions of density and conduct training using the RF algorithm with a 10-fold cv.
"""
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('../(a-d) Datasets/raw/data.csv')

# Filter out the non-missing data points
non_empty_df = data.dropna(subset=['Density293K'])

# Define input features and target variable
features = non_empty_df[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
target = non_empty_df['Density293K']

# Split the dataset into training and test sets with 80:20 proportions
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2,
                                                                            random_state=42)

# Create a Random Forest regression model
rf_model = RandomForestRegressor()

# Initialize K-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
cv_mse_scores = []
cv_mae_scores = []
cv_medae_scores = []
cv_rd_scores = []
cv_r2_scores = []

# Perform 10-fold cross-validation
for train_idx, val_idx in kf.split(train_features):
    train_fold_features, val_fold_features = train_features.iloc[train_idx], train_features.iloc[val_idx]
    train_fold_target, val_fold_target = train_target.iloc[train_idx], train_target.iloc[val_idx]

    # Create and train the model
    rf_model.fit(train_fold_features, train_fold_target)

    # Make predictions
    val_predictions = rf_model.predict(val_fold_features)

    # Calculate evaluation metrics
    mse = mean_squared_error(val_fold_target, val_predictions)
    mae = mean_absolute_error(val_fold_target, val_predictions)
    medae = median_absolute_error(val_fold_target, val_predictions)
    rd = np.mean(np.abs((val_fold_target - val_predictions) / val_fold_target)) * 100
    r2 = r2_score(val_fold_target, val_predictions)

    # Append metrics to respective lists
    cv_mse_scores.append(mse)
    cv_mae_scores.append(mae)
    cv_medae_scores.append(medae)
    cv_rd_scores.append(rd)
    cv_r2_scores.append(r2)

# Calculate mean and standard deviation of the metrics
cv_mse_mean = np.mean(cv_mse_scores)
cv_mse_std = np.std(cv_mse_scores)
cv_mae_mean = np.mean(cv_mae_scores)
cv_mae_std = np.std(cv_mae_scores)
cv_medae_mean = np.mean(cv_medae_scores)
cv_medae_std = np.std(cv_medae_scores)
cv_rd_mean = np.mean(cv_rd_scores)
cv_rd_std = np.std(cv_rd_scores)
cv_r2_mean = np.mean(cv_r2_scores)
cv_r2_std = np.std(cv_r2_scores)

# Output metrics
print("Cross-Validation Mean Squared Error:", cv_mse_mean)
print("Cross-Validation MSE Standard Deviation:", cv_mse_std)
print("Cross-Validation Mean Absolute Error:", cv_mae_mean)
print("Cross-Validation MAE Standard Deviation:", cv_mae_std)
print("Cross-Validation Median Absolute Error:", cv_medae_mean)
print("Cross-Validation MedAE Standard Deviation:", cv_medae_std)
print("Cross-Validation Relative Deviation (%):", cv_rd_mean)
print("Cross-Validation RD Standard Deviation:", cv_rd_std)
print("Cross-Validation R^2 Score:", cv_r2_mean)
print("Cross-Validation R2 Standard Deviation:", cv_r2_std)

# Fit the model on the entire training set
rf_model.fit(train_features, train_target)

# Make predictions on the test set
test_predictions = rf_model.predict(test_features)

# Calculate evaluation metrics on the test set
test_mse = mean_squared_error(test_target, test_predictions)
test_mae = mean_absolute_error(test_target, test_predictions)
test_medae = median_absolute_error(test_target, test_predictions)
test_rd = np.mean(np.abs((test_target - test_predictions) / test_target)) * 100
test_r2 = r2_score(test_target, test_predictions)

# Output test set metrics
print("\nTest Set:")
print("Mean Squared Error:", test_mse)
print("Mean Absolute Error:", test_mae)
print("Median Absolute Error:", test_medae)
print("Relative Deviation (%):", test_rd)
print("R^2 Score:", test_r2)

# Save the model as a pickle file
with open('rf_model.pickle', 'wb') as file:
    pickle.dump(rf_model, file)
