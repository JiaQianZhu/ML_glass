"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to perform the generation test on the joint prediction model.
"""

import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import pandas as pd
import numpy as np

# Load the data_pred.
data = pd.read_csv('../(a-d) Datasets/US_patent_data.csv')

# Define input features and target variables.
features = data[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
targets = ['YoungModulus', 'Density293K', 'CTEbelowTg', 'Tstrain', 'Tsoft']

# Load the model.
with open('model/joint_prediction/rf_model.pickle', 'rb') as file:
    rf_model = pickle.load(file)

# Predict the target values.
predictions = rf_model.predict(features)

# Calculate the regression evaluation metrics.
mse = mean_squared_error(data[targets], predictions)
rmse = mean_squared_error(data[targets], predictions, squared=False)
mae = mean_absolute_error(data[targets], predictions)
medae = median_absolute_error(data[targets], predictions)
rd = medae / np.mean(data[targets])

# Calculate the percentage difference based on actual values.
percentage_diff = ((data[targets] - predictions) / data[targets]) * 100

# Output the evaluation metrics.
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Median Absolute Error (MedAE):", medae)
print("Relative Difference (RD):", rd)

# Calculate the residuals.
residuals = data[targets] - predictions

# Calculate the range of differences.
diff_range = np.abs(data[targets] - predictions).max(axis=0)

# Calculate the residuals of five properties.
residuals_YoungModulus = residuals['YoungModulus']
residuals_Density293K = residuals['Density293K']
residuals_CTE = residuals['CTEbelowTg']
residuals_Tstrain = residuals['Tstrain']
residuals_Tsoft = residuals['Tsoft']

# Output the residuals variation of five properties.

print("Residuals Variation for YoungModulus:")
print("Min:", residuals_YoungModulus.min())
print("Max:", residuals_YoungModulus.max())
print("Mean:", residuals_YoungModulus.mean())
print("Median:", residuals_YoungModulus.median())

print("\nPercentage Difference for YoungModulus:")
print("Min:", percentage_diff['YoungModulus'].min())
print("Max:", percentage_diff['YoungModulus'].max())
print("Mean:", percentage_diff['YoungModulus'].mean())
print("Median:", percentage_diff['YoungModulus'].median())

print("\nResiduals Variation for Density293K:")
print("Min:", residuals_Density293K.min())
print("Max:", residuals_Density293K.max())
print("Mean:", residuals_Density293K.mean())
print("Median:", residuals_Density293K.median())

print("\nPercentage Difference for Density293K:")
print("Min:", percentage_diff['Density293K'].min())
print("Max:", percentage_diff['Density293K'].max())
print("Mean:", percentage_diff['Density293K'].mean())
print("Median:", percentage_diff['Density293K'].median())



print("\nResiduals Variation for CTE:")
print("Min:", residuals_CTE.min())
print("Max:", residuals_CTE.max())
print("Mean:", residuals_CTE.mean())
print("Median:", residuals_CTE.median())


print("\nPercentage Difference for CTE:")
print("Min:", percentage_diff['CTEbelowTg'].min())
print("Max:", percentage_diff['CTEbelowTg'].max())
print("Mean:", percentage_diff['CTEbelowTg'].mean())
print("Median:", percentage_diff['CTEbelowTg'].median())


print("\nResiduals Variation for Tstrain:")
print("Min:", residuals_Tstrain.min())
print("Max:", residuals_Tstrain.max())
print("Mean:", residuals_Tstrain.mean())
print("Median:", residuals_Tstrain.median())

print("\nPercentage Difference for Tstrain:")
print("Min:", percentage_diff['Tstrain'].min())
print("Max:", percentage_diff['Tstrain'].max())
print("Mean:", percentage_diff['Tstrain'].mean())
print("Median:", percentage_diff['Tstrain'].median())

print("\nResiduals Variation for Tsoft:")
print("Min:", residuals_Tsoft.min())
print("Max:", residuals_Tsoft.max())
print("Mean:", residuals_Tsoft.mean())
print("Median:", residuals_Tsoft.median())

print("\nPercentage Difference for Tsoft:")
print("Min:", percentage_diff['Tsoft'].min())
print("Max:", percentage_diff['Tsoft'].max())
print("Mean:", percentage_diff['Tsoft'].mean())
print("Median:", percentage_diff['Tsoft'].median())
