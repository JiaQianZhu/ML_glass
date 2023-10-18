"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to fill in the missing values of density in the original dataset with trained rf model.
"""


import pickle
import pandas as pd


data = pd.read_csv('../(a-d) Datasets/raw/data.csv')


with open('../model/individual_prediction/density_rf_model.pickle', 'rb') as file:
    rf_model = pickle.load(file)

# Filter out the missing parts of the "Density293K" column.
missing_data = data[data['Density293K'].isnull()]

# Extract the missing features.
features_missing = missing_data[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]

# Make predictions using the model.
predictions = rf_model.predict(features_missing)

# Fill in the predicted values into the original dataset.
data.loc[data['Density293K'].isnull(), 'Density293K'] = predictions

# Save the dataset after filling in the values.
data.to_csv('../data_pred/clean/density_filled.csv', index=False)
