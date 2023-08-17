"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to fill in the missing values of Young's Modulus
in the original dataset with trained knn model.
"""

import pickle
import pandas as pd


data = pd.read_csv('../data/raw/data.csv')

with open('../model/individual_prediction/Young_knn_model.pickle', 'rb') as file:
    knn_model = pickle.load(file)

# Filter out the missing parts of the "Young's Modulus" column.
missing_data = data[data['YoungModulus'].isnull()]

# Extract the missing features.
features_missing = missing_data[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]

# Make predictions using the model.
predictions = knn_model.predict(features_missing)

# Fill in the predicted values into the original dataset.
data.loc[data['YoungModulus'].isnull(), 'YoungModulus'] = predictions

# Save the dataset after filling in the values.
data.to_csv('../data/clean/Young_filled.csv', index=False)
