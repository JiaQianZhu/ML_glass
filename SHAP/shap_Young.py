"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to generate the shap plots of Young's Modulus.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.font_manager as fm

font_prop = fm.FontProperties(weight='bold', size=16)

data = pd.read_csv('../data/clean/dataset_completed.csv')
features = data[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
targets = ['YoungModulus', 'Density293K', 'CTEbelowTg', 'Tstrain', 'Tsoft']

# Split the dataset into train, test, and validation sets with 64:20:16 portions.
X_train_val, X_test, y_train_val, y_test = train_test_split(features, data[targets], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Create the random forest regression model.
rf_model = RandomForestRegressor()

# Fit the model.
rf_model.fit(X_train, y_train)

# Predict on the test set.
y_test_pred_rf = rf_model.predict(X_test)

# Initialize the SHAP explainer with the trained random forest model.
explainer = shap.Explainer(rf_model)

# Calculate SHAP values for the validation set.
shap_values = explainer.shap_values(X_val)

column_index_YoungModulus = list(y_val.columns).index('YoungModulus')
shap_values_YoungModulus = shap_values[column_index_YoungModulus]


# Plot the SHAP summary plot for 'YoungModulus'
shap.summary_plot(shap_values_YoungModulus, X_val, feature_names=X_val.columns, show=False)
plt.title("SHAP on Young's Modulus", fontweight='bold', fontsize=18)
plt.subplots_adjust(top=0.9)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.savefig("../results/shap/Young's Modulus.jpg", dpi=300)



