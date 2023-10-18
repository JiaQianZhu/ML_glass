import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score

data = pd.read_csv('../(a-d) Datasets/raw/data.csv')

# Filter out the non-missing data_pred points.
non_empty_df = data.dropna(subset=['Density293K'])

# Define input features and target variables.
features = non_empty_df[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
target = non_empty_df['Density293K']

# Split the dataset into training and test sets with 80:20 portions.
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and evaluate different regression models.
models = {
    'SVM': SVR(),
    'MLP': MLPRegressor(),
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'XGBoost': XGBRegressor(),
    'CART': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
    'RF': RandomForestRegressor(),
    'GPR': GaussianProcessRegressor(kernel=RBF())
}

model_scores = {}  # To store R^2 scores for each model.

for model_name, model in models.items():
    model.fit(train_features, train_target)

    # Make predictions on the validation set.
    test_predictions = model.predict(test_features)

    # Calculate R^2 score on the validation set.
    test_r2 = r2_score(test_target, test_predictions)

    model_scores[model_name] = test_r2

# Sort the models by R^2 scores in descending order.
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
print('Model evaluation on original density dataset:')
for i, (model_name, r2) in enumerate(sorted_models):
    print(f"{i + 1}. Model: {model_name}, R^2 Score: {r2:.4f}")
