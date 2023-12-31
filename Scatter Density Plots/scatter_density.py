"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to generate the scatter density plot of density.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from matplotlib.font_manager import FontProperties

data = pd.read_csv('../(a-d) Datasets/clean/dataset_completed.csv')

# Create a font with bold style
font = FontProperties(weight='bold')

features = data[['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3']]
target = 'Density293K'

# Split the dataset into train, test, and validation sets with 64:20:16 portions.
X_train_val, X_test, y_train_val, y_test = train_test_split(features, data[target], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Create the random forest regression model.
rf_model = RandomForestRegressor()

# Fit the model.
rf_model.fit(X_train, y_train)

# Predict on the holdout set.
y_test_pred = rf_model.predict(X_test)

# Calculate regression evaluation metrics.
r2 = r2_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
medae = median_absolute_error(y_test, y_test_pred)
rd = np.abs((y_test - y_test_pred) / y_test) * 100  # Calculate RD as a percentage.
rd_mean = rd.mean()


# Create the main plot.
fig, ax = plt.subplots(figsize=(8, 8))

scatter = ax.scatter(y_test, y_test_pred, c=y_test - y_test_pred, cmap='cool', s=50 * (np.abs(y_test - y_test_pred)))

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density', weight='bold', fontsize=16)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')

# Add the identity line.
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', ls='--')

# Set the axis labels.
ax.set_ylabel('Predicted Values of Density', fontsize=16, fontweight='bold')
ax.set_xlabel('Reported Values', fontsize=16, fontweight='bold')
bold_font = FontProperties(weight='extra bold')
ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontproperties=bold_font)
ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), fontproperties=bold_font)
ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontproperties=bold_font)
ax.tick_params(axis='x', width=2, length=6, which='major', direction='in', pad=10)
ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), fontproperties=bold_font)
ax.tick_params(axis='y', width=2, length=6, which='major', direction='in', pad=10)

# Set the aspect ratio as equal.
ax.set_aspect('equal')

# Create the first inset subplot on the right side
ax_inset1 = fig.add_axes([0.15, 0.65, 0.15, 0.17])
ax_inset1.hist(y_test - y_test_pred, bins=700, color='blue', alpha=0.8)
ax_inset1.axvline(0, color='black', ls='--')
ax_inset1.set_xlabel('Pred.residual', fontsize=13, fontweight='bold')
ax_inset1.yaxis.tick_right()

bold_font = FontProperties(weight='extra bold')
ax_inset1.xaxis.set_tick_params(labelsize=12)
ax_inset1.yaxis.set_tick_params(labelsize=12)

ax_inset1.yaxis.set_ticklabels(ax_inset1.yaxis.get_ticklabels(), fontproperties=bold_font)
ax_inset1.yaxis.set_label_position('right')
ax_inset1.set_ylabel('Frequency', rotation=90, labelpad=10, fontsize=13, fontweight='bold')

# Set the x-axis range to -0.1 to 0.1
ax_inset1.set_xlim(-0.1, 0.1)
ax_inset1.get_xticklabels()[0].set_weight('bold')
ax_inset1.get_xticklabels()[-1].set_weight('bold')
ax_inset1.get_xticklabels()[-2].set_weight('bold')

# Set the y-axis range to 0 to 200
ax_inset1.set_ylim(0, 200)
ax_inset1.get_yticklabels()[0].set_weight('bold')
ax_inset1.get_yticklabels()[-1].set_weight('bold')
ax_inset1.get_yticklabels()[-2].set_weight('bold')


bold_font = FontProperties(weight='bold')
ax_inset1.yaxis.set_ticklabels(ax_inset1.yaxis.get_ticklabels(), fontproperties=bold_font)
ax_inset1.tick_params(axis='y', width=2, length=6, which='major', direction='in', pad=10)
ax_inset1.tick_params(axis='x', width=2, length=6, which='major', direction='in', pad=10)
ax_inset1.set_box_aspect(1)

# Create the second inset subplot in the lower right corner.
ax_inset2 = fig.add_axes([0.44, 0.25, 0.25, 0.15], facecolor='blue', frame_on=True)
ax_inset2.axis('off')
ax_inset2.text(0, 0.8, f'MSE = {mse:.2e}', fontsize=18, fontweight='bold')
ax_inset2.text(0, 0.6, f'MAE = {mae:.2e}', fontsize=18, fontweight='bold')
ax_inset2.text(0, 0.4, f'MedAE = {medae:.2e}', fontsize=18, fontweight='bold')
ax_inset2.text(0, 0.2, f'RD = {rd_mean:.2f}%', fontsize=18, fontweight='bold')
ax_inset2.text(0, 0.0, f'R² = {r2:.4f}', fontsize=18, fontweight='bold')

# Show the plot.
plt.show()

