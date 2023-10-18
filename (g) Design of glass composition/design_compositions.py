"""
Author:JiaQian Zhu
Date:2023-8-17
Usage:This module offers method to provide potential composition ratios using the joint prediction model.
"""

import random
import pickle
import pandas as pd


def generate_oxide_ratio():
    # Generate oxide ratios within a specific percentage range, rounded to four decimal places.
    sio2 = round(random.uniform(0.65, 0.73), 4)
    al2o3 = round(random.uniform(0.10, 0.14), 4)
    cao = round(random.uniform(0.03, 0.09), 4)
    mgo = round(random.uniform(0.02, 0.06), 4)
    sro = round(random.uniform(0.00, 0.06), 4)
    b2o3 = round(random.uniform(0.03, 0.08), 4)
    sb2o3 = round(random.uniform(0, 0.01), 4)

    # Ensure that the molar sum is equal to 1.
    total_sum = sio2 + al2o3 + cao + mgo + sro + b2o3 + sb2o3
    sio2 /= total_sum
    al2o3 /= total_sum
    cao /= total_sum
    mgo /= total_sum
    sro /= total_sum
    b2o3 /= total_sum
    sb2o3 /= total_sum
    return [sio2, al2o3, cao, mgo, sro, b2o3, sb2o3]


# load the model
with open('model/joint_prediction/rf_model.pickle', 'rb') as file:
    rf_model = pickle.load(file)


# Define the target ranges of five properties.
young_modulus_min = 77.95
density_293k_max = 2.503
cte_below_tg_min = 3.3e-06
cte_below_tg_max = 3.9e-06
t_strain_min = 990.203
t_soft_min = 1269.761

# Create an empty dataset to store the data_pred that meets the criteria.
filtered_data = pd.DataFrame(columns=['SiO2', 'Al2O3', 'CaO', 'MgO', 'SrO', 'B2O3', 'Sb2O3',
                                      'YoungModulus', 'Density293K', 'CTEbelowTg', 'Tstrain', 'Tsoft'])

num_samples = 50000
# Design oxide ratios and predict target values.
for _ in range(num_samples):  
    # Design oxide proportions.
    oxide_ratio = generate_oxide_ratio()  

    # Predict target values.
    features = [oxide_ratio]
    targets = rf_model.predict(features)

    # Check if the target values are within the specified range.
    young_modulus = targets[0][0]
    density_293k = targets[0][1]
    cte_below_tg = targets[0][2]
    t_strain = targets[0][3]
    t_soft = targets[0][4]

    if young_modulus >= young_modulus_min and \
            density_293k <= density_293k_max and \
            cte_below_tg_min <= cte_below_tg <= cte_below_tg_max and \
            t_strain >= t_strain_min and \
            t_soft >= t_soft_min:
        # Save the data_pred that meets the criteria to a new dataset.
        data_row = oxide_ratio + [young_modulus, density_293k, cte_below_tg, t_strain, t_soft]
        filtered_data.loc[len(filtered_data)] = data_row

# Save the data_pred that meets the criteria to a new dataset.
filtered_data.to_csv('results/oxide_composition/oxide_ratios', index=False)
