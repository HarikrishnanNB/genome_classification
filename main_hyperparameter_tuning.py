"""
This is the executing file for hyperparameter tuning.

Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
Dtd: 31 - July - 2020
"""


import numpy as np
from Codes import  hyperparameter_tuning_bc

CLASSIFICATION_TYPE = "binary_class"
EPSILON = np.arange(0.18, 0.1901, 0.0001)
INITIAL_NEURAL_ACTIVITY = np.array([0.34], dtype='float64')
DISCRIMINATION_THRESHOLD = np.array([0.499], dtype='float64')

BEST_INA, BEST_DT, BEST_EPS = hyperparameter_tuning_bc(CLASSIFICATION_TYPE, EPSILON, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD)
# BEST_INA = BEST INITIAL NEURAL ACTIVITY
# BEST_DT = BEST DISCRIMINATION THRESHOLD
# BEST_EPS = BEST EPSILON
