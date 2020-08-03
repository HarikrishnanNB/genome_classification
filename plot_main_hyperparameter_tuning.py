# -*- coding: utf-8 -*-
"""
This is the plot file for the hyperparameter tuning (main_hyperparamter_tuning.py).

Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
Dtd: 31 - July - 2020
"""
import os
import numpy as np
import matplotlib.pyplot as plt

CLASSIFICATION_TYPE = "binary_class"
PATH = os.getcwd()
RESULT_PATH = PATH + '/NEUROCHAOS-RESULTS/'  + CLASSIFICATION_TYPE + '/CROSS_VALIDATION/'
print("Loading Hyperparameter Tuning Results")
F1SCORE_MATRIX = np.load(RESULT_PATH + 'H_FSCORE.npy')
EPSILON_MATRIX = np.load(RESULT_PATH + 'H_EPS.npy')

plt.figure(figsize=(15, 10))
plt.plot(EPSILON_MATRIX[0, :], F1SCORE_MATRIX[0, :], '-*k', markersize=12)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(True)
plt.xlabel(r'$\epsilon$', fontsize=25)
plt.ylabel('Average F1-score', fontsize=25)
plt.savefig(RESULT_PATH+"/binary-class-hyperparameter-tuning.jpg", format='jpg', dpi=200)
plt.show()
