"""
This is the plotting file for the larger binary classification data.
Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
"""
import numpy as np
import os
import matplotlib.pyplot as plt

## PLOTING
DATA_NAME = "larger_binary_class"


PATH = os.getcwd()
RESULT_PATH = PATH + '/NEUROCHAOS-LTS-RESULTS/'  + DATA_NAME + '/LTS/'




FSCORE_NEUROCHAOS = np.load(RESULT_PATH+"/fscore_neurochaos.npy" )    
STD_FSCORE_NEUROCHAOS = np.load(RESULT_PATH+"/std_fscore_neurochaos.npy") 
 
FSCORE_SVM = np.load(RESULT_PATH+"/fscore_svm.npy" )    
STD_FSCORE_SVM = np.load(RESULT_PATH+"/std_fscore_svm.npy")   
SAMPLES_PER_CLASS = np.load(RESULT_PATH+"/samples_per_class.npy")


# Plotting F1-score vs. Samples per class
plt.figure(figsize=(15,10))

plt.plot(SAMPLES_PER_CLASS ,FSCORE_NEUROCHAOS[:,0], linewidth = 2.0,  linestyle='solid', color ='r', marker='s',ms=10, label="Neurochaos-SVM")
plt.plot(SAMPLES_PER_CLASS ,FSCORE_SVM[:,0], linewidth = 2.0, linestyle='--', color ='k', marker='o', ms=10, label="SVM (linear)")

plt.xticks(SAMPLES_PER_CLASS,fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.xlabel('Number of training samples per class', fontsize=22)
plt.ylabel('Average F1-score', fontsize=22)
plt.legend(loc="upper right", fontsize=22)
plt.savefig(RESULT_PATH+ "/F1_low_training_sample_regime_larger_binary_classification.jpg", format='jpg', dpi=300)
plt.show()

# Plotting Standard deviation of F1-score vs. smaples per class
plt.figure(figsize=(15,10))
plt.plot(SAMPLES_PER_CLASS ,STD_FSCORE_NEUROCHAOS[:,0], linewidth = 2.0,  linestyle='solid', color ='r', marker='s',ms=10, label="Neurochaos-SVM")
plt.plot(SAMPLES_PER_CLASS ,STD_FSCORE_SVM[:,0], linewidth = 2.0, linestyle='--', color ='k', marker='o', ms=10, label="SVM (linear)")


plt.xticks(SAMPLES_PER_CLASS,fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.xlabel('Number of training samples per class', fontsize=22)
plt.ylabel('Standard deviation of F1-scores', fontsize=22)
plt.legend(loc="upper right", fontsize=22)
plt.savefig(RESULT_PATH+ "/standard_deviation_low_training_sample_regime_larger_binary_classification.jpg", format='jpg', dpi=300)
plt.show()
