"""
This is the main file for low training sample regime for multiclass classification problem.
In order edit the function file - go to Codes.py and edit the function low_training_sample_five_class


Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
"""
from Codes import low_training_sample_five_class
CLASSIFICATION_TYPE = "multi_class"
TRIALS = 200
MAX_SAMPLES = 6
INITIAL_NEURAL_ACTIVITY = 0.34
DISCRIMINATION_THRESHOLD = 0.499
EPSILON = 0.1835


FSCORE_NEUROCHAOS, STANDARD_DEVIATION_FSCORE_NEUROCHAOS, FSCORE_SVM, STANDARD_DEVIATION_FSCORE_SVM = low_training_sample_five_class(CLASSIFICATION_TYPE, TRIALS, MAX_SAMPLES, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)

print("Average F1-score of Neurochaos = ", FSCORE_NEUROCHAOS)
print("Average F1-score of SVM = ", FSCORE_SVM)
