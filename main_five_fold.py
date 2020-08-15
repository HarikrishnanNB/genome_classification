"""
This is the executing file for the code of five fold validation for binary class, multiclass and
larger dataset. Go to five_fold_valifation.py in Codes.py to edit the function.

"""
from Codes import five_fold_validation

EPSILON = 0.183
INITIAL_NEURAL_ACTIVITY = 0.34
DISCRIMINATION_THRESHOLD = 0.499
CLASSIFICATION_TYPE = "larger_data_binary_class"
FOLDER_NAME = "five-fold-valid-"
if (CLASSIFICATION_TYPE == "binary_class" or CLASSIFICATION_TYPE == "larger_data_binary_class"):
    TARGET_NAMES = ['Class-0', 'Class-1']
elif CLASSIFICATION_TYPE == "multi_class":
    TARGET_NAMES = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']

MEAN_FOLD_ACCURACY_NEUROCHAOS, MEAN_FOLD_FSCORE_NEUROCHAOS, MEAN_FOLD_ACCURACY_SVM, MEAN_FOLD_FSCORE_SVM = five_fold_validation(CLASSIFICATION_TYPE, EPSILON, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, FOLDER_NAME, TARGET_NAMES)

print("Average Fold accuracy neurochaos = ", MEAN_FOLD_ACCURACY_NEUROCHAOS)
print("Average Fold f1score neurochaos = ", MEAN_FOLD_FSCORE_NEUROCHAOS)
print("Average Fold accuracy svm = ", MEAN_FOLD_ACCURACY_SVM)
print("Average Fold f1score svm = ", MEAN_FOLD_FSCORE_SVM)
