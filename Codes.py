"""
This module does hyperparameter tuning for the Neurochaos-SVM method.
This module finds the best epsilon and discrimnation threshold for a given
intial neural activicty. The hyperparameters are saved as the program completes
its execution.

Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
Dtd: 31 - July - 2020
"""



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX
from load_data import get_data


def hyperparameter_tuning_bc(classification_type, epsilon, initial_neural_activity, discrimination_threshold):
    """
    Parameters
    ----------
    classification_type : string
        DESCRIPTION- classification_type = "binary_class", loads binary
        classification data.
        classification_type = "multi_class", loads multiclass classification
        data.
    epsilon : array, 1D
        DESCRIPTION - epsilon - is the neighbourhood of the stimulus. Epsilon is
        a value between 0 and 0.3.

    initial_neural_activity : array, 1D (This array should contain
    only one element, for eg. np.array([0.34],dtype = 'float64')).

        DESCRIPTION - Every chaotic neuron has an initial neural activity.
        The firinig of chaotic neuron starts from this value.

    discrimination_threshold : array, 1D
        DESCRIPTION - discrimination threshold is used to calculate the
        fraction of time the chaotic trajrectory is above this threshold.
        For more informtion, refer the following: https://aip.scitation.org/doi/abs/10.1063/1.5120831?journalCode=cha

    Returns
    -------
    best_initial_neural_activity : array, 1D (This array has only one element).
        DESCRIPTION - return initial neural activity
    best_discrimination_threshold : array, 1D
        DESCRIPTION - return discrimniation threshold
    best_epsilon : array, 1D
        DESCRIPTION - return best epsilon

    """
    full_genome_data, full_genome_label = get_data(classification_type)
    accuracy_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    f1score_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    q_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    b_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    epsilon_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))

    # Define the split - into 2 folds
    k_fold = KFold(n_splits=3, random_state=42, shuffle=True)

    # returns the number of splitting iterations in the cross-validator
    k_fold.get_n_splits(full_genome_data)
    print(k_fold)

    KFold(n_splits=3, random_state=42, shuffle=True)
    row = -1
    col = -1

    initial_condition_instance = initial_neural_activity[0]
    for threshold_instance in discrimination_threshold:
        row = row+1
        col = -1

        for epsilon_instance in epsilon:
            col = col+1
            acc_temp = []
            fscore_temp = []

            for train_index, val_index in k_fold.split(full_genome_data):

                train_genome_data = full_genome_data[train_index]
                val_genome_data = full_genome_data[val_index]
                train_genome_label = full_genome_label[train_index]
                val_genome_label = full_genome_label[val_index]
                print(" train data (%) = ",
                      (train_genome_data.shape[0]/full_genome_data.shape[0])*100)
                print("val data (%) = ",
                      (val_genome_data.shape[0]/full_genome_data.shape[0])*100)


                neurochaos_train_data_features = CFX.transform(train_genome_data,
                                                               initial_condition_instance,
                                                               20000, epsilon_instance,
                                                               threshold_instance)
                neurochaos_val_data_features = CFX.transform(val_genome_data,
                                                             initial_condition_instance,
                                                             20000, epsilon_instance,
                                                             threshold_instance)

                # Neurochaos-SVM with linear kernel.

                classifier_neurochaos_svm = LinearSVC(random_state=0, tol=1e-5, dual=False)
                classifier_neurochaos_svm.fit(neurochaos_train_data_features,
                                              train_genome_label[:, 0])
                predicted_neurochaos_val_label = classifier_neurochaos_svm.predict(neurochaos_val_data_features)
                # Accuracy
                acc_neurochaos = accuracy_score(val_genome_label, predicted_neurochaos_val_label)*100
                # Macro F1- Score
                f1score_neurochaos = f1_score(val_genome_label, predicted_neurochaos_val_label, average="macro")

                acc_temp.append(acc_neurochaos)
                fscore_temp.append(f1score_neurochaos)

            q_matrix[row, col] = initial_condition_instance
            b_matrix[row, col] = threshold_instance
            epsilon_matrix[row, col] = epsilon_instance
            # Average Accuracy
            accuracy_matrix[row, col] = np.mean(acc_temp)
            # Average Macro F1-score
            f1score_matrix[row, col] = np.mean(fscore_temp)

            print("q_matrix = ", q_matrix[row, col],
                  "b_matrix = ", b_matrix[row, col],
                  "epsilon = ", epsilon_matrix[row, col])
            print("Three fold Average F-SCORE %.3f" %f1score_matrix[row, col])

            print('--------------------------')
    # Creating a result path to save the results.
    path = os.getcwd()
    result_path = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/CROSS_VALIDATION/'


    try:
        os.makedirs(result_path)
    except OSError:
        print("Creation of the result directory %s failed" % result_path)
    else:
        print("Successfully created the result directory %s" % result_path)

    print("Saving Hyperparameter Tuning Results")
    np.save(result_path + 'H_ACCURACY.npy', accuracy_matrix)
    np.save(result_path + 'H_FSCORE.npy', f1score_matrix)
    np.save(result_path + 'H_INITIAL_CONDITION.npy', q_matrix)
    np.save(result_path + 'H_THRESHOLD.npy', b_matrix)
    np.save(result_path + 'H_EPS.npy', epsilon_matrix)
    # =============================================================================
    # best hyperparameters
    # =============================================================================
    # Computing the maximum F1-score obtained during crossvalidation.
    maximum_fscore = np.max(f1score_matrix)

    best_initial_neural_activity = []
    best_discrimination_threshold = []
    best_epsilon = []
    for row in range(0, f1score_matrix.shape[0]):

        for col in range(0, f1score_matrix.shape[1]):

            if f1score_matrix[row, col] == np.max(f1score_matrix):

                best_initial_neural_activity.append(q_matrix[row, col])
                best_discrimination_threshold.append(b_matrix[row, col])
                best_epsilon.append(epsilon_matrix[row, col])

    print("maximum f1score_neurochaos = ", maximum_fscore)
    print("best initial neural activity = ", best_initial_neural_activity)
    print("best discrimination threshold = ", best_discrimination_threshold)
    print("best epsilon = ", best_epsilon)

    return best_initial_neural_activity, best_discrimination_threshold, best_epsilon

def classification_report_csv_(report, num_classes):
    """
    Parameters
    ----------
    report : classification metric report
        DESCRIPTION - Contains the precision recall f1score
    num_classes : int
        DESCRIPTION - 2 or 5, if 2 the report for binary class is returned
        if 5 the report for 5 class classification is returned.
    Returns
    -------
    dataframe - contains precision recall f1score

    """
    if num_classes == 2:
        report_data = []
        lines = report.split('\n')
        report_data.append(lines[0])
        report_data.append(lines[2])
        report_data.append(lines[3])
        report_data.append(lines[5])
        report_data.append(lines[6])
        report_data.append(lines[7])
        dataframe = pd.DataFrame.from_dict(report_data)
    #    dataframe.to_csv('report.csv', index = False)
        return dataframe
    elif num_classes == 5:
        report_data = []
        lines = report.split('\n')
        report_data.append(lines[0])
        report_data.append(lines[2])
        report_data.append(lines[3])
        report_data.append(lines[4])
        report_data.append(lines[5])
        report_data.append(lines[6])
        report_data.append(lines[8])
        report_data.append(lines[9])
        report_data.append(lines[10])
        dataframe = pd.DataFrame.from_dict(report_data)
    #    dataframe.to_csv('report.csv', index = False)
        return dataframe


def five_fold_validation(classification_type, epsilon, initial_neural_activity, discrimination_threshold, folder_name, target_names):
    """
    This module does the five_fold_crossvalidation and saves the classifcation
    report. At present the results for binary classification and five class classification is saved.
    Author: Harikrishnan N B
    Email: harikrishnannb07@gmail.com
    Dtd: 2 - August - 2020

    Parameters
    ----------
    classification_type : string
        DESCRIPTION - classification_type == "binary_class" loads binary classification genome data.
        classification_type == "multi_class" loads multiclass genome data
    epsilon : scalar, float
        DESCRIPTION - A value in the range 0 and 0.3. for eg. epsilon = 0.1835
    initial_neural_activity : scalar, float
        DESCRIPTION - The chaotic neurons has an initial neural activity.
        Initial neural activity is a value in the range 0 and 1.
    discrimination_threshold : scalar, float
        DESCRIPTION - The chaotic neurons has a discrimination threhold.
        discrimination threshold is a value in the range 0 and 1.
    folder_name : string
        DESCRIPTION - the name of the folder to store results. For eg., if
        folder_name = "hnb", then this function will create two folder "hnb-svm"
        and "hnb-neurochaos" to save the classification report.
    target_names : array, 1D, string
        DESCRIPTION - if there are two classes, then target_names = ['class-0', class-1]
        Note- At the present version of the code, the results for binary classification
        and five class classification will be saved.
    Returns
    -------
    mean_fold_accuracy_neurochaos, mean_fold_fscore_neurochaos, mean_fold_accuracy_svm, mean_fold_fscore_svm

    The above are the average accuracy and f1 score for the five fold validation for neurochaos and svm respectively.
    """

    path = os.getcwd()
    result_path_svm = path + '/NEUROCHAOS-RESULTS/'  + classification_type +'/' + folder_name +'-svm/'
    result_path_neurochaos = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/' + folder_name +'-neurochaos/'

    # Creating Folder to save the results
    try:
        os.makedirs(result_path_neurochaos)
    except OSError:
        print("Creation of the result directory %s failed" % result_path_neurochaos)
    else:
        print("Successfully created the result directory %s" % result_path_neurochaos)

    try:
        os.makedirs(result_path_svm)
    except OSError:
        print("Creation of the result directory %s failed" % result_path_svm)
    else:
        print("Successfully created the result directory %s" % result_path_svm)

    full_genome_data, full_genome_label = get_data(classification_type)

    num_classes = len(np.unique(full_genome_label)) # Number of classes
    print("**** Genome data details ******")
    for class_label in range(np.max(full_genome_label)+1):
        print("Total Data instance in Class -", class_label, " = ", full_genome_label.tolist().count([class_label]))

    # Stratified five fold cross validation.
    stratified_k_fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True) # Define the split - into 5 folds
    stratified_k_fold.get_n_splits(full_genome_data, full_genome_label) # returns the number of splitting iterations in the cross-validator

    print(stratified_k_fold)
    StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    iterations = 0

    acc_temp_neurochaos = []
    fscore_temp_neurochaos = []

    acc_temp_svm = []
    fscore_temp_svm = []

    for train_index, val_index in stratified_k_fold.split(full_genome_data, full_genome_label):

        iterations = iterations+1

        print("iterations = ", iterations)

        # Spliting into training and validation
        train_genome_data, val_genome_data = full_genome_data[train_index], full_genome_data[val_index]
        train_genome_label, val_genome_label = full_genome_label[train_index], full_genome_label[val_index]


        print(" train data (%) = ", (train_genome_data.shape[0]/full_genome_data.shape[0])*100)
        print("val data (%) = ", (val_genome_data.shape[0]/full_genome_data.shape[0])*100)

        print("initial neural activity = ", initial_neural_activity, "discrimination threshold = ", discrimination_threshold, "epsilon = ", epsilon)

        # Extracting Neurochaos features from the data
        neurochaos_train_data_features = CFX.transform(train_genome_data, initial_neural_activity, 20000, epsilon, discrimination_threshold)
        neurochaos_val_data_features = CFX.transform(val_genome_data, initial_neural_activity, 20000, epsilon, discrimination_threshold)

        # Start of Neurochaos classifier
        neurochaos_classifier = LinearSVC(random_state=0, tol=1e-5, dual=False)

        neurochaos_classifier.fit(neurochaos_train_data_features, train_genome_label[:, 0])
        predicted_neurochaos_val_label = neurochaos_classifier.predict(neurochaos_val_data_features)

        acc_neurochaos = accuracy_score(val_genome_label, predicted_neurochaos_val_label)*100
        f1score_neurochaos = f1_score(val_genome_label, predicted_neurochaos_val_label, average="macro")
        report_neurochaos = classification_report(val_genome_label, predicted_neurochaos_val_label, target_names=target_names)

        # Saving the classification report to csv file for neurochaos classifier.
        print(report_neurochaos)
        if num_classes == 2:
            classification_report_csv_(report_neurochaos, num_classes).to_csv(result_path_neurochaos+'neurochaos_report_'+ str(iterations) +'.csv', index=False)
        elif num_classes == 5:
            classification_report_csv_(report_neurochaos, num_classes).to_csv(result_path_neurochaos+'neurochaos_report_'+ str(iterations) +'.csv', index=False)
        else:
            print("could not save classfication results-the current code saves the result of 2 class and 5 class problems")

        confusion_matrix_neurochaos = cm(val_genome_label, predicted_neurochaos_val_label)
        print("Confusion matrixfor Neurochaos\n", confusion_matrix_neurochaos)

        acc_temp_neurochaos.append(acc_neurochaos)
        fscore_temp_neurochaos.append(f1score_neurochaos)

        # End of Neurochaos classifier.

        # Start of SVM classifier
        svm_classifier = LinearSVC(random_state=0, tol=1e-5, dual=False)

        svm_classifier.fit(train_genome_data, train_genome_label[:, 0])
        predicted_svm_val_label = svm_classifier.predict(val_genome_data)

        acc_svm = accuracy_score(val_genome_label, predicted_svm_val_label)*100
        f1score_svm = f1_score(val_genome_label, predicted_svm_val_label, average="macro")
        report_svm = classification_report(val_genome_label, predicted_svm_val_label, target_names=target_names)

        # Saving the classification report to csv file for svm classifier
        print(report_svm)
        if num_classes == 2:
            classification_report_csv_(report_svm, num_classes).to_csv(result_path_svm+'report_svm_'+ str(iterations) +'.csv', index=False)
        elif num_classes == 5:
            classification_report_csv_(report_svm, num_classes).to_csv(result_path_svm+'report_svm_'+ str(iterations) +'.csv', index=False)
        else:
            print("could not save classfication results-the current code saves the result of 2 class and 5 class problems")


        confusion_matrix_svm = cm(val_genome_label, predicted_svm_val_label)
        print("Confusion matrix for SVM\n", confusion_matrix_svm)

        acc_temp_svm.append(acc_svm)
        fscore_temp_svm.append(f1score_svm)

        # End of SVM classifier

    mean_fold_accuracy_svm = np.mean(acc_temp_svm)
    mean_fold_fscore_svm = np.mean(fscore_temp_svm)

    mean_fold_accuracy_neurochaos = np.mean(acc_temp_neurochaos)
    mean_fold_fscore_neurochaos = np.mean(fscore_temp_neurochaos)


    np.save(result_path_neurochaos + 'initial_neural_activity.npy', initial_neural_activity)
    np.save(result_path_neurochaos + 'discrimination_threshold.npy', discrimination_threshold)
    np.save(result_path_neurochaos + 'EPS.npy', epsilon)
    return mean_fold_accuracy_neurochaos, mean_fold_fscore_neurochaos, mean_fold_accuracy_svm, mean_fold_fscore_svm




def low_training_sample_binary_class(classification_type, trials, max_samples, initial_neural_activity, discrimination_threshold, epsilon):
    """

    This is the function module for the low training sample regime for binary
    classification dataset

    Parameters
    ----------
    classification_type : string
        DESCRIPTION - This code is only for binary classification,
        classification_type= "larger_binary_class"
    trials : int
        DESCRIPTION - The number of random trials of training. For eg. if trials = 20, we do
        20 random trails of training and find the macro averged f1-score of the test data for the 20 ransod trials.
    max_sample : int
        DESCRIPTION - The upper limit on the number of samples per class for training. For eg., if
        max_samples = 6, means the maximum number of samples per class used for training is 6.
    initial_neural_activity : scalar, float
        DESCRIPTION - The chaotic neurons has an initial neural activity.
        Initial neural activity is a value in the range 0 and 1.
    discrimination_threshold : scalar, float
        DESCRIPTION - The chaotic neurons has a discrimination threhold.
        discrimination threshold is a value in the range 0 and 1.
    epsilon : scalar, float
        DESCRIPTION - A value in the range 0 and 0.3. for eg. epsilon = 0.1835

    Returns
    -------
    fscore_neurochaos : array, 2D, float
        DESCRIPTION. - The macro averged f1-score for training with 1, 2.., max_samples per class.
        This will provide averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.
    standard_deviation_fscore_neurochaos : array, 2D, flaot
        DESCRIPTION - This will provide the standard deviation of averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.
    fscore_svm : array, 2D, float
        DESCRIPTION. - The macro averged f1-score for training with 1, 2.., max_samples per class.
        This will provide averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.
    standard_deviation_fscore_svm : array, 2D, flaot
        DESCRIPTION - This will provide the standard deviation of averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.

    """
    full_genome_data, full_genome_label = get_data(classification_type)
    # Indices of class-0 and class-1 data instances
    index_0 = np.where(full_genome_label == 0)[0]
    index_1 = np.where(full_genome_label == 1)[0]

    print("**** TOTAL_DATA ******")
    for class_label in range(np.max(full_genome_label)+1):
        print("Data instance in Class -", class_label, " = ", full_genome_label.tolist().count([class_label]))

    # Neurochaos Feature Extraction
    neurochaos_genome_data_0 = CFX.transform(full_genome_data[index_0, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)
    neurochaos_genome_data_1 = CFX.transform(full_genome_data[index_1, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)

    # Input data (without Neurochaos Feature Extraction)
    genome_data_0 = full_genome_data[index_0, :]
    genome_data_1 = full_genome_data[index_1, :]


    # Number of samples per class. For eg. if max_samples is 4, we compute
    # training with 1, 2 3 and 4 samples. Each of these training is done
    # with N random trials.
    samples_per_class = np.arange(1, max_samples + 1, 1)

    # Initialization for Neurochaos
    accuracy_neurochaos = np.zeros((len(samples_per_class), 1))
    fscore_neurochaos = np.zeros((len(samples_per_class), 1))
    standard_deviation_fscore_neurochaos = np.zeros((len(samples_per_class), 1))

    # Initialization for SVM
    accuracy_svm = np.zeros((len(samples_per_class), 1))
    fscore_svm = np.zeros((len(samples_per_class), 1))
    standard_deviation_fscore_svm = np.zeros((len(samples_per_class), 1))

    for num_instance in samples_per_class:
        # Neurochaos array for appending accuracy and f1score
        accuracy_neurochaos_list = []
        f1score_neurochaos_list = []

        # SVM array for appending accuracy and f1score
        accuracy_svm_list = []
        f1score_svm_list = []
        # We do N random trials of training
        for trial_number in range(0, trials):

            train_neurochaos_genome_data_0, test_neurochaos_genome_data_0, train_neurochaos_genome_label_0, test_neurochaos_genome_label_0 = train_test_split(neurochaos_genome_data_0, full_genome_label[index_0], test_size=1 - (num_instance/neurochaos_genome_data_0.shape[0]), random_state=trial_number)
            train_neurochaos_genome_data_1, test_neurochaos_genome_data_1, train_neurochaos_genome_label_1, test_neurochaos_genome_label_1 = train_test_split(neurochaos_genome_data_1, full_genome_label[index_1], test_size=1 - (num_instance/neurochaos_genome_data_1.shape[0]), random_state=trial_number)


            train_genome_data_0, test_genome_data_0, train_genome_label_0, test_genome_label_0 = train_test_split(genome_data_0, full_genome_label[index_0], test_size=1 - (num_instance/genome_data_0.shape[0]), random_state=trial_number)
            train_genome_data_1, test_genome_data_1, train_genome_label_1, test_genome_label_1 = train_test_split(genome_data_1, full_genome_label[index_1], test_size=1 - (num_instance/genome_data_1.shape[0]), random_state=trial_number)

            print("num of samples per class = ", num_instance, "trial number = ", trial_number)

            test_neurochaos_genome_data = np.vstack((test_neurochaos_genome_data_0, test_neurochaos_genome_data_1))
            test_neurochaos_genome_label = np.vstack((test_neurochaos_genome_label_0, test_neurochaos_genome_label_1))


            train_neurochaos_genome_data = np.vstack((train_neurochaos_genome_data_0, train_neurochaos_genome_data_1))
            train_neurochaos_genome_label = np.vstack((train_neurochaos_genome_label_0, train_neurochaos_genome_label_1))


            test_genome_data = np.vstack((test_genome_data_0, test_genome_data_1))
            test_genome_label = np.vstack((test_genome_label_0, test_genome_label_1))

            train_genome_data = np.vstack((train_genome_data_0, train_genome_data_1))
            train_genome_label = np.vstack((train_genome_label_0, train_genome_label_1))

            # Neurochaos feature extraction followed by SVM classifier
            classifier_neurochaos = LinearSVC(random_state=0, tol=1e-5, dual=False)
            classifier_neurochaos.fit(train_neurochaos_genome_data, train_neurochaos_genome_label[:, 0])
            predicted_labels_neurochaos = classifier_neurochaos.predict(test_neurochaos_genome_data)

            accuracy_neurochaos_val = accuracy_score(test_neurochaos_genome_label, predicted_labels_neurochaos)*100

            f1score_neurochaos_val = f1_score(test_neurochaos_genome_label, predicted_labels_neurochaos, average="macro")

            accuracy_neurochaos_list.append(accuracy_neurochaos_val)
            f1score_neurochaos_list.append(f1score_neurochaos_val)


            classifier_svm = LinearSVC(random_state=0, tol=1e-5, dual=False)
            classifier_svm.fit(train_genome_data, train_genome_label[:, 0])
            predicted_labels_svm = classifier_svm.predict(test_genome_data)

            accuracy_svm_val = accuracy_score(test_genome_label, predicted_labels_svm)*100

            f1score_svm_val = f1_score(test_genome_label, predicted_labels_svm, average="macro")

            accuracy_svm_list.append(accuracy_svm_val)
            f1score_svm_list.append(f1score_svm_val)


        accuracy_neurochaos[num_instance-1, 0] = np.mean(accuracy_neurochaos_list)
        fscore_neurochaos[num_instance-1, 0] = np.mean(f1score_neurochaos_list)
        standard_deviation_fscore_neurochaos[num_instance-1, 0] = np.std(f1score_neurochaos_list)


        accuracy_svm[num_instance-1, 0] = np.mean(accuracy_svm_list)
        fscore_svm[num_instance-1, 0] = np.mean(f1score_svm_list)
        standard_deviation_fscore_svm[num_instance-1, 0] = np.std(f1score_svm_list)

    print("Saving Results")

    path = os.getcwd()
    result_path = path + '/NEUROCHAOS-LTS-RESULTS/'  + classification_type + '/LTS/'


    try:
        os.makedirs(result_path)
    except OSError:
        print("Creation of the result directory %s failed" % result_path)
    else:
        print("Successfully created the result directory %s" % result_path)

    np.save(result_path+"/fscore_neurochaos.npy", fscore_neurochaos)
    np.save(result_path+"/std_fscore_neurochaos.npy", standard_deviation_fscore_neurochaos)
    np.save(result_path+"/samples_per_class.npy", samples_per_class)

    np.save(result_path+"/fscore_svm.npy", fscore_svm)
    np.save(result_path+"/std_fscore_svm.npy", standard_deviation_fscore_svm)
    return fscore_neurochaos, standard_deviation_fscore_neurochaos, fscore_svm, standard_deviation_fscore_svm

def low_training_sample_five_class(classification_type, trials, max_samples, initial_neural_activity, discrimination_threshold, epsilon):
    """
    This is the function module for the low training sample regime for five class
    classification dataset. To add more classes or reduce the number of classes,
    the user has to modify this function file.

    Parameters
    ----------
    classification_type : string
        DESCRIPTION - This code is only for five class classification,
        classification_type= "multi_class"
    trials : int
        DESCRIPTION - The number of random trials of training. For eg. if trials = 20, we do
        20 random trails of training and find the macro averged f1-score of the test data for the 20 ransod trials.
    max_sample : int
        DESCRIPTION - The upper limit on the number of samples per class for training. For eg., if
        max_samples = 6, means the maximum number of samples per class used for training is 6.
    initial_neural_activity : scalar, float
        DESCRIPTION - The chaotic neurons has an initial neural activity.
        Initial neural activity is a value in the range 0 and 1.
    discrimination_threshold : scalar, float
        DESCRIPTION - The chaotic neurons has a discrimination threhold.
        discrimination threshold is a value in the range 0 and 1.
    epsilon : scalar, float
        DESCRIPTION - A value in the range 0 and 0.3. for eg. epsilon = 0.1835

    Returns
    -------
    fscore_neurochaos : array, 2D, float
        DESCRIPTION. - The macro averged f1-score for training with 1, 2.., max_samples per class.
        This will provide averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.
    standard_deviation_fscore_neurochaos : array, 2D, flaot
        DESCRIPTION - This will provide the standard deviation of averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.
    fscore_svm : array, 2D, float
        DESCRIPTION. - The macro averged f1-score for training with 1, 2.., max_samples per class.
        This will provide averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.
    standard_deviation_fscore_svm : array, 2D, flaot
        DESCRIPTION - This will provide the standard deviation of averaged f1-score for N random trials of training with 1, 2,..., max_samples
        per class.

    """
    full_genome_data, full_genome_label = get_data(classification_type)
    # Indices of class-0 and class-1 data instances
    index_0 = np.where(full_genome_label == 0)[0]
    index_1 = np.where(full_genome_label == 1)[0]
    index_2 = np.where(full_genome_label == 2)[0]
    index_3 = np.where(full_genome_label == 3)[0]
    index_4 = np.where(full_genome_label == 4)[0]

    print("**** TOTAL_DATA ******")
    for class_label in range(np.max(full_genome_label)+1):
        print("Data instance in Class -", class_label, " = ", full_genome_label.tolist().count([class_label]))

    # Neurochaos Feature Extraction
    neurochaos_genome_data_0 = CFX.transform(full_genome_data[index_0, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)
    neurochaos_genome_data_1 = CFX.transform(full_genome_data[index_1, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)
    neurochaos_genome_data_2 = CFX.transform(full_genome_data[index_2, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)
    neurochaos_genome_data_3 = CFX.transform(full_genome_data[index_3, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)
    neurochaos_genome_data_4 = CFX.transform(full_genome_data[index_4, :], initial_neural_activity, 20000, epsilon, discrimination_threshold)


    # Input data (without Neurochaos Feature Extraction)
    genome_data_0 = full_genome_data[index_0, :]
    genome_data_1 = full_genome_data[index_1, :]
    genome_data_2 = full_genome_data[index_2, :]
    genome_data_3 = full_genome_data[index_3, :]
    genome_data_4 = full_genome_data[index_4, :]

    # Number of samples per class. For eg. if max_samples is 4, we compute
    # training with 1, 2 3 and 4 samples. Each of these training is done
    # with N random trials.
    samples_per_class = np.arange(1, max_samples + 1, 1)

    # Initialization for Neurochaos
    accuracy_neurochaos = np.zeros((len(samples_per_class), 1))
    fscore_neurochaos = np.zeros((len(samples_per_class), 1))
    standard_deviation_fscore_neurochaos = np.zeros((len(samples_per_class), 1))

    # Initialization for SVM
    accuracy_svm = np.zeros((len(samples_per_class), 1))
    fscore_svm = np.zeros((len(samples_per_class), 1))
    standard_deviation_fscore_svm = np.zeros((len(samples_per_class), 1))

    for num_instance in samples_per_class:
        # Neurochaos array for appending accuracy and f1score
        accuracy_neurochaos_list = []
        f1score_neurochaos_list = []

        # SVM array for appending accuracy and f1score
        accuracy_svm_list = []
        f1score_svm_list = []
        # We do N random trials of training
        for trial_number in range(0, trials):

            train_neurochaos_genome_data_0, test_neurochaos_genome_data_0, train_neurochaos_genome_label_0, test_neurochaos_genome_label_0 = train_test_split(neurochaos_genome_data_0, full_genome_label[index_0], test_size=1 - (num_instance/neurochaos_genome_data_0.shape[0]), random_state=trial_number)
            train_neurochaos_genome_data_1, test_neurochaos_genome_data_1, train_neurochaos_genome_label_1, test_neurochaos_genome_label_1 = train_test_split(neurochaos_genome_data_1, full_genome_label[index_1], test_size=1 - (num_instance/neurochaos_genome_data_1.shape[0]), random_state=trial_number)
            train_neurochaos_genome_data_2, test_neurochaos_genome_data_2, train_neurochaos_genome_label_2, test_neurochaos_genome_label_2 = train_test_split(neurochaos_genome_data_2, full_genome_label[index_2], test_size=1 - (num_instance/neurochaos_genome_data_2.shape[0]), random_state=trial_number)
            train_neurochaos_genome_data_3, test_neurochaos_genome_data_3, train_neurochaos_genome_label_3, test_neurochaos_genome_label_3 = train_test_split(neurochaos_genome_data_3, full_genome_label[index_3], test_size=1 - (num_instance/neurochaos_genome_data_3.shape[0]), random_state=trial_number)
            train_neurochaos_genome_data_4, test_neurochaos_genome_data_4, train_neurochaos_genome_label_4, test_neurochaos_genome_label_4 = train_test_split(neurochaos_genome_data_4, full_genome_label[index_4], test_size=1 - (num_instance/neurochaos_genome_data_4.shape[0]), random_state=trial_number)


            train_genome_data_0, test_genome_data_0, train_genome_label_0, test_genome_label_0 = train_test_split(genome_data_0, full_genome_label[index_0], test_size=1 - (num_instance/genome_data_0.shape[0]), random_state=trial_number)
            train_genome_data_1, test_genome_data_1, train_genome_label_1, test_genome_label_1 = train_test_split(genome_data_1, full_genome_label[index_1], test_size=1 - (num_instance/genome_data_1.shape[0]), random_state=trial_number)
            train_genome_data_2, test_genome_data_2, train_genome_label_2, test_genome_label_2 = train_test_split(genome_data_2, full_genome_label[index_2], test_size=1 - (num_instance/genome_data_2.shape[0]), random_state=trial_number)
            train_genome_data_3, test_genome_data_3, train_genome_label_3, test_genome_label_3 = train_test_split(genome_data_3, full_genome_label[index_3], test_size=1 - (num_instance/genome_data_3.shape[0]), random_state=trial_number)
            train_genome_data_4, test_genome_data_4, train_genome_label_4, test_genome_label_4 = train_test_split(genome_data_4, full_genome_label[index_4], test_size=1 - (num_instance/genome_data_4.shape[0]), random_state=trial_number)

            print("num of samples per class = ", num_instance, "trial number = ", trial_number)

            test_neurochaos_genome_data = np.concatenate((test_neurochaos_genome_data_0, test_neurochaos_genome_data_1, test_neurochaos_genome_data_2, test_neurochaos_genome_data_3, test_neurochaos_genome_data_4))
            test_neurochaos_genome_label = np.concatenate((test_neurochaos_genome_label_0, test_neurochaos_genome_label_1, test_neurochaos_genome_label_2, test_neurochaos_genome_label_3, test_neurochaos_genome_label_4))


            train_neurochaos_genome_data = np.concatenate((train_neurochaos_genome_data_0, train_neurochaos_genome_data_1, train_neurochaos_genome_data_2, train_neurochaos_genome_data_3, train_neurochaos_genome_data_4))
            train_neurochaos_genome_label = np.concatenate((train_neurochaos_genome_label_0, train_neurochaos_genome_label_1, train_neurochaos_genome_label_2, train_neurochaos_genome_label_3, train_neurochaos_genome_label_4))


            test_genome_data = np.vstack((test_genome_data_0, test_genome_data_1, test_genome_data_2, test_genome_data_3, test_genome_data_4))
            test_genome_label = np.vstack((test_genome_label_0, test_genome_label_1, test_genome_label_2, test_genome_label_3, test_genome_label_4))

            train_genome_data = np.vstack((train_genome_data_0, train_genome_data_1, train_genome_data_2, train_genome_data_3, train_genome_data_4))
            train_genome_label = np.vstack((train_genome_label_0, train_genome_label_1, train_genome_label_2, train_genome_label_3, train_genome_label_4))

            # Neurochaos feature extraction followed by SVM classifier
            classifier_neurochaos = LinearSVC(random_state=0, tol=1e-5, dual=False)
            classifier_neurochaos.fit(train_neurochaos_genome_data, train_neurochaos_genome_label[:, 0])
            predicted_labels_neurochaos = classifier_neurochaos.predict(test_neurochaos_genome_data)

            accuracy_neurochaos_val = accuracy_score(test_neurochaos_genome_label, predicted_labels_neurochaos)*100

            f1score_neurochaos_val = f1_score(test_neurochaos_genome_label, predicted_labels_neurochaos, average="macro")

            accuracy_neurochaos_list.append(accuracy_neurochaos_val)
            f1score_neurochaos_list.append(f1score_neurochaos_val)


            classifier_svm = LinearSVC(random_state=0, tol=1e-5, dual=False)
            classifier_svm.fit(train_genome_data, train_genome_label[:, 0])
            predicted_labels_svm = classifier_svm.predict(test_genome_data)

            accuracy_svm_val = accuracy_score(test_genome_label, predicted_labels_svm)*100

            f1score_svm_val = f1_score(test_genome_label, predicted_labels_svm, average="macro")

            accuracy_svm_list.append(accuracy_svm_val)
            f1score_svm_list.append(f1score_svm_val)


        accuracy_neurochaos[num_instance-1, 0] = np.mean(accuracy_neurochaos_list)
        fscore_neurochaos[num_instance-1, 0] = np.mean(f1score_neurochaos_list)
        standard_deviation_fscore_neurochaos[num_instance-1, 0] = np.std(f1score_neurochaos_list)


        accuracy_svm[num_instance-1, 0] = np.mean(accuracy_svm_list)
        fscore_svm[num_instance-1, 0] = np.mean(f1score_svm_list)
        standard_deviation_fscore_svm[num_instance-1, 0] = np.std(f1score_svm_list)

    print("Saving Results")

    path = os.getcwd()
    result_path = path + '/NEUROCHAOS-LTS-RESULTS/'  + classification_type + '/LTS/'


    try:
        os.makedirs(result_path)
    except OSError:
        print("Creation of the result directory %s failed" % result_path)
    else:
        print("Successfully created the result directory %s" % result_path)

    np.save(result_path+"/fscore_neurochaos.npy", fscore_neurochaos)
    np.save(result_path+"/std_fscore_neurochaos.npy", standard_deviation_fscore_neurochaos)
    np.save(result_path+"/samples_per_class.npy", samples_per_class)

    np.save(result_path+"/fscore_svm.npy", fscore_svm)
    np.save(result_path+"/std_fscore_svm.npy", standard_deviation_fscore_svm)
    return fscore_neurochaos, standard_deviation_fscore_neurochaos, fscore_svm, standard_deviation_fscore_svm
