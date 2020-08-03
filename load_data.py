"""
This module loads the genome sequence corresponding to SARS-COV-2 and other
coronaviruses. The genome sequence is mapped to the following numeric values:
    C = 0.25
    T = 0.50
    G = 0.75
    A = 1.0
The absolute value of the fast fourier transform coefficients is found next.
The module returns the absolute value of the fft coefficients of the input data
and labels.

Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
Date: 29 July 2020
"""

import logging
import numpy as np
import pandas as pd
from numpy.fft import fft


def get_data(classification_type):
    """
    Parameters
    ----------
    classification_type : string
        DESCRIPTION : classification_type == "binary_class"
        binary classification data will be loaded.
        classification_type == "multi_class"
        multiclass classification data is loaded

    Returns
    -------

    fourier_data_normalized : array, 2D
    labels : array, 2D

    """
    if classification_type == 'multi_class':

        data_covid = np.array(pd.read_csv(classification_type+'/data/data.csv', header=None))
        labels = np.array(pd.read_csv(classification_type+'/data/labels.csv', header=None))
        num_instance = data_covid.shape[0] # Number of rows in the data.
        num_features = data_covid.shape[1] # Number of columns in the data.
        fourier_features = np.zeros((num_instance, num_features))

        #Computing the absolute value Fast Fourier transform coefficients of each data instance.
        for data_instance in range(0, num_instance):

            fourier_features[data_instance, :] = np.abs(fft(data_covid[data_instance, :]))

        # Normalization done for each row.
        numerator = fourier_features.T - np.min(fourier_features, axis=1)
        denominator = np.max(fourier_features, axis=1) - np.min(fourier_features, axis=1)
        fourier_data_normalized = (numerator/denominator).T

        # Checking whether the data is normalized.
        try:
            assert np.min(fourier_data_normalized) >= 0.0 and np.max(fourier_data_normalized) <= 1.0
        except AssertionError:
            logging.error("Error-Data should be in the range [0, 1]", exc_info=True)

        return fourier_data_normalized, labels

    elif classification_type == 'binary_class':

        data_covid = np.array(pd.read_csv(classification_type + '/data/data.csv', header=None))
        labels = np.array(pd.read_csv(classification_type + '/data/labels.csv', header=None))
        num_instance = data_covid.shape[0] # Number of rows in the data.
        num_features = data_covid.shape[1] # Number of columns in the data.
        fourier_features = np.zeros((num_instance, num_features))

        # Computing the absolute value Fast Fourier transform coefficients of each data instance.
        for data_instance in range(0, num_instance):

            fourier_features[data_instance, :] = np.abs(fft(data_covid[data_instance, :]))

        # Normalization done for each row.
        numerator = fourier_features.T - np.min(fourier_features, axis=1)
        denominator = np.max(fourier_features, axis=1) - np.min(fourier_features, axis=1)
        fourier_data_normalized = (numerator/denominator).T
        # Checking whether the data is normalized.
        try:
            assert np.min(fourier_data_normalized) >= 0.0 and np.max(fourier_data_normalized) <= 1.0
        except AssertionError:
            logging.error("Error-Data should be in the range [0, 1]", exc_info=True)

        return fourier_data_normalized, labels

    else:

        data_path = "PREPROCESSED_DATA/"
        cov_1_data = np.load(data_path + "COV_1_DATA.npy")# SARS-COV-1 data
        cov_1_label = np.load(data_path + "COV_1_LABEL.npy")
        cov_2_data = np.load(data_path + "COV_2_DATA.npy")# SARS-COV-2 data
        cov_2_label = np.load(data_path + "COV_2_LABEL.npy")

        data_covid = np.vstack((cov_1_data, cov_2_data))
        labels = np.vstack((cov_1_label, cov_2_label))

        num_instance = data_covid.shape[0] # Number of rows in the data.
        num_features = data_covid.shape[1] # Number of columns in the data.
        fourier_features = np.zeros((num_instance, num_features))

        # Absolute value of fast fourier transform of the input data.
        for data_instance in range(0, num_instance):

            fourier_features[data_instance, :] = np.abs(fft(data_covid[data_instance, :]))

        #Normalization done for each row.
        numerator = fourier_features.T - np.min(fourier_features, axis=1)
        denominator = np.max(fourier_features, axis=1) - np.min(fourier_features, axis=1)
        fourier_data_normalized = (numerator/denominator).T

        # Checking whether the data is normalized.

        try:
            assert np.min(fourier_data_normalized) >= 0.0 and np.max(fourier_data_normalized) <= 1.0
        except AssertionError:
            logging.error("Error-Data should be in the range [0, 1]", exc_info=True)

        return fourier_data_normalized, labels
    