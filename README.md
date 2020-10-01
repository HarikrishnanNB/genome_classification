# Neurochaos-SVM based classification of SARS-CoV-2 among coronaviruses


## ChaosFEX - Feature Extraction using Chaos

### Neurochaos - SVM

[Video explanation on YouTube](https://www.youtube.com/watch?v=8JQstLi4COk) on the usage of chaotic maps as kernels and highlighting chief ideas and inspiration.

Reference Paper:

1) Harikrishnan NB and Nithin Nagaraj, "Neurochaos Inspired Hybrid Machine Learning Architecture for Classification", IEEE International Conference on Signal Processing and Communication (SPCOM 2020), IISc. Bengaluru (Virutal Conference), July 20-23, 2020. (ORAL-HKNB).  (Github link to the code https://github.com/pranaysy/ChaosFEX- courtesy - Dr. Pranay Yadav and Harikrishnan N B). 

2) Harikrishnan Nellippallil Balakrishnan, Aditi Kathpalia, Snehanshu Saha, Nithin Nagaraj, "ChaosNet: A chaos based artificial neural network architecture for classification", Chaos: An  Interdisciplinary  Journal  of  Non-linear  Science, Vol. 29, No. 11, pp. 113125-1 -- 113125-17 (2019); https://doi.org/10.1063/1.5120831.

### Dependencies

1. ``` Python 3```

2. ```Numpy```

3. ```Numba```

4. ```Matplotlib```

### Installation

1. git-clone into a working directory.

### How to run

1. unzip binary_class.7z (Data Source: https://github.com/albertotonda/deep-learning-coronavirus-genome/tree/master/Corona%20V5.2/data, Reference: Lopez-Rincon, A., Tonda, A., Mendoza-Maldonado, L., Claassen, E., Garssen, J., & Kraneveld, A. D. (2020). Accurate identification of sars-cov-2 from viral genome sequences using deep learning. bioRxiv.).
2. unzip multi_class.7z (Data Source: https://github.com/albertotonda/deep-learning-coronavirus-genome/tree/master/Corona%20V4.2, Reference: Lopez-Rincon, A., Tonda, A., Mendoza-Maldonado, L., Claassen, E., Garssen, J., & Kraneveld, A. D. (2020). Accurate identification of sars-cov-2 from viral genome sequences using deep learning. bioRxiv.).
3. unzip PREPROCESSED_DATA.7z (larger binary classificaton data). The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1ysvHLL879iHvoV5YX1Gj7UkRm6jbQ3WR?usp=sharing). 
4. main_hyperparameter_tuning.py - file to find the best hyperparametes. 
5. plot_main_hyperparameter_tuning.py - plots the average f1score for three fold validation vs. epsilon.
6. main_five_fold.py - file to do five fold validation for binary classificaion and multiclass classification.
7. main_binary_class_low_training_sample.py - executing file for low training sample regime for binary classification task.
8. plot_additional_data.py - plots for the the low training sample regime for the larger binary classification task.
9. main_multi_class_low_training_sample.py - executing file for low training sample regime for multiclass classification task.
10. plot_multi_class.py - plots for the the low training sample regime for the multiclass classification task.
11 hyperparameter_tuning_svm.py: Hyperparameter tuning code for SVM.


### Licence

Copyright 2020 [Harikrishnan N. B.](https://github.com/HarikrishnanNB), [Pranay Yadav](https://github.com/pranaysy), and [Nithin Nagaraj](https://sites.google.com/site/nithinnagaraj2/)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
   http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

