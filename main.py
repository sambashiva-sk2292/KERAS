# This Python file uses the following encoding: iso-8859-1

import argparse
import pdb # use pdb.set_trace() to set a "break point" when debugging
import os, sys
import numpy as np
import sys
import scipy
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path
import copy
import warnings
import statistics
from matplotlib.pyplot import figure
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# warnings.simplefilter('error') # treat warnings as errors
figure(num=None, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=30)

def Baseline(X_mat, y_vec, X_new):
    pred_new = np.zeros((X_new.shape[0],))
    if (y_vec == 1).sum() > (y_vec.shape[0] / 2):
        pred_new = np.where(pred_new == 0, 1, pred_new)
    return pred_new

## Load the spam data set and Scale the input matrix
def Parse(fname):
    all_rows = []
    with open(fname) as fp:
        for line in fp:
            line = line.strip()
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    temp_ar = temp_ar.astype(float)
    for col in range(1, temp_ar.shape[1]): # for all but last column (output)
        std = np.std(temp_ar[:, col])
        if(std == 0):
            print("col " + str(col) + " has an std of 0")
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    np.random.seed(0)
    np.random.shuffle(temp_ar)
    return temp_ar

parser = argparse.ArgumentParser(description='Compare serveral ML models')
parser.add_argument('input_file', type=str, help='input data')
args = parser.parse_args()
temp_ar = Parse(args.input_file)
X = temp_ar[:, 1:] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, 0]]).T 
y = y.astype(int)
num_row = X.shape[0]
X = X.reshape(num_row, 16, 16, 1)

#For 5-fold cross-validation, create a variable fold_vec which randomly assigns each observation to a fold from 1 to 5.
num_folds = 5
tempt_array = np.array([1,2,3,4,5])
fold_vec = np.repeat(tempt_array, num_row/num_folds)
np.random.shuffle(fold_vec)

#For each fold ID, you should create variables x_train, y_train, x_test, y_test based on fold_vec.
# for loop over foldID+++++++++++++++++++++++++++
foldID = 1
is_test = (fold_vec == foldID)
is_train = (fold_vec != foldID)
X_train = X[np.where(is_train)[0]]
y_train = y[np.where(is_train)[0]]
X_test = X[np.where(is_test)[0]]
y_test = y[np.where(is_test)[0]]
img_row = 16
img_col = 16
num_class = 10
num_obs = X_train.shape[0]
y_train = tf.keras.utils.to_categorical(y_train, num_class)
y_test = tf.keras.utils.to_categorical(y_test, num_class)

##Compute validation loss for each number of epochs, and define a variable best_epochs which is the number of 
#epochs that results in minimal validation loss.

#convolution model
epochs = 20
convolution_model = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    keras.layers.Conv2D(filters = 64, kernel_size = [3,3], activation = 'relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(rate = 0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(rate = 0.5),
    keras.layers.Dense(num_class, activation = 'softmax')
])

convolution_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

con_results = convolution_model.fit(X_train, y_train, validation_split = 0.2, epochs=epochs)

#deep model
deep_model = keras.Sequential([
        keras.layers.Flatten(input_shape = (num_obs, img_row, img_row, 1)),
        keras.layers.Dense(784, activation='relu'),
        keras.layers.Dense(270, activation='relu'),
        keras.layers.Dense(270, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_class, activation='softmax')
    ])

deep_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

deep_results = deep_model.fit(X_train, y_train, validation_split = 0.2, epochs=epochs)