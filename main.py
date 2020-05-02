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

def Baseline(X_mat, y_vec, y_test):
    counts = list()
    for i in range(10):
        column = y_vec[:, i]
        counts.append(np.sum(column))
    winner = counts.index(max(counts))
    accuracy = np.sum(y_test[:, winner]) / y_test.shape[0]
    return accuracy

## Load the spam data set and Scale the input matrix
def Parse(fname):
    all_rows = []
    with open(fname) as fp:
        count = 0
        for line in fp:
            line = line.strip()
            row = line.split(' ')
            all_rows.append(row)
            count += 1
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
# for loop over foldID+ for foldID in (1:5)
convolution_list = list()
deep_list = list()
baseline_list = list()
for i in range(1, 6):
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
    epochs = 5
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
            keras.layers.Flatten(),
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

    ## Re-fit the model on the entire train set using best_epochs and validation_split=0.

    # choose the min validation loss and tran loss
    con_min_val_loss = min(con_results.history['val_loss'])
    deep_min_val_loss = min(deep_results.history['val_loss'])

    con_best_epoch = con_results.history['val_loss'].index(con_min_val_loss) + 1
    deep_best_epoch = deep_results.history['val_loss'].index(deep_min_val_loss) + 1

    #retrain convolution model
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

    con_results = convolution_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=con_best_epoch)

    convolution_list.append(con_results.history['val_acc'][-1] * 100)

    #retrain deep model 
    deep_model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(784, activation='relu'),
            keras.layers.Dense(270, activation='relu'),
            keras.layers.Dense(270, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_class, activation='softmax')
        ])

    deep_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    deep_results = deep_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=deep_best_epoch)

    deep_list.append(deep_results.history['val_acc'][-1] * 100)

    baseline_acc = Baseline(X_train, y_train, y_test)
    baseline_list.append(baseline_acc)

# end of loop

print(convolution_list)
print(deep_list)
print(baseline_list)

auc = dict()
auc['convolution'] = convolution_list
auc['deep'] = deep_list
auc['baseline'] = baseline_list

plt.xlabel("area")
plt.ylabel("algorithm")
for algorithm in auc:
    for value in auc[algorithm]:
        my_color = None
        if(algorithm == 'baseline'):
            my_color = 'black'
        elif(algorithm == 'convolution'):
            my_color = 'red'
        elif(algorithm == 'deep'):
            my_color = 'blue'
        plt.scatter(value, algorithm, color=my_color)
plt.tight_layout()
plt.savefig("auc_plot.png")
