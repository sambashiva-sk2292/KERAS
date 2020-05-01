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

## Divide the data into 80% train, 20% test observations (out of all observations in the whole data set).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
