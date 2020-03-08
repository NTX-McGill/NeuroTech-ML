#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:16:38 2020

@author: marley
"""
from featurize import compute_features,all_names
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
import pickle
from datetime import datetime

ALL_FEATURES = ['iemg','mav','mmav','mmav2','var','rms','rms_3','zc','wamp','wl','ssch','wfl','freq_feats','freq_var']
LABEL_MAP = {'k': 3, ';':5, 'j': 2, 'l': 4, 'p': 5, 'u': 2, 'o':4, '.': 4,
          'm':2, 'n': 2, '[':5, ']': 5, "'": 5, 'h': 2, '/':5, '\\':5,
          'a':10, 'c': 8, 'f': 7, 's': 9, 'd':8, 'e':8, 'g':7, 'q':10, 'r':7, 't':7, 'v':7, 'w':9, 'x':9, 'z':10
          }
SEED = 7

def load_windows(filename, channels):
    channel_names = ['channel {}'.format(i) for i in channels]
    # reads file
    df = pd.read_pickle(filename)
    df.reset_index(inplace=True)
    # set labels
    df['keypressed'] = df['keypressed'].map(LABEL_MAP)
    print("Key press values: {}".format(df['keypressed'].unique()))
    return df

def sample_baseline(df, labels, baseline_sample_factor=1):
    """
    Select a subset of the baseline: convert the selected rows' label from NaN to 0
    runs in-place

    Parameters
    ----------
    df : pd.DataFrame
    labels : list
    baseline_sample_factor : int
        default 1 -> same number of baseline samples as single class
        represents the amount to multiply the number of samples

    Returns
    -------
    None.

    """
    n_baseline_samples = int(np.sum(df['keypressed'] > 0)/len(labels)) * baseline_sample_factor
    baseline_samples = df[np.logical_not(df['keypressed'].notnull())].sample(n=n_baseline_samples, replace=False, random_state=SEED)
    df.loc[baseline_samples.index, ['keypressed']] = 0
    
def generate_model_name(filename=""):
    now = datetime.now()
    name = "model_" + filename + "-" + now.strftime("%m_%d_%Y_%H:%M:%S") + ".pkl"
    return name

def save_model(classifier, filename="", name=None):
    if not name:
        name = generate_model_name(filename)
    with open(name, 'wb') as f:
        pickle.dump(classifier, f)
    

# features to use
feature_names = ALL_FEATURES[:]

window_size = 2
channels = [1,2,3,4,5,6,7,8]
filename = 'windows-2020-02-23.pkl'
file_prefix = filename.split(".")[0]
channel_names = ['channel {}'.format(i) for i in channels]


df = load_windows(filename, channels)
labels = df['keypressed'].unique()
sample_baseline(df, labels)
subset = df[df['keypressed'].notnull()]
features, all_ch_names = compute_features(subset, channel_names, feature_names, mutate=True)

cols = all_ch_names + ['keypressed'] 
# cols = all_names(channel_names, feature_names) + ['keypressed']
dataset = features[cols].to_numpy()

# don't shuffle the dataset
# np.random.shuffle(dataset)
X = dataset[:,:-1]
Y = dataset[:,-1]
print("size of dataset:", X.shape)


validation_size = 0.20

# Test options and evaluation metric
scoring = 'accuracy'
# Spot Check Algorithms
models = []
# models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=SEED)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
    
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, shuffle=False)

# classifier = svm.SVC(kernel='linear').fit(X_train, Y_train)
classifier = LinearDiscriminantAnalysis().fit(X_train, Y_train)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_validation, Y_validation,
                                 # display_labels=[2,3,4,5,'base'], # this might have been false
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

# output model to pickle file
save_model(classifier, file_prefix)




