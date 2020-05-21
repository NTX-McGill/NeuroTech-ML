#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:54:39 2020

@author: roland
"""
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Sampling frequency of the OpenBCI
SAMPLING_FREQ = 250

#Indices of the channel in the OpenBCI raw data file
CHANNELS = [1,2,3,4,5,6,7,8]

#Column names of the EMG data DataFrame
DATA_COLS = ['channel 1', 'channel 2', 'channel 3', 'channel 4', 
             'channel 5', 'channel 6', 'channel 7', 'channel 8', 
             'timestamp(ms)', 'hand', 'finger', 'keypressed', 'id', 'mode']

#Column names of the windowed data DataFrame
WINDOW_COLS = ['channel 1', 'channel 2', 'channel 3', 'channel 4', 
               'channel 5', 'channel 6', 'channel 7', 'channel 8', 
               'hand', 'finger', 'keypressed', 'id', 'mode']

#Mapping of data collection mode to int
MODE_MAP = {'Guided': 1, 'Self-directed': 2, 'In-the-air': 3, 'Guided-in-the-air':4}    

#Mapping of hand used to press key to int
HAND_MAP = {'left': 1, 'right': 2}

#Mapping of finger used to press key to int
HAND_FINGER_MAP = {'left' : {'thumb': 6, 'index finger': 7, 'middle finger': 8, 'ring finger': 9, 'pinkie': 10},
                   'right': {'thumb': 1, 'index finger': 2, 'middle finger': 3, 'ring finger': 4, 'pinkie': 5}}

#Mapping of keys to finger (same as in HAND_FINGER_MAP)
LABEL_MAP = {'1':10, '2':9, '3':8, '4':7, '5':7, '6': 2, '7':2, '8':3, '9':4, '0': 5,
             'q':10, 'w':9, 'e':8, 'r':7, 't':7, 'y':2, 'u':2, 'i':3, 'o':4, 'p':5,
             'a':10, 's':9, 'd':8, 'f':7, 'g':7, 'h':2, 'j':2, 'k':3, 'l':4, ';':5,
             'z':10, 'x':9, 'c':8, 'v':7, 'b':7, 'n':2, 'm':2, ',':3, '.':4, '/':5,
             '[':5, ']':5, "'":5, '\\':5 , 'space': 1, 'Shift': 10, 'Backspace':5, '`': 10,
             '=': 5,
             'baseline':0}

#All features currently implemented
ALL_FEATURES = ['iemg','mav','mmav','var', 'var_abs', 'rms','rms_3', 'wl', 'zc','ssc', 'wamp','freq_feats','freq_var']

#All models used for training
ALL_MODELS = {
          'LR': LogisticRegression,
          'LDA': LinearDiscriminantAnalysis,
          'KNN': KNeighborsClassifier,
          'CART': DecisionTreeClassifier,
          'NB' : GaussianNB,
          'SVM': SVC
          }

#Random seed used for training
SEED = 7