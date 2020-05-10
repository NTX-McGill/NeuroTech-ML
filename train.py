#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:16:38 2020

@author: marley
"""
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix,confusion_matrix

from siggy.constants import *
from featurize import compute_features,all_names

#ALL_FEATURES = ['iemg','mav','mmav','var', 'var_abs', 'rms','rms_3', 'wl', 'zc','ssc', 'wamp','freq_feats','freq_var']
ALL_FEATURES = ['iemg', 'mav', 'mmav', 'var', 'var_abs', 'rms', 'rms_3', 'wl', 'zc', 'ssc', 'wamp', 'freq_feats', 'freq_feats_min_max', 'freq_var', 'freq_misc'] 

LABEL_MAP = {'k': 3, ';':5, 'j': 2, 'l': 4, 'p': 5, 'u': 2, 'o':4, '.': 4,
          'm':2, 'n': 2, '[':5, ']': 5, "'": 5, 'h': 2, '/':5, '\\':5,
          'a':10, 'c': 8, 'f': 7, 's': 9, 'd':8, 'e':8, 'g':7, 'q':10, 'r':7, 't':7, 'v':7, 'w':9, 'x':9, 'z':10
          }
ALL_MODELS = {
          'LR': LogisticRegression,
          'LDA': LinearDiscriminantAnalysis,
          'KNN': KNeighborsClassifier,
          'CART': DecisionTreeClassifier,
          'NB' : GaussianNB,
          'SVM': SVC
          }

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
    baseline_samples = df[np.logical_not(df['keypressed'].notnull())].sample(n=n_baseline_samples, replace=False)
    df.loc[baseline_samples.index, ['keypressed']] = 0

def print_dataset_stats(df):
    print(df['finger'].value_counts())
    print(df['mode'].value_counts())
    
def test_all_models(X, Y, model_names, scoring='accuracy', n_splits=10):
    results = []
    for name in model_names:
        model = ALL_MODELS[name]() # instantiate model from dictionary
        
        kfold = model_selection.KFold(n_splits=n_splits)
        
        # print f1 scores, but only save accuracy (as cv_results)
        f1micro = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1_micro')
        f1macro = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1_macro')
        f1w = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1_weighted')
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        
        print("model is ", name)
        print("micro is ", f1micro, "avg is ", sum(f1micro)/len(f1micro))
        print("macro is ", f1macro, "avg is ", sum(f1macro)/len(f1macro))
        print("weighted is ", f1w, "avg is ", sum(f1w)/len(f1w))
        print("accuracy is ", cv_results, "avg is ", sum(cv_results)/len(cv_results))
        
        results.append(cv_results)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return results

def extract(df, cols, shuffle=False):
    # extract X and Y numpy arrays
    dataset = df[cols].to_numpy()
    if shuffle:
        np.random.shuffle(dataset)
    X = dataset[:,:-1]
    Y = dataset[:,-1]
    return X, Y

def plot_confmat(classifier, X_validation, Y_validation, plot_counts=True, title="", ax=None):
    # Plot non-normalized confusion matrix
    titles_options = [ (title + ", normalized", 'true')]
    if plot_counts:   
        titles_options.append((title + ", without normalization", None))
        
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_validation, Y_validation,
                                      # display_labels=[2,3,4,5,'base'], # this might have been false
                                      cmap=plt.cm.Blues,
                                      normalize=normalize,
                                      ax=ax)
        disp.ax_.set_title(title)
    
        print(title)
        print(disp.confusion_matrix)
    return disp.confusion_matrix

def train_and_plot_confmat(model_name, X_train, Y_train, X_validation, Y_validation, plot_counts=True, title="", ax=None):
    
    classifier = ALL_MODELS[model_name]()
    classifier.fit(X_train, Y_train)
    np.set_printoptions(precision=2)
    confusion_matrix = plot_confmat(classifier, X_validation, Y_validation, plot_counts=plot_counts, title=title, ax=ax)
    return classifier, confusion_matrix

def run_test_confmat_split(train, test, model_name, cols, plot_counts=False, title="", ax=None):
    # when the train/test split is provided
    X_train, Y_train = extract(train, cols)
    X_validation, Y_validation = extract(test, cols)
    return train_and_plot_confmat(model_name, X_train, Y_train, X_validation, Y_validation, plot_counts=plot_counts, title=title, ax=ax)
    
def run_test_confmat_single_fold(X, Y, model_name, validation_size=0.2, plot_counts=True, title="", ax=None):
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, shuffle=False)

    # classifier = svm.SVC(kernel='linear').fit(X_train, Y_train)
    return train_and_plot_confmat(model_name, X_train, Y_train, X_validation, Y_validation,plot_counts=plot_counts, title=title, ax=ax)

def run_test_confmat_folds(X, Y, model_name, validation_size=0.2,plot_counts=True, title='', ax=None):
    confuzzle = []
    kf = model_selection.KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        Y_train, Y_validation = Y[train_index], Y[test_index]
        
        classifier = ALL_MODELS[model_name]()
        classifier.fit(X_train, Y_train)
        np.set_printoptions(precision=2)
        
        Y_predict = classifier.predict(X_validation)
        
        Y_validation, Y_predict = np.append(Y_validation, np.arange(11)), np.append(Y_predict, np.arange(11))
        
        if len(confuzzle) == 0:
            confuzzle = confusion_matrix(Y_validation, Y_predict)
        else:
            confuzzle += confusion_matrix(Y_validation, Y_predict) - np.identity(11, dtype=int)
    
    
    df_cm = pd.DataFrame(confuzzle, index = [i for i in range(11)], columns = [i for i in range(11)])
    print(df_cm)
    if plot_counts:
        # raw
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
        plt.title('Not Normalized\n'+title)
        plt.show()
    # normalized
    df_cm = df_cm.apply(lambda row: row / np.sum(row), axis=1)
    if ax:
        ax.set_title(title + ", normalized")
    else: 
        plt.figure(figsize = (10,7))
        plt.title(title + ', normalized')
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, ax=ax)
    
    return classifier, df_cm

def run_test_confmat(X, Y, model_name, validation_size=0.2, test_all_folds=True, plot_counts=True, title='', ax=None):
    if test_all_folds:
        return run_test_confmat_folds(X, Y, model_name, validation_size=validation_size,plot_counts=plot_counts, title=title, ax=ax)
    return run_test_confmat_single_fold(X, Y, model_name, validation_size=validation_size,plot_counts=plot_counts, title=title, ax=ax)

def leave_subject_out_crossval(features, cols, model_name='LDA'):
    plt.figure(figsize=(48,20))
    for subject in features['id'].unique():
        ax = plt.subplot(2,6,subject)
        train = features[features.id != subject]
        test = features[features.id == subject]
        run_test_confmat_split(train, test, model_name, cols, title="Subject {}".format(subject), ax=ax)

def within_subject(features, cols, validation_size=0.2, test_all_folds=False):
    plt.figure(figsize=(48,20))
    for subject in features['id'].unique():
        ax = plt.subplot(2,6,subject)
        single_subject = features[features.id == subject]
        X, Y = extract(single_subject, cols)
        classifier, result = run_test_confmat(X,Y, model_name, test_all_folds=test_all_folds, validation_size=validation_size,plot_counts=False,
                                              title="Subject {}".format(subject), ax=ax)

# output model to pickle file
def generate_model_name(filename=""):
    now = datetime.now()
    name = "model_" + filename + "-" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".pkl"
    return name

def save_model(classifier, feature_names, filename="", name=None):
    if not name:
        name = generate_model_name(filename)
        print(name)
    with open(name, 'wb') as f:
        pickle.dump({'classifier': classifier, 'features': feature_names}, f)

def get_feature_names(df):
    """
    Returns a list of unique names of features in a dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing windows and feature information of said windows

    Returns
    -------
    list
        Unique names of features in df.

    """
    
    #Get names of features in dataframe, prefixed by 'channel #_'
    feature_names_with_ch = [col for col in df.columns if '_' in col]
    
    #Remove prefix
    feature_names_no_ch = ['_'.join(name.split('_')[1:]) for name in feature_names_with_ch]
    
    #Get unique feature names
    return list(set(feature_names_no_ch))

def add_ch_to_feature_name(feats):
    """
    Appends the names of channels to each feature name in feats, in order to 
    get the value of the feature in every channel

    Parameters
    ----------
    feats : list
        Names of features.

    Returns
    -------
    feat_names_with_ch : list
        Strings of the form: 'channel i_(feature name)'.

    """
    
    channel_names = ['channel {}'.format(i) for i in range(1, 9)]
    
    #Prepend 'channel #' to every feature passed
    feat_names_with_ch = []
    for f in feats:
        feat_names_with_ch.extend(['_'.join([ch, f]) for ch in channel_names])
    
    return feat_names_with_ch

def get_subset(df, ids=None, modes=None, feats=None):
    """
    Returns a subset of df which conatins only subject ids, modes, and features specified.

    Parameters
    ----------
    df : DataFrame
        Dataframe to extract information from.
    ids : list, optional
        List of subject ids (ints or strings) to include in new dataframe. 
        The default is None.
    modes : list, optional
        List of modes (ints) to include in new dataframe. 
        The default is None.
    feats : list, optional
        List of feature names (strings) to include in new dataframe. 
        Feature names should match the name of the function used in their calculation.
        The default is None.

    Returns
    -------
    returned_df : DataFrame
        Subset of df, containing only subject ids, modes, and features specified

    """
    
    #If nothing passed, no need to make a subset - can just use df
    if not (ids or modes or feats):
        raise ValueError('No parameters were passed. The DataFrame passed is already a subset of itself.')
    
    returned_df = df.copy()
    
    if feats:
        #Error check features
        feats_in_df = get_feature_names(df)
        incorrect_feats = [f for f in feats if f not in feats_in_df]
        if len(incorrect_feats) > 0:
            raise ValueError('Invalid feature(s): {}. Available features are the following: {}.'.format(
                incorrect_feats, feats_in_df))
            
        #Keep non-feature information and features specified
        cols = [c for c in df.columns if '_' not in c] + add_ch_to_feature_name(feats)
        returned_df = returned_df[cols]
    
    if ids:
        #Error check ids
        ids_in_df = df['id'].unique()
        incorrect_ids = [subject_id for subject_id in ids if int(subject_id) not in ids_in_df]
        if len(incorrect_ids) > 0:
            raise ValueError('Invalid ID(s): {}. Available IDs are the for following: {}'.format(
                incorrect_ids, ids_in_df))
        
        #Keep only rows with speficied ids
        returned_df = returned_df.loc[returned_df['id'].isin(ids)]
    
    if modes:
        #Error check modes
        modes_in_df = df['mode'].unique()
        incorrect_modes = [mode for mode in modes if mode not in modes_in_df]
        if len(incorrect_modes) > 0:
            raise ValueError('Invalid mode(s): {}. Available modes are the following: {}.'.format(
            incorrect_modes, modes_in_df))
        
        #Keep only rows with specified modes
        returned_df = returned_df.loc[returned_df['mode'].isin(modes)]
    
    if returned_df.empty:
        print("***WARNING: The subset you requested is empty!***")
    
    returned_df.reset_index(drop=True, inplace=True)
    return returned_df

def grid_search(df, models, id_params=None, mode_params=None, feature_params=None, scoring='accuracy', n_splits=10):
    """
    Exhaustively trains all models passed on all combinations of given parameters.

    Parameters
    ----------
    df : DataFrame
        Dataframe of all the calculated features.
    models : list
        List of model names.
    id_params : list, optional
        List of combinations (list) of subject ids. 
        The default is None.
    mode_params : list, optional
        List of combinations (list) of modes. 
        The default is None.
    feature_params : list, optional
        List of combinations (list) of features. 
        The default is None.
    scoring : string, optional
        How cross-validation is scored. 
        The default is 'accuracy'.
    n_splits : int, optional
        Number of splits to use in cross-validation. 
        The default is 10.

    Returns
    -------
    results : 
        Array saving all cross-validation results.
    """
    
    #If no parameters are passed, confirm that user wants to use all the data
    if not (id_params or mode_params or feature_params):
        print('***Warning: Test on the entire dataset? (y)***')
        response = input().lower()
        
        if response != 'y':
            print('Aborting grid search')
            return
    
    #If a parameter is not passed, use all values found in df
    if not id_params:
        id_params = [df['id'].unique()]
    if not mode_params:
        mode_params = [df['mode'].unique()]
    if not feature_params:
        feature_params = [get_feature_names(df)]
    
    results = [[[[ [] for feats in feature_params] 
                      for modes in mode_params] 
                      for subjs in id_params] 
                      for model in models]
    for i, model in enumerate(models):
        for j, subjs in enumerate(id_params):
            for k, modes in enumerate(mode_params):
                for l, feats in enumerate(feature_params):
                    print('Current combination being test:\n',
                          'model: {}\n'.format(model),
                          'Subjects: {}\n'.format(subjs),
                          'modes: {}\n'.format(modes),
                          'features: {}\n'.format(feats))
                    
                    #Get subset of df containing current settings
                    df_subset = get_subset(df, subjs, modes, feats)
                    
                    #Extract features and labels from df
                    cols = add_ch_to_feature_name(feats) + ['finger']
                    dataset = df_subset[cols].to_numpy()
                    
                    #Split dataset into features and labels
                    X = dataset[:, :-1]
                    Y = dataset[:, -1]
                    
                    print('Size of dataset:', X.shape)
                    
                    #Instantiate model from dictionary
                    classifier = ALL_MODELS[model]()
                    
                    #Do cross-validation
                    kfold = model_selection.KFold(n_splits=n_splits)
                    cv_results = model_selection.cross_val_score(classifier, X, Y, cv=kfold, scoring=scoring)
                    
                    results[i][j][k][l].append(cv_results)
                    msg = "%s: %f (%f)" % (model, cv_results.mean(), cv_results.std())
                    print(msg)
                    
                    clf, confmat = run_test_confmat(X, Y, model, validation_size=validation_size, test_all_folds=test_all_folds)
                    results[i][j][k][l].append(confmat)

    return results
 #%% 
if __name__ == '__main__':
    """
    Here there are two different modes you can run, either you load the windows file and compute the features
    or you can load the features directly.    
    """
    
    # features to use
    feature_names = ALL_FEATURES[:-1]
    # models to use
    model_names = ['LDA', 'CART', 'LR', 'KNN', 'SVM', 'NB']
    test_all_folds=True
    n_splits=10
    validation_size = 0.20
    
    
    window_size = 2
    channels = [1,2,3,4,5,6,7,8]
    label_name='finger'
    
    # filename = 'windows_date_all_subject_all_mode_1_2.pkl'
    # filename ='siggy/windows/windows_date_2020-03-08_subject_all_mode_4.pkl'
    # filename = 'features_windows-2020-03-03.pkl'
    filename = 'features_windows_date_all_subject_all_mode_1_2_4_groups_ok_good.pkl'

    if 'features' in filename:
        ### MODE 1 : LOAD THE FEATURES DIRECTLY
        
        file_prefix = filename.split(".")[0].split('/')[-1]
        features = pd.read_pickle(filename)
        
        all_ch_names = [i for i in features.columns if 'channel' in i]
        all_ch_names = [i for i in all_ch_names if '_' in i]
        
    else:
        ### MODE 2 : LOAD THE WINDOWS, COMPUTE THE FEATURES
        
        file_prefix = filename.split(".")[0].split('/')[-1]
        channel_names = ['channel {}'.format(i) for i in channels]
        
        df = load_windows(filename, channels)
        labels = df[label_name].unique()
        sample_baseline(df, labels)
        subset = df[df[label_name].notnull()]
        #%% 
        features, all_ch_names = compute_features(subset, channel_names, feature_names, mutate=True)
        # write pickle file
        feat_filename = 'features_' + filename
        with open(feat_filename, 'wb') as f_out:
            pickle.dump(features, f_out)
            print('Saved windows to file {}'.format(filename))
    
    print_dataset_stats(features)
    cols = all_ch_names + [label_name] 
    
    # Add grouping and shuffle so that folds are more randomized
    # Hoping to lower standard deviation between folds
    groups = [df for _,df in features.groupby([(features.finger != features.finger.shift()).cumsum()])]
    random.shuffle(groups)
    features_shuffled = pd.concat(groups).reset_index(drop=True)
    
    X, Y = extract(features_shuffled, cols)
    results = test_all_models(X,Y, model_names, n_splits=n_splits)
    
    """ UNCOMMENT TO RUN AND SAVE LDA
    #%% 
    model_name = 'LDA'
    classifier, result = run_test_confmat(X,Y, model_name, test_all_folds=test_all_folds, validation_size=validation_size,
                                          title='file : '+filename)
    
    #%% 
    
    subset = features[features['mode'] == 1]

    leave_subject_out_crossval(subset, cols)
    within_subject(features, cols)
    
    save_model(classifier, feature_names, file_prefix)
    
    # models = ['LDA', 'SVM']
    # ids = [[1]]
    # modes = [[1], [2]]
    # feats = [['mav', 'zc', 'ssch', 'wl']]
    # test_results = grid_search(features, models, id_params=ids, mode_params=modes, feature_params=feats)
    """

