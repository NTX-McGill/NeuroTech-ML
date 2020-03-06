#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:41:42 2020

@author: marley
"""
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import seaborn as sns
from features import *


def get_name(channel_name, feature_name):
    return "{}_{}".format(channel_name, feature_name)

def all_names(channel_names, feature_names):
    names = []
    for channel_name in channel_names:
        for feature_name in feature_names:
            names.append(get_name(channel_name, feature_name))
    return names

def compute_feature(df, channel_names, feature_name, to_df=True):
    """
    Get features from window, non-mutating
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with windows
    channel_names : list of strings
    feature_name : string
        string name of the feature function
    to_df : bool
        if output should be converted to a dataframe

    Returns
    -------
    df_result : pd.DataFrame (to_df = True) or dictionary
        new dataframe with feature columns for each channel

    """
    fn = globals()[feature_name]
    computed_features = []
    for _, row in df.iterrows():
        val = [fn(row[channel_name]) for channel_name in channel_names]
        computed_features.append(val)
    computed_features = np.array(computed_features).T
    
    result = {get_name(channel_name, feature_name): feature
                              for channel_name, feature in zip(channel_names, computed_features)}
    if to_df:
        result = pd.DataFrame(result)
    return result

def compute_features(df, channel_names, feature_names, mutate=False):
    """
    Get features from window, non-mutating
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with windows
    channel_names : list of strings
    feature_names : list of strings

    Returns
    -------
    df_result : pd.DataFrame
        new dataframe with feature columns for each channel

    """
    all_results = {}
    for feature_name in feature_names:
        result = compute_feature(df, channel_names, feature_name, to_df=False)
        all_results.update(result)
    if mutate:
        for key, val in all_results.items():
            df[key] = val
        return df
    df_result = pd.DataFrame(all_results)
    # TODO: extract other 'labels' from df?
    df_result['keypressed'] = df['keypressed']
    return df_result



if __name__ == "__main__":
    # features to use
    # options are: iemg, mav, mmav, var, rms, zc, wamp, wl
    feature_names = ['mav', 'var', 'rms', 'name_of_new']
    channels = [1,2,3,4]
    channel_names = ['channel {}'.format(i) for i in channels]
    filename = 'windows-2020-02-16.pkl'
    labels = {'k': 3, ';':5, 'j': 2, 'l': 4, 'p': 5, 'u': 2, 'o':4, '.': 4,
              'm':2, 'n': 2, '[':5, ']': 5, "'": 5, 'h': 2, '/':5, '\\':5}
    
    df = pd.read_pickle(filename)
    df['keypressed'] = df['keypressed'].map(labels)
    print("Key press values: {}".format(df['keypressed'].unique()))
    
    features = compute_features(df, channel_names, feature_names)