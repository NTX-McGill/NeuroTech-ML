# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:35:12 2020

@author: marley, edited by miasya
"""

import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import seaborn as sns
from features import *
import pickle
import numpy as np

# New: requires match_labels.py for filtering the data
#from match_labels import *

class Prediction():
    def __init__(self, num_channels=8, shift=0.1, order=2, fs=250, 
                 notch_freq=60.0, low=5.0, high=50.0,
                 should_filter=True, model_filename=None):
        
        if (model_filename):
            # 'model_windows-2020-02-23-03_08_2020_15_48_56.pkl'
            with open(model_filename, 'rb') as f:
                data = pickle.load(f)
                self.clf = data['classifier']
                self.features = data['features']
                
        #Parameters for filters
        self.num_channels = num_channels
        self.shift = shift
        self.shift_samples = int(shift * 250)
        self.order = order
        self.sampling_freq = fs
        self.notch_freq = notch_freq
        self.low_pass = low
        self.high_pass = high
        
        
        self.channel_names = ['channel {}'.format(i) for i in range(1,9)]
        self.initialize_filters()
        
    def initialize_filters(self):
        #Set up the filters
        self.notch_b, self.notch_a = signal.iirnotch(self.notch_freq, self.notch_freq / 6, fs=self.sampling_freq)
        self.butter_b, self.butter_a = signal.butter(self.order, 
                                                     [self.low_pass / (self.sampling_freq / 2), self.high_pass / (self.sampling_freq / 2)], 
                                                     'bandpass')
        nz = signal.lfilter_zi(self.notch_b, self.notch_a)
        bz = signal.lfilter_zi(self.butter_b, self.butter_a)
        self.notch_z = [nz for i in range(self.num_channels)]
        self.butter_z = [bz for i in range(self.num_channels)]
        return
    
    def apply_filter(self, arr):
        # [8 x 250]
        
        #Filter each channel
        for i in range(self.num_channels):
            channel = arr[i]
            #Get conditions for channel
            temp_notch_z, temp_butter_z = self.notch_z[i], self.butter_z[i]
            
            #Notch filter
            for j, datum in enumerate(channel):
                filtered_sample, temp_notch_z = signal.lfilter(self.notch_b, self.notch_a, [datum], zi=temp_notch_z)
                channel[j] = filtered_sample[0]
                
                if j == self.shift_samples - 1:
                    self.notch_z[i] = temp_notch_z
            
            #Butterworth bandpass
            for j, datum in enumerate(channel):
                filtered_sample, temp_butter_z = signal.lfilter(self.butter_b, self.butter_a, [datum], zi=temp_butter_z)
                channel[j] = filtered_sample[0]
                
                if j == self.shift_samples - 1:
                    self.butter_z[i] = temp_butter_z
                
        return arr

    def get_name(self, channel_name, feature_name):
        return "{}_{}".format(channel_name, feature_name)
    
    def compute_feature(self, data, channel_names, feature_name, to_df=True):
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
            if output should be converfeatures(ted to a dataframe
    
        Returns
        -------
        df_result : pd.DataFrame (to_df = True) or dictionary
            new dataframe with feature columns for each channel
        
        """
        fn = globals()[feature_name]
        # computed_features = []
        new_channel_names = []
        computed_features = [[fn(ch) for ch in data]]
        computed_features = np.array(computed_features).T
        result = {}
        if computed_features.ndim > 2:
            for i in range(computed_features.shape[0]):
                feat = computed_features[i]
                for channel_name, actual_feat in zip(channel_names, feat):
                    new_name = self.get_name(channel_name, feature_name) + "_" + str(i)
                    result[new_name] = actual_feat
                    new_channel_names.append(new_name)
        else:
            new_channel_names = [self.get_name(channel_name,feature_name) for channel_name in channel_names]
            result = {self.get_name(channel_name, feature_name): feature
                                  for channel_name, feature in zip(channel_names, computed_features)}
        if to_df:
            result = pd.DataFrame(result)
        return result,new_channel_names
    
    def compute_features(self, data, channel_names, feature_names, mutate=False):
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
        all_ch_names = []
        for feature_name in feature_names:
            result,new_channel_names = self.compute_feature(data, channel_names, feature_name, to_df=False)
            all_results.update(result)
            all_ch_names = all_ch_names + new_channel_names
        
        return all_results,all_ch_names
    
    
    def predict_function(self, arr):
        # assume already filtered
        """
        arr_filtered = np.zeros((arr.shape))
        for ch in range(arr.shape[1]):
            arr_filtered[:,ch]= filter_signal(arr[:,ch])
        """
        filtered_arr = self.apply_filter(arr)
        res, _ = self.compute_features(filtered_arr, self.channel_names, self.features)
        input_arr = np.array(list(res.values()))
        return self.clf.predict_proba(np.squeeze(input_arr).reshape(1, -1))
    
   
    