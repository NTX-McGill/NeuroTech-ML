#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:06:02 2020

@author: miasya

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from siggy.constants import LABEL_MAP

#SUBJECT_IDS = [1,2,3,4,5,6,7,8,9,10,11,12]
# temp cut out 1 and 10 because less keypress data for them
SUBJECT_IDS = [2,3,4,5,6,7,8,9,11,12]
SCALAR_FEATURES = ['iemg','mav','mmav','var', 'var_abs', 'rms', 'wl', 'zc','ssc', 'wamp']
channels = [1,2,3,4,5,6,7,8]

# list of lists, where for each feature, we store a list of strings of format channel #_feature_name
all_feat_channels = []
for feat in SCALAR_FEATURES:
    all_feat_channels.append(['channel {}_{}'.format(i,feat) for i in channels])

# Use pkl file
features_filename = 'features_2020-03-22_windows_date_all_subject_all_mode_1_2_4.pkl'
with open(features_filename, 'rb') as f:
    data = pickle.load(f)

#%%
# For each channel, we have the EMG signal, and several signal features
# We also have the columns 'hand', 'finger', 'keypressed', 'id', 'mode'
#data.columns.values

#%%
# I look at the distribution of keypresses and eliminate rows where we have fewer than 2000 datapoints
# On this dataset, this eliminates all punctuation as well as letters y,w,b,g,v,q,x,z
# which are not frequent letters in the English language to begin with.
# If we observe things on common letters, we can reasonably assume that the same things are reflected in less common letters
# I also visually checked that every keypressed has many different id, so that we have representation from dif subjects
counts = data['keypressed'].value_counts()
#data = data[data['keypressed'].isin(counts[counts >= 1000].index)]
# temp to speed up processing
data = data[data['keypressed'].isin(counts[counts >= 3000].index)]

#%%
grouped = data.groupby(['keypressed', 'id'])

#%%
n_bins = 20 # for the histogram
keypresses = data['keypressed'].unique() # all possible keys with sufficient data

# We go feature by feature, showing all channels for every type of keypress, for every different person
for f, feature in enumerate(SCALAR_FEATURES):
    for kp in keypresses:
        
        fig, axs = plt.subplots(len(SUBJECT_IDS), len(channels),figsize=(len(channels)*3,len(SUBJECT_IDS)*2), sharex=True, sharey=True)    
        fig.suptitle('feature: {}, keypress: {}'.format(feature, kp))

        # Each row of the subplot will be all 8 channels for a specific subject id
        n = 0
        for name, group in grouped:
            # Only look at the proper keypress and id
            if name[0] != kp or name[1] not in SUBJECT_IDS:
                continue
                
            # For each channel, histogram for the feature, normalized using weights
            for fc, feat_channel in enumerate(all_feat_channels[f]):
                
                # Tint the background grey if we don't have 100 samples in the desired channel
                # to make us aware of this, because normalization will otherwise overrepresent.
                # If we have enough samples, make sure to colour the hand properly
                if len(group[feat_channel]) < 100:
                    axs[n, fc].set_facecolor('DarkGray')
                elif LABEL_MAP[kp] > 5: # the activity in on the left hand (channels 5,6,7,8)
                    colored_ch = [5,6,7,8]
                else: # right hand
                    colored_ch = [1,2,3,4]
                    
                if fc+1 in colored_ch:
                    color = 'SpringGreen'
                else:
                    color = 'Tomato'
                    
                weights = np.ones_like(group[feat_channel])/float(len(group[feat_channel]))
                axs[n, fc].hist(group[feat_channel],bins=n_bins, weights=weights, color=color)
                axs[n, fc].set_title('id: {}, ch: {}'.format(str(name[1]), fc+1))         
            n += 1
            
        plt.savefig(os.path.join('histograms','feature_{}_keypress_{}.png'.format(feature, kp)))
        #plt.show()
        plt.close()

#%%

#TODO
# Colour the channels where activation is most likely to be seen, via Michelle's observations
# Eliminate empty rows (in case where insufficient subject info)

# background white for tons of data, darker for less data?
# also this is mode blind, which could screw everything up

# Just try to interpret these 50 graphs?
# ^^ ask what features more likely to differ, which more likely to be consistent? or will model detect this?
# ^^ ask what keys are more likely to differ? but we know its space, c, and b from data trials?


