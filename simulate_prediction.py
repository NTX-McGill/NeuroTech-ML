#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:00:46 2020

@author: marley
"""

from real_time_class import RealTimeML
from siggy.match_labels import load_data, filter_dataframe, create_windows, select_files
import numpy as np
import matplotlib.pyplot as plt

length = 250
shift = 0.1
plot_length = 5 # in seconds

channel_names = ['channel {}'.format(i) for i in range(1,9)]


model_file= 'model_windows_date_all_subject_all_mode_1_2-03_18_2020_22_33_39.pkl'

# take some data file
data_file, label_file = select_files('data', dates=['2020-03-03'], modes=[1])[0]


raw_data = load_data(data_file, label_file)
raw_data = filter_dataframe(raw_data, filter_type='original_filter')


windows = create_windows(raw_data, shift=shift, offset=0, take_everything=True, drop_rest=False)
        
label_col = 'keypressed' if any(raw_data['keypressed'].notnull()) else 'finger'
start = np.where(raw_data[label_col].notnull())[0][0]
end = np.where(raw_data[label_col].notnull())[0][-1]
data = raw_data.iloc[start:end]
data = data[channel_names].to_numpy()

# make sure the data is aligned
n_windows = 4
for i in range(n_windows):
    plt.subplot(2,2, i+1)
    s = int(i*shift*250)
    plt.plot(data[s:s+250, 0], label='original')
    plt.plot(windows['channel 1'].iloc[i], label='windowed')
    plt.legend()

windows_fixed = windows[channel_names].to_numpy()

ML = RealTimeML(model_filename=model_file)
all_predictions = []
for win in windows_fixed:   
    all_predictions.append(ML.predict_function(win))
    
predictions = np.squeeze(np.array(all_predictions))
    
windows[np.logical_not(windows[label_col].notnull())] = 0
labels = windows['finger'].to_numpy().astype(int)
labels_onehot = np.zeros((labels.size, labels.max()+1))
labels_onehot[np.arange(labels.size),labels] = 1
#%% 
for i in range(1,min([20, len(windows)*shift/plot_length])):
    plt.figure(figsize=(24,18))
    s, e = np.array([i, i+1]) * (plot_length *250)
    signal_segment = data[s:e]
    # for each hand plot spahetti line plot
    ax1 = plt.subplot(5,1,1)
    for ch in range(0,4):
        ax1.plot(signal_segment[:,ch])
        ax1.set_xlim([0,len(signal_segment)])
        ax1.set_ylim([signal_segment.min(), signal_segment.max()])
    ax1.set_title('hand one')
        
    ax2 = plt.subplot(5,1,2)
    for ch in range(4,8):
        ax2.plot(signal_segment[:,ch])
        ax2.set_xlim([0,len(signal_segment)])
        ax2.set_ylim([signal_segment.min(), signal_segment.max()])
    ax2.set_title('hand two')
    
    plt.subplot(5,1,3)
    s, e = np.array([i, i+1]) * int(plot_length/shift)
    segment = predictions[s:e]
    plt.imshow(segment.T, cmap=plt.cm.Blues, aspect='auto')
    
    
    plt.subplot(5,1,4)
    segment = labels_onehot[s:e]
    plt.imshow(segment.T, cmap=plt.cm.Blues, aspect='auto')
