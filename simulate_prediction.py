#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:00:46 2020

@author: marley
"""

from real_time_class import Prediction
from siggy.match_labels import load_data, filter_dataframe, create_windows, select_files
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

length = 250
shift = 0.1
plot_length = 30 # in seconds

channel_names = ['channel {}'.format(i) for i in range(1,9)]

model_file = 'model_windows_date_all_subject_all_mode_1_2_4_groups_ok_good.pkl'
ML = Prediction(model_filename=model_file)

# take all ok and good data files with mode 4 because that's in air
selected_files = select_files(path_data='data', path_trials_json='siggy',  modes=[4], trial_groups=['ok', 'good'])

#%%
for data_file, label_file in selected_files:
    """
    for every file get the emg data, window it, and generate predictions 
    so in the next loop we can plot them against the actual keypress value
    """
    
    # load data
    raw_data = load_data(data_file, label_file)
    raw_data = filter_dataframe(raw_data, filter_type='original_filter')
    
    # create windows
    windows = create_windows(raw_data, shift=shift, offset=0, take_everything=True, drop_rest=False)
    
    # make numpy array with desired data
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
    
    # Get all prediction vector for every window
    all_predictions = []
    for win in windows_fixed:   
        all_predictions.append(ML.predict_function(win))
        
    predictions = np.squeeze(np.array(all_predictions))
    
    # One hot actual finger vector only has 1 value that is 1, rest are 0
    windows[np.logical_not(windows[label_col].notnull())] = 0
    labels = windows['finger'].to_numpy().astype(int)
    labels_onehot = np.zeros((labels.size, labels.max()+1))
    labels_onehot[np.arange(labels.size),labels] = 1
    
    #for i in range(1,min([20, len(windows)*shift/plot_length])):
    #for entire spread
    for i in range(1,int(len(windows)*shift/plot_length)-1):
        """
        show multiple consecutive segments of 1 file. 4 stacked subplots
        """
        
        # start and end for plotting
        s, e = np.array([i, i+1]) * (plot_length *250)
        signal_segment = data[s:e]
        
        if (len(signal_segment) < 10):
            continue
        
        print("from", str((start+s)/length))
        print("to", str((start+e)/length))
        
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(24,18))
        fig.suptitle("mode 4 model ok+good from index {} to {} ie time {} to {}".format(str(start+s), str(start+e), str((start+s)/length), str((start+e)/length)))
        
        # for each hand plot spahetti line plot
        #ax1 = plt.subplot(4,1,1)
        for ch in range(0,4):
            ax1.plot(signal_segment[:,ch])
            ax1.set_xlim([0,len(signal_segment)])
            ax1.set_ylim([signal_segment.min(), signal_segment.max()])
        ax1.set_title('hand one')
            
        #ax2 = plt.subplot(4,1,2)
        for ch in range(4,8):
            ax2.plot(signal_segment[:,ch])
            ax2.set_xlim([0,len(signal_segment)])
            ax2.set_ylim([signal_segment.min(), signal_segment.max()])
        ax2.set_title('hand two')
        
        # prediction plot heatmap
        #ax3 = plt.subplot(4,1,3)
        s2, e2 = np.array([i, i+1]) * int(plot_length/shift)
        segment = predictions[s2:e2]
        ax3.imshow(segment.T, cmap=plt.cm.Blues, aspect='auto')
        ax3.set_title('finger predictions')
        
        # actual value plot heatmap
        #ax4 = plt.subplot(4,1,4)
        segment = labels_onehot[s2:e2]
        ax4.imshow(segment.T, cmap=plt.cm.Blues, aspect='auto')
        ax4.set_title('actual finger')
        
        # save plots to subfolder
        plt.savefig(os.path.join('sim_pred_images_30s', '{}_times_{}_{}.jpg'.format(data_file, str((start+s)/length), str((start+e)/length))))
        plt.close() # to save memory