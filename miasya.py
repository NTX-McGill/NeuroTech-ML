# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:49:41 2020

@author: miasya
"""

"""

****QUICK USAGE NOTE***
- You can play with everything within the "CHANGE HERE" box
- Beware - The axes aren't consistent!
- Ask me if you need any explanations!

"""


"""

# DONE:
# take data windows of 1s (use rolands code) but with no excess info
# Featurize
# put in model, get prediction ie most likely keypress/finger (if >0.5)
# plot predictions as markings on the stacked signal windows
# add actual labels (number and letter)

# TO DO:
# Match the colours of finger label to channel
# standardize the axes and general increase in readability

"""

from real_time_class import RealTimeML
# import from siggy directory
from siggy.match_labels import append_labels, filter_dataframe, label_window
import numpy as np
import matplotlib.pyplot as plt

channels = [1,2,3,4,5,6,7,8,13]

##############################################
###### YOU CAN CHANGE FROM HERE ##############
##############################################

length = 250
shift = 0.1 * length

channel_names = ['channel {}'.format(i) for i in channels[:-1]]
feature_names = ['mav']

data_file = 'data/2020-03-08/012_trial2_self-directed_OpenBCI-RAW-2020-03-08_19-02-54.txt'
label_file = 'data/2020-03-08/012_trial2_self-directed_2020-03-08-19-06-09-042.txt'

#%%
res = append_labels(data_file, label_file, channels)
data = filter_dataframe(res)
windows = label_window(data, offset=0, take_everything=True)

# for now we make windows into a numpy array cause it's not liking the df
windows = windows[channel_names].iloc[:1000]
  
windows_fixed = windows.to_numpy()
#%% 
ML = RealTimeML()
all_predictions = []
for win in windows_fixed:   
    all_predictions.append(ML.predict_function(win))
        
# do a bunch of plots 
for start_sec in [i*5 for i in range(1, 6)]:
    start = length * start_sec
    end = start + length*5
    #%%
    # now we have the predictions, and we need to plot it against all the channels
    fig = plt.subplots(figsize=(20,15),sharex=True)
    
    emg = data.to_numpy()
    
    # for each hand plot spahetti line plot
    for ch in range(0,4):
        plt.subplot(5,1,1)
        plt.plot(emg[start:end,ch])
    plt.title('hand one')
        
    for ch in range(4,8):
        plt.subplot(5,1,2)
        plt.plot(emg[start:end,ch])
    plt.title('hand two')
    
    #%%
    # pred val holds whichever index is greatest and has a probability > 0.5
    pred_vals = np.zeros((int((emg.shape[0]-length)/shift)+1,10))
    
    # This gets the timestamps we need for the eventplot later
    time = 0
    time_index = 0
    for pred in all_predictions:
        index = np.argmax(pred[0])
        if (pred[0][index] > 0.5):
            pred_vals[time_index][index] = time
        time += shift
        time_index += 1
        
    #%%
    # on last subplot show events for each classification
    # almost a rainbow?
    color = ['r','DarkOrange', 'y', 'g', 'b','c','m','k','Crimson', 'SpringGreen']
    
    # fit pred_vals to window that we're displaying
    fitted_pred_vals = pred_vals[int(start/shift):int(end/shift)];
    fitted_pred_vals[fitted_pred_vals != 0] -= start;
    
    # Now make eventplot for predicted values
    plt.subplot(5,1,3)
    plt.title('predictions every window - hand one')
    for clas in range(0,5):
        plt.eventplot(fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0], colors=color[clas])
    
        # now put the labels, markers is marker times, so the x position?
        markers = fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0]
        
        # and put markers of uniform colour above event lines
        for m in markers:
            plt.text(int(m),1,str(clas))
            
    plt.subplot(5,1,4)
    plt.title('predictions every window - hand two')     
    for clas in range(5,10):
    
        plt.eventplot(fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0], colors=color[clas])
    
        # now put the labels, markers is marker times, so the x position?
        markers = fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0]
        
        # and put markers of uniform colour above event lines
        for m in markers:
            plt.text(int(m),1,str(clas))
    #%%
    # Now add actual labels over the predicted
            
    # Keep only actual logged values
    labels = res[['timestamp(ms)','finger','keypressed']].dropna()
    
    # Only keep ones in the plotting ranges
    labels['timestamp(ms)'] = labels['timestamp(ms)'] - labels.iloc[0,0]
    fitted_labels = labels[labels['timestamp(ms)'] >= start]
    fitted_labels = fitted_labels[fitted_labels['timestamp(ms)'] <= end]
    fitted_labels['timestamp(ms)'] = fitted_labels['timestamp(ms)'] - start
    
    #%%
    
    # Now plot actual labels
    plt.subplot(5,1,5)
    plt.title('actual keypresses')
    
    for row in range(fitted_labels.shape[0]):
        # plot spike
        plt.eventplot([fitted_labels.iloc[row,0]], colors=color[row])
        
        # plot text (finger number AND character)
        plt.text(fitted_labels.iloc[row,0],1,fitted_labels.iloc[row,1])
        plt.text(fitted_labels.iloc[row,0],1.5,fitted_labels.iloc[row,2])
    
    #%%
            
