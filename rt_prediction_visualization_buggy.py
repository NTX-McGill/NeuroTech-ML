# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:49:41 2020

@author: miasya
"""

"""
****QUICK USAGE NOTE***
- This produces several graphs of the same style but at different points
- You can play with everything within the "CHANGE HERE" box
"""


"""
# DONE:
# take data windows of 1s (use rolands code) but with no excess info
# Featurize
# put in model, get prediction ie most likely keypress/finger (if >0.5)
# plot predictions as markings on the stacked signal windows
# add actual labels (number and letter)
# standardize the axes and general increase in readability

# TO DO:
# Match the colours of finger label to channel
# When everything works, make the code beautiful
"""

from real_time_class import RealTimeML
# import from siggy directory
from siggy.match_labels import load_data, filter_dataframe, create_windows
import numpy as np
import matplotlib.pyplot as plt

channels = [1,2,3,4,5,6,7,8,13]

###### YOU CAN CHANGE FROM HERE ##############
length = 250
shift = 0.2

channel_names = ['channel {}'.format(i) for i in channels[:-1]]
feature_names = ['iemg', 'mav', 'mmav', 'mmav2', 'var','rms', 'zc', 'wamp', 
                 'wl', 'ssch', 'wfl']

# When I try early data I get a list index out of range error in append_labels
# but these ones seem to work
# also note that you need to use self directed or guided trials in order to see
# any actual keypresses
data_file = 'data/2020-03-08/012_trial2_self-directed_OpenBCI-RAW-2020-03-08_19-02-54.txt'
label_file = 'data/2020-03-08/012_trial2_self-directed_2020-03-08-19-06-09-042.txt'

# TO HERE ####################################
#%%
res = load_data(data_file, label_file, channels)
data = filter_dataframe(res)        # LATER MAKE THIS USING FILTER BY WINDOW

# take everything means don't cut off the empty stuff at the start
# shift means have a window start every 0.5 seconds
# offset is like if we want to trim margins from start and end of the entire trace
windows = create_windows(data, shift=shift, offset=0, take_everything=True)

# for now we make windows into a numpy array cause it's not liking the df
# We also only take the first 10000 to speed it up
#windows = windows[channel_names].iloc[:10000]
windows = windows[channel_names]
windows_fixed = windows.to_numpy()
#%% 
ML = RealTimeML()
all_predictions = []
for win in windows_fixed:   
    all_predictions.append(ML.predict_function(win))
        
 #%%
# pred val holds whichever index is greatest and has a probability > 0.5
# more specifically it's has shape (# windows, # classes)
# so for every row, it is a 10D array, where each index is represents a class
# and in that array, 9/10 or 10/10 of those values should be zero
# because it will be non-zero if that class was active
# and the value stored will be the modified timestamp
# which basically is the time in seconds * 250 (to make the emg data which
# is in 1/250ths of a second)
# I have the data this way because it helps with colour coding
pred_vals = np.zeros((windows_fixed.shape[0],10))

# This gets the timestamps we need for the eventplot later
modified_timestamp = 0
for i, pred in enumerate(all_predictions):
    class_index = np.argmax(pred[0])
    if (pred[0,class_index] > 0.5):
        pred_vals[i,class_index] = modified_timestamp
    modified_timestamp += shift * length
        
        
# Now normalize the amplitude
# This helps avoid more subtle signals from being completely overwhelmed
# by higher amplitude signals during graphing
# -- later, to do this in real time, we need to estimate the max and min
for ch in channel_names:
    max_amp = data[ch].max()
    min_amp = data[ch].min()
    data[ch] = (data[ch] - min_amp) / (max_amp - min_amp)

# make times in terms of first timestamp and keep only actual logged values
labels = res[['timestamp(ms)','finger','keypressed']].dropna()
labels['timestamp(ms)'] = labels['timestamp(ms)'] - data['timestamp(ms)'].iloc[0]

# do several plots of the same style but at different time intervals 
# Here we go for traces of length 20s
for start_sec in [i*10 for i in range(20, 26)]:
    start = length * start_sec
    end = start + (length * 10)
    #%%
    # now we have the predictions, need to plot it against all the channels
    fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(24,18),sharex=True)

    emg = data.to_numpy()
    
    x = np.linspace(start,end,num=(end-start)) # x is based on index of emg plot
    
    # for each hand plot spahetti line plot
    ax1 = plt.subplot(5,1,1)
    for ch in range(0,4):
        ax1.plot(x,emg[start:end,ch])
    ax1.set_title('hand one')
        
    ax2 = plt.subplot(5,1,2, sharex=ax1, sharey=ax1)
    for ch in range(4,8):
        ax2.plot(x,emg[start:end,ch])
    ax2.set_title('hand two')
    
   
        
    #%%
    # on last subplot show events for each classification
    # almost a rainbow?
    color = ['r','DarkOrange','y','g','b','c','m','k','Crimson','SpringGreen']
    
    # fit pred_vals to window that we're displaying
    # the start/shift should get the right index in the pred_vals
    fitted_pred_vals = pred_vals[int(start/(shift*length)):int(end/(shift*length)),:];
    
    # Now make eventplot for predicted values
    # RIGHT FLUSH MEANS THAT THE PREDICTION IS PLACED AT THE END OF THE WINDOW
    ax3 = plt.subplot(5,1,3,sharex=ax1)
    ax3.set_title('predictions every window - hand one - note:right flush')
    for clas in range(0,5):
        ax3.eventplot(fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0],
                      colors=color[clas])
    
        # now put the labels, markers is marker times, so the x position
        markers = fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0]
        
        # and put markers of uniform colour above event lines
        for m in markers:
            ax3.text(int(m),1,str(clas))
            
    ax4 = plt.subplot(5,1,4,sharex=ax1)
    ax4.set_title('predictions every window - hand two - note:right flush')     
    for clas in range(5,10):
    
        ax4.eventplot(fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0],
                      colors=color[clas])
    
        # now put the labels, markers is marker times, so the x position?
        markers = fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0]
        
        # and put markers of uniform colour above event lines
        for m in markers:
            ax4.text(int(m),1,str(clas))
    #%%
    # Now add actual labels over the predicted
        
    print('start in ms is', start / length * 1000)
    print('end in ms is', end / length * 1000)
    
    ## Make numpy for graphing like we did for emg
    fitted_labels = labels
    # we do start/length*1000 to get the time in milliseconds
    fitted_labels = fitted_labels[fitted_labels['timestamp(ms)'] >= (start / length * 1000)]
    fitted_labels = fitted_labels[fitted_labels['timestamp(ms)'] <= (end / length * 1000)]
    
    # make the index line up
    fitted_labels['timestamp(ms)'] /= 1000 # to seconds
    fitted_labels['timestamp(ms)'] *= 250 # 1s = 250 points, 
    fitted_labels = fitted_labels.to_numpy()    

    #%%
    
    # Now plot actual labels
    ax5 = plt.subplot(5,1,5,sharex=ax1)
    ax5.set_title('actual keypresses')

    ax5.eventplot(fitted_labels[:,0])# add colour coding later
    
    # and put markers of uniform colour above event lines
    for m, time in enumerate(fitted_labels[:,0]):
        #plot text (finger number AND character)
        ax5.text(time,1,str(fitted_labels[m,1]))
        ax5.text(time,1.5,str(fitted_labels[m,2]))
        
    plt.show()
    #%%
            
