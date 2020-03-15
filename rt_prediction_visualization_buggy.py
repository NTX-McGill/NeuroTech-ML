# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:49:41 2020

@author: miasya
"""

"""
****QUICK USAGE NOTE***
- You can play with everything within the "CHANGE HERE" box
- Ask me if you need any explanations!
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
# Change filtering to be window-wise to be consistent with the real time model
# When everything works, make the code beautiful
"""

from real_time_class import RealTimeML
# import from siggy directory
from siggy.match_labels import append_labels, filter_dataframe, label_window
import numpy as np
import matplotlib.pyplot as plt

channels = [1,2,3,4,5,6,7,8,13]

##############################################
###### YOU CAN CHANGE FROM HERE ##############

length = 250
shift = 0.5 * length

channel_names = ['channel {}'.format(i) for i in channels[:-1]]
#feature_names = ['iemg', 'mav', 'mmav', 'mmav2', 'var','rms', 'zc', 'wamp', 
#                 'wl', 'ssch', 'wfl']

# Keeping it simple just so we can fix plots
feature_names = ['wl']


# When I try early data I get a list index out of range error in append_labels
# but these ones seem to work
# also note that you need to use self directed or guided trials in order to see
# any actual keypresses
data_file = 'data/2020-03-08/012_trial2_self-directed_OpenBCI-RAW-2020-03-08_19-02-54.txt'
label_file = 'data/2020-03-08/012_trial2_self-directed_2020-03-08-19-06-09-042.txt'

# TO HERE ####################################
#%%
res = append_labels(data_file, label_file, channels)
data = filter_dataframe(res)        # LATER MAKE THIS USING FILTER BY WINDOW

# take everything means don't cut off the empty stuff at the start
# shift means have a window start every 0.5 seconds
# offset is like if we want to trim margins from start and end of the entire trace
windows = label_window(data, shift=shift, offset=0, take_everything=True)

# for now we make windows into a numpy array cause it's not liking the df
# We also only take the first 10000 to speed it up
windows = windows[channel_names].iloc[:10000]
windows_fixed = windows.to_numpy()
#%% 
ML = RealTimeML()
all_predictions = []
for win in windows_fixed:   
    all_predictions.append(ML.predict_function(win))
        

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
for start_sec in [i*20 for i in range(8, 12)]:
    start = length * start_sec
    end = start + length*20
    #%%
    # now we have the predictions, need to plot it against all the channels
    fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(20,15),sharex=True)

    emg = data.to_numpy()
    
    # for each hand plot spahetti line plot
    ax1 = plt.subplot(5,1,1)
    for ch in range(0,4):
        ax1.plot(emg[start:end,ch])
    ax1.set_title('hand one')
        
    ax2 = plt.subplot(5,1,2, sharex=ax1, sharey=ax1)
    for ch in range(4,8):
        ax2.plot(emg[start:end,ch])
    ax2.set_title('hand two')
    
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
    color = ['r','DarkOrange','y','g','b','c','m','k','Crimson','SpringGreen']
    
    # fit pred_vals to window that we're displaying
    fitted_pred_vals = pred_vals[int(start/shift):int(end/shift)];
    fitted_pred_vals[fitted_pred_vals != 0] -= start;
    
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
            
            
    # thinking
    """
    So in data aka emg, each 250 index is 1 second
    250 i = 1 s
    in labels, the timestamps need to be divided by 1000 to be in seconds
    let j be in the value in the timestamp(ms)
    1 j = 1000 ms => 1/1000 * j = 1 ms
    1000 ms = 1 s 
    But then the times need to
    1000 ms = 250 i
    1000/250  * ms = i
    (1/1000) * (1000/250) j = i => 1/250
    So really we just need to divide it by 250
    
    """
            
    ## Make numpy for graphing like we did for emg
    # we do start/length*1000 to get the time in milliseconds
    # which SHOULD correpond to timestamp but doesn't cause life sux
    fitted_labels = labels
    
    #fitted_labels['timestamp'] = fitted_labels['timestamp(ms)'] / 1000
    #fitted_labels.drop(columns='timestamp(ms)')
    
    print('start in ms is', start / length * 1000)
    print('end in ms is', (end / length * 1000))
    
    fitted_labels = fitted_labels[fitted_labels['timestamp(ms)'] >= (start / length * 1000)]
    fitted_labels = fitted_labels[fitted_labels['timestamp(ms)'] <= (end / length * 1000)]
    
    # make the index line up
    
    # see this is where everything goes wrong sadness
    fitted_labels['timestamp(ms)'] -= (start / length * 1000) 
    fitted_labels['timestamp(ms)'] *= 1000
    fitted_labels['timestamp(ms)'] /= 250
    
    

    fitted_labels = fitted_labels.to_numpy()    
    # see you just need to ask yourself why the hell you didn't make
    # the horizontal axis the time in seconds to begin with
    
    #%%
    
    # Now plot actual labels
    ax5 = plt.subplot(5,1,5,sharex=ax1)
    ax5.set_title('actual keypresses')

    #ax5.eventplot(fitted_labels[:,0], colors=color[fitted_labels[:,1]-1])
    # ignore colours for now
    ax5.eventplot(fitted_labels[:,0])
    
    """
                                             
    # and put markers of uniform colour above event lines
    for m, time in enumerate(fitted_labels[:,0]):
        #plot text (finger number AND character)
        ax5.text(time,1,fitted_labels[m,1])
        ax5.text(time,1.5,fitted_labels[m,2])
        
    """
    plt.show()
    #%%
            
