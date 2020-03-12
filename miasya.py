# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:49:41 2020

@author: miasya
"""

"""

- take data windows of 1s (use rolands code) but with no excess info
- Featurize
- put it through model, get prediction
- get most likely keypress/finger (if >0.5)
- plot it as markings on the stacked signal windows
- make the visual good
# look into eventplot matplotlib
# add actual
# all channels 1 graph
# two hands
# show actual letter
# show number that corresponds to that letter
# standardize the axes

# match the colours in hand plots, also separate finger into 2 subplots?

"""

from real_time_class import RealTimeML
from match_labels import append_labels, filter_dataframe, label_window
import numpy as np
import matplotlib.pyplot as plt

channels = [1,2,3,4,5,6,7,8]
    
#%%
data_file = 'miasya_data_test.txt'
    
raw_data = np.loadtxt(data_file,
                      delimiter=',',
                      skiprows=7,
                      usecols=channels)
#%%
data = np.zeros((raw_data.shape))

for ch in range(raw_data.shape[1]):
     data[:,ch] = filter_signal(raw_data[:,ch])

#%%
length = 250 # 1.00 seconds
shift = 50  # 0.20 seconds
windows = []

# for all possible windows
for i in range(0, data.shape[0]-length, shift):
    #Handle windowing the data
    w = np.zeros((data.shape[1], length))
    
    # for all channels
    for j in range(data.shape[1]):
        w[j] = data[i:i+length, j]
            
    windows.append(w)
    
#%%
# Windows is a list, where every entry of the list is a 250 by 8 numpy ndarry
# that represents the 250 data points for each of the 8 channels
# The time of the occurance is related to the index of within the windows list
# do the math lol where shift is 0.2s

# Fix plots later
#x = np.linspace(0,250)
#plt.plot(x,windows[0][0,:])

# Now to featurize each of the windoes individually

# Convert to pandas dataframe

#feature_names = ['mav']
#channel_names = ['channel {}'.format(i) for i in channels]
    
rt = RealTimeML()

all_predictions = []
for win in windows:   
    all_predictions.append(rt.predict_function(win))
    
    
#%%
# now we have the predictions, and we need to plot it against all the channels
plt.clf()
fig = plt.subplots(figsize=(20,15))

# arbitrary start and end points for graphing
start = 10000
end = start + 1000 

# for each hand plot spaghetti line plot
for ch in range(0,4):
    plt.subplot(3,1,1)
    plt.plot(data[start:end,ch])
plt.title('hand one')
    
for ch in range(4,8):
    plt.subplot(3,1,2)
    plt.plot(data[start:end,ch])
plt.title('hand two')

#%%
# pred val holds whichever index is greatest and has a probability > 0.5
pred_vals = np.zeros((int((data.shape[0]-length)/shift)+1,10))

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
color = ['b','g','r','c','m','y','k','Crimson', 'DarkOrange', 'SpringGreen']
plt.subplot(3,1,3)

# fit pred_vals to window that we're displaying
fitted_pred_vals = pred_vals[int(start/shift):int(end/shift)];
fitted_pred_vals[fitted_pred_vals != 0] -= start;
#%%
for clas in range(pred_vals.shape[1]):
    plt.eventplot(fitted_pred_vals[:,clas], colors=color[clas])

    # now put the labels, markers is marker times, so the x position?
    markers = fitted_pred_vals[:,clas][fitted_pred_vals[:,clas] != 0]
    
    for m in markers:
        plt.text(int(m),1,str(clas))
    
#%%
# now show actual marking
# i need this from rolands stuff
data_file = 'data/2020-03-08/012_trial2_self-directed_OpenBCI-RAW-2020-03-08_19-02-54.txt'
label_file = 'data/2020-03-08/012_trial2_self-directed_2020-03-08-19-06-09-042.txt'

res = append_labels(data_file, label_file, channels)
#%%
labels = res[['timestamp(ms)','finger','keypressed']]
raw_data = res[['channel 1','channel 2','channel 3','channel 4','channel 5',
            'channel 6','channel 7','channel 8']]

#%%
data = filter_dataframe(res)
#%%
windows = label_window(data)
        
#plt.show()