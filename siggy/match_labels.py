#!/usr/bin/env python
# coding: utf-8

# In[438]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz


# In[439]:


def get_ms(str_time):
  """
    Convert timestamp in keyboard markings file to unix milliseconds
    inputs:
      str_time     (string)
    outputs:
      milliseconds (float)
  """

  date_time_str = '2020-02-09 ' + str_time
  date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S:%f')

  timezone = pytz.timezone('America/New_York')
  timezone_date_time_obj = timezone.localize(date_time_obj)
  return timezone_date_time_obj.timestamp() * 1000


# In[440]:


def closest_time(times, marker_time):
    """
        Get row index of data timestamp closest to marker_time 
        inputs:
          times       (ndarray)
          marker_time (float)
        outputs:
          row index   (int)
    """
    
    return np.argmin(np.abs(times - marker_time))


# In[2]:


def append_labels(data_file, labels_file, channels):
    """
        Append ASCII values of labels in keyboard markings file to nearest (in terms of time) 
        data point from data set
        inputs:
          data_file     (string)
          labels_file   (string)
        outputs:
          labelled_data (DataFrame)
    """
    
    #Load data from files
    data = np.loadtxt(data_file,
                      delimiter=',',
                      skiprows=7,
                      usecols=channels)
    labels = pd.read_csv(labels_file)
    
    #Get useful columns
    emg = data[:, :-1]
    data_timestamps = data[:, -1]
    label_timestamps = labels['timestamp(ms)']
    keyspressed = labels[' keypressed'] #Tell Software to fix the col name
    
    #Map keyspressed to a column of data
    new_col = pd.Series(np.full(len(emg), np.NaN))
    for i in range(len(label_timestamps)): 
        new_col[closest_time(data_timestamps, get_ms(label_timestamps[i]))] = keyspressed[i]
    
    #Put everything into a DataFrame
    names = ["channel " + str(i) for i in range(1, data.shape[1])] + ['timestamp(ms)']
    labelled_data = pd.DataFrame(data, columns=names)
    labelled_data["keyspressed"] = new_col
    
    return labelled_data


# In[446]:


def label_window(data, length=1, shift=0.1, offset=2):
    """
        Combines data points from data into labelled windows
        inputs:
            data    (DataFrame)
            length  (int)
            shift   (int)
            offset  (int)
        output:
            windows (DataFrame) 
    """
    SAMPLING_FREQ = 250
    
    #Convert arguments
    length, shift, offset = int(length*SAMPLING_FREQ), int(shift*SAMPLING_FREQ), int(offset*SAMPLING_FREQ)
    
    #Get data of interest  
    start = np.where(data['keyspressed'].notnull())[0][0]
    end = np.where(data['keyspressed'].notnull())[0][-1]

    emg = data.iloc[start - offset:end + offset, :-2]
    labels = data.iloc[start - offset:end + offset, -1]
    
    #Create windows with labels
    windows = []
    window_labels = []
    for i in range(0, emg.shape[0], shift):
        #Handle windowing the data
        w = []
        for j in range(emg.shape[1]):
            channel = emg.iloc[i:i+length, j]
            w.append(channel)
        windows.append(w)
        
        #Handle the labels of the windows
        w_labels = labels[i:i+length][labels[i:i+length].notnull()]
        
        if len(w_labels) == 0: #No keys pressed
            window_labels.append(np.NaN) 
        elif len(w_labels) == 1: #One key pressed
            window_labels.append(w_labels.iloc[0]) 
        else: #If more than one keypressed in window, take closest one to middle of window
            indices = np.array([np.where(labels.iloc[i:i+length] == l) for l in w_labels])
            mid_ind = (2*i + length)//2
            window_labels.append(w_labels.iloc[np.argmin(np.abs(indices - mid_ind))])
    
    #Put everything into a DataFrame
    names = ["channel " + str(i) for i in range(1, data.shape[1]-1)]
    windows_df = pd.DataFrame(windows, columns=names)
    windows_df['keyspressed'] = pd.Series(window_labels)
    
    return windows_df


# In[449]:


if __name__ == '__main__':
    markers = '../data/001_trial1_right_log_18-09-46-931825.txt'
    fname = '../data/001_trial1_right_OpenBCI-RAW-2020-02-09_17-59-22.txt'
    labelled_raw = append_labels(fname, markers, channel)
    label_window(labelled_raw)

