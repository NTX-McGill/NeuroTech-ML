#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz
from glob import glob
from scipy import signal


# In[2]:


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


# In[10]:


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


# In[4]:


def append_labels_deprecated(data_file, labels_file, channels):
    """
        Only works for Feb 09 dataset, no longer useful
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


# In[12]:


def append_labels(data_file, label_file, channels):
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
    labels = pd.read_csv(label_file, 
                         skiprows= 9,
                         sep=", ", 
                         names=['timestamp(datetime)', 'timestamp(ms)', 'type', 'hand', 'finger', 'keypressed'], 
                         header=None, 
                         engine='python')
    
    hand_legend = {'left': 1, 'right': 2}
    finger_legend = {'thumb': 1, 'index finger': 2, 'middle finger': 3, 'ring finger': 4, 'pinkie': 5}
    
    #Get useful columns
    emg = data[:, :-1]
    data_timestamps = data[:, -1]
    label_timestamps = labels['timestamp(ms)']
    
    #Get labels
    hands = labels['hand']
    fingers = labels['finger']
    keys = labels['keypressed']
    
    #Map labels to a column of data
    #Initialize label Series to NaN
    hand_labels = pd.Series(np.full(len(emg), np.NaN))
    finger_labels = pd.Series(np.full(len(emg), np.NaN))
    key_labels = pd.Series(np.full(len(emg), np.NaN))
    
    #Append each label to nearest timestamp in data
    for i in range(len(label_timestamps)):
        ind = closest_time(data_timestamps, label_timestamps[i])
        
        #If there are keystrokes, no need for hand/finger labels
        if any(keys.notnull()):
            key_labels[ind] = keys[i]
        else:
            hand_labels[ind], finger_labels[ind], key_labels[ind] = hand_legend[hands[i]], finger_legend[fingers[i][:-1]], keys[i]
    
    #Put everything into a DataFrame
    names = ["channel " + str(i) for i in range(1, data.shape[1])] + ['timestamp(ms)']
    labelled_data = pd.DataFrame(data, columns=names)
    labelled_data['hand'] = hand_labels
    labelled_data['finger'] = finger_labels
    labelled_data["keypressed"] = key_labels
    
    return labelled_data


# In[26]:


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
    
    def append_labels(have_keys):
        pass
    
    SAMPLING_FREQ = 250
    
    #Convert arguments
    length, shift, offset = int(length*SAMPLING_FREQ), int(shift*SAMPLING_FREQ), int(offset*SAMPLING_FREQ)
    
    #Get data of interest
    have_keys = True if any(data['keypressed'].notnull()) else False
    
    if have_keys:
        start = np.where(data['keypressed'].notnull())[0][0]
        end = np.where(data['keypressed'].notnull())[0][-1]
    else:
        start = np.where(data['hand'].notnull())[0][0]
        end = np.where(data['hand'].notnull())[0][-1]
    
    emg = data.iloc[start - offset:end + offset, :-4]
    
    #Create windows with labels
    windows = []
    window_hand_labels, window_finger_labels, window_key_labels = [], [], []
    for i in range(0, emg.shape[0], shift):
        #Handle windowing the data
        w = []
        for j in range(emg.shape[1]):
            channel = emg.iloc[i:i+length, j]
            w.append(np.array(channel))
            
        full_window = True if len(w[0]) == length else False
        
        if full_window:
            windows.append(w)

            #Handle the labels of the windows

            hands, fingers, keys = [], [], []
            if have_keys:
                key_labels = data.iloc[start - offset:end + offset, -1]
                w_key_labels = key_labels[i:i+length][key_labels[i:i+length].notnull()]

                if len(w_key_labels) == 0: #No keys pressed
                    window_key_labels.append(np.NaN)
                elif len(w_key_labels) == 1: #One key pressed
                    window_key_labels.append(w_key_labels.iloc[0]) 
                else: #If more than one keypressed in window, take closest one to middle of window
                    indices = np.array([np.where(key_labels.iloc[i:i+length] == l)[0][0] for l in w_key_labels])
                    mid_ind = (2*i + length)//2
                    window_key_labels.append(w_key_labels.iloc[np.argmin(np.abs(indices - mid_ind))])
            else:
                hand_labels = data.iloc[start - offset:end + offset, -3]
                finger_labels = data.iloc[start - offset:end + offset, -2]
                w_hand_labels = hand_labels[i:i+length][hand_labels[i:i+length].notnull()]
                w_finger_labels = finger_labels[i:i+length][finger_labels[i:i+length].notnull()]

                if len(w_hand_labels) == 0: #No keys pressed
                    window_hand_labels.append(np.NaN)
                    window_finger_labels.append(np.NaN)
                elif len(w_hand_labels) == 1: #One key pressed
                    window_hand_labels.append(w_hand_labels.iloc[0]) 
                    window_finger_labels.append(w_finger_labels.iloc[0])
                else: #If more than one keypressed in window, take closest one to middle of window
                    indices = np.array([np.where(hand_labels.iloc[i:i+length] == l)[0][0] for l in w_hand_labels])
                    mid_ind = (2*i + length)//2
                    window_hand_labels.append(w_hand_labels.iloc[np.argmin(np.abs(indices - mid_ind))])
                    window_finger_labels.append(w_hand_labels.iloc[np.argmin(np.abs(indices - mid_ind))])
    
    #Put everything into a DataFrame
    names = ["channel " + str(i) for i in range(1, data.shape[1]-3)]
    windows_df = pd.DataFrame(windows, columns=names)
    
    if have_keys:
        windows_df['hand'] = pd.Series(np.full(len(windows), np.NaN))
        windows_df['finger'] = pd.Series(np.full(len(windows), np.NaN))
        windows_df['keypressed'] = pd.Series(window_key_labels)
    else:
        windows_df['hand'] = pd.Series(window_hand_labels)
        windows_df['finger'] = pd.Series(window_finger_labels)
        windows_df['keypressed'] = pd.Series(np.full(len(windows), np.NaN))
    
    return windows_df


# In[8]:


def merge_data(directory, filter_data=True):
    """
    Combines all datasets in 'directory'
    inputs:
      directory (string)
    """
    
    SAMPLING_FREQ = 250
    def notch_mains_interference(data):
        notch_freq_Hz = np.array([60.0])  # main + harmonic frequencies
        for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
            bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])  # set the stop band
            b, a = signal.butter(3, bp_stop_Hz / (SAMPLING_FREQ / 2.0), 'bandstop')
            arr = signal.lfilter(b, a, data, axis=0)
            print("Notch filter removing: " +
                  str(bp_stop_Hz[0]) +
                  "-" +
                  str(bp_stop_Hz[1]) +
                  " Hz")
        return arr

    def filter_signal(arr, lowcut=5.0, highcut=120.0, order=4, notch=True):
        if notch:
            arr = notch_mains_interference(arr)
        nyq = 0.5 * SAMPLING_FREQ
        b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return signal.lfilter(b, a, arr)

    channels = [1,2,3,4,13]
    files = sorted(glob(directory + '/*.txt'))

    out = pd.DataFrame()
    windows = pd.DataFrame()
    for i in range(0, len(files), 2):
        print("Appending trial with labels:", files[i])
        data = append_labels(files[i+1], files[i], channels)
        
        if filter_data:
            filtered_data = pd.DataFrame()
            for j in channels[:-1]:
                filtered_data['channel ' + str(j)] = filter_signal(np.array(data.iloc[:,j-1]))
            filtered_data['timestamp(ms)'] = data['timestamp(ms)']
            filtered_data['hand'] = data['hand']
            filtered_data['finger'] = data['finger']
            filtered_data['keypressed'] = data['keypressed']
            
            out = out.append(filtered_data)
            w = label_window(filtered_data)
            windows = windows.append(w)
        else:
            out = out.append(data)
            w = label_window(data)
            windows = windows.append(w)
            
        print("Adding windows with shape:", str(w.shape) + ". Current total size:", str(windows.shape))
        print("Adding data with shape:", str(data.shape) + ". Current total size:", str(out.shape))
        
    return out, windows


# In[27]:


if __name__ == '__main__':
#     markers = '../data/2020-02-16/001_trial2_right_air_2020-02-16-19-15-12-263.txt'
#     fname = '../data/2020-02-16/001_trial2_right_air_OpenBCI-RAW-2020-02-16_19-14-29.txt'
#     channels = [1,2,3,4,13]

#     labelled_raw = append_labels(fname, markers, channels)
#     out = label_window(labelled_raw)
    directory = '../data/2020-02-16/'
    labelled_raw, windows = merge_data(directory)


# In[165]:


# windows.to_csv('windows-2020-02-23.csv', index=False)
# windows.to_pickle('windows-2020-02-23.pkl')

