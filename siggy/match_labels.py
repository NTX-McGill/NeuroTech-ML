import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy import signal

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

def notch_filter(freq=60.0, order=3, fs=250):
    """
        Design notch filter. Outputs numerator and denominator polynomials of iir filter.
        inputs:
            freq  (float)
            order (int)
            fs    (int)
        outputs:
            (ndarray), (ndarray)
    """
    nyq = fs / 2
    bp_stop_f = freq + 3.0 * np.array([-1,1])
    return signal.butter(order, bp_stop_f / nyq, 'bandstop')
    
def butter_filter(low=5.0, high=120.0, order=4, fs=250):
    """
        Design butterworth filter. Outputs numerator and denominator polynomials of iir filter.
        inputs:
            low   (float)
            high  (float)
            order (int)
            fs    (int)
        outputs:
            (ndarray), (ndarray)
    """
    nyq = fs / 2
    return signal.butter(order, [low / nyq, high / nyq], 'bandpass')

def filter_signal(arr, notch=True):
    """
        Apply butterworth (and optionally notch) filter to a signal. Outputs the filtered signal.
        inputs:
            arr   (ndarray)
            notch (boolean)
        outputs:
            (ndarray)
    """
    if notch:
        nb, na = notch_filter()
        arr = signal.lfilter(nb, na, arr)
        
    bb, ba = butter_filter()
    return signal.lfilter(bb, ba, arr)

def filter_dataframe(df):
    """
        Filters the signals in a dataframe.
        inputs:
            df          (DataFrame)
        outputs:
            filtered_df (DataFrame)
    """
    filtered_df = df.copy()
    
    for col in df.columns:
        if 'channel' in col:
            filtered_df[col] = filter_signal(np.array(df[col]))
        
    return filtered_df

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
    
    #Constants
    mode_legend = {'Guided': 1, 'Self-directed': 2, 'In-the-air': 3}    
    hand_legend = {'left': 1, 'right': 2}
    hand_finger_legend = {'left' : {'thumb': 6, 'index finger': 7, 'middle finger': 8, 'ring finger': 9, 'pinkie': 10},
                          'right': {'thumb': 1, 'index finger': 2, 'middle finger': 3, 'ring finger': 4, 'pinkie': 5}}
    LABEL_MAP = {'1':10, '2':9, '3':8, '4':7, '5':7, '6': 2, '7':2, '8':3, '9':4, '0': 5,
             'q':10, 'w':9, 'e':8, 'r':7, 't':7, 'y':2, 'u':2, 'i':3, 'o':4, 'p':5,
             'a':10, 's':9, 'd':8, 'f':7, 'g':7, 'h':2, 'j':2, 'k':3, 'l':4, ';':5,
             'z':10, 'x':9, 'c':8, 'v':7, 'b':7, 'n':2, 'm':2, ',':3, '.':4, '/':5,
             '[':5, ']':5, "'":5, '\\':5 , 'space': 1, 'Shift': 10, 'Backspace':5}
    
    #Load data from files
    data = np.loadtxt(data_file,
                      delimiter=',',
                      skiprows=7,
                      usecols=channels + [13])
    labels = pd.read_csv(label_file, 
                         skiprows= 10,
                         sep=", ", 
                         names=['timestamp(datetime)', 'timestamp(ms)', 'type', 'hand', 'finger', 'keypressed'], 
                         header=None, 
                         engine='python')
    
    #Parse metadata at start of file for patient id and mode of data collection
    with open(label_file) as f:
        meta = f.readlines()[1:8]
        idx = int(meta[0].split(':')[1].strip(' ",\n'))
        mode = mode_legend[meta[2].split(':')[1].strip(' ",\n')]
    
    #Get useful columns
    emg = data[:, :-1]
    data_timestamps = data[:, -1]
    label_timestamps = labels['timestamp(ms)']
    
    #Get labels
    hands = labels['hand']
    fingers = labels['finger']
    keys = labels['keypressed']
    
    #Map labels to a column of data
    #Initialize label Series to NaN, subject id, data collection mode
    hand_labels = pd.Series(np.full(len(emg), np.NaN))
    finger_labels = pd.Series(np.full(len(emg), np.NaN))
    key_labels = pd.Series(np.full(len(emg), np.NaN))
    id_labels = pd.Series(np.full(len(emg), idx))
    mode_labels = pd.Series(np.full(len(emg), mode))
    
    #Append each label to nearest timestamp in data
    for i in range(len(label_timestamps)):
        ind = closest_time(data_timestamps, label_timestamps[i])
        
        #If there are key presses, ignore "prompt_end" lines, otherwise only use "prompt_end" lines
        #... prompt_end, left, index finger,   <-- Example of labels in "prompt_end" line
        #... keystroke, , , k                  <-- Example of labels in non-"prompt_end" line
        if any(keys.notnull()):
            #Ignore "prompt_end" lines
            if keys[i]: 
                hand_labels[ind], finger_labels[ind], key_labels[ind] = (LABEL_MAP[keys[i]] - 1) // 5 + 1, LABEL_MAP[keys[i]], keys[i]    
        else:
            hand_labels[ind], finger_labels[ind] = hand_legend[hands[i]], hand_finger_legend[hands[i]][fingers[i][:-1]]
    
    #Put everything into a DataFrame
    names = ["channel " + str(i) for i in range(1, data.shape[1])] + ['timestamp(ms)']
    labelled_data = pd.DataFrame(data, columns=names)
    labelled_data['hand'] = hand_labels
    labelled_data['finger'] = finger_labels
    labelled_data['keypressed'] = key_labels
    labelled_data['id'] = id_labels
    labelled_data['mode'] = mode_labels
    
    return labelled_data

def label_window(data, length=1, shift=0.1, offset=2, take_everything=False):
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
    
    #Check whether to look for keypressed or hand/fingers as labels
    have_keys = True if any(data['keypressed'].notnull()) else False

    #Find starting index
    if have_keys:
        start = np.where(data['keypressed'].notnull())[0][0]
        end = np.where(data['keypressed'].notnull())[0][-1]
    else:
        start = np.where(data['hand'].notnull())[0][0]
        end = np.where(data['hand'].notnull())[0][-1]
    if take_everything:
        start = 0
        end = len(data)
    
    ch_ind = []
    for i in range(len(data.columns)):
        if 'channel' in data.columns[i]: ch_ind.append(i)
    
    emg = data.iloc[start - offset:end + offset, ch_ind]
    
    #Create windows with labels
    windows = []
    window_hand_labels, window_finger_labels, window_key_labels = [], [], []
    for i in range(0, emg.shape[0], shift):
        #Handle windowing the data
        w = []
        for j in range(emg.shape[1]):
            channel = emg.iloc[i:i+length, j]
            w.append(np.array(channel))
        
        #Only use windows with enough data points
        if len(w[0]) != length: continue
        
        windows.append(w)

        #Handle the labels of the windows
        if have_keys:
            key_labels = data['keypressed'][start - offset: end + offset]
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
            hand_labels = data['hand'][start - offset:end + offset]
            finger_labels = data['finger'][start - offset:end + offset]
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
    names = [data.columns[i] for i in ch_ind]
    windows_df = pd.DataFrame(windows, columns=names)
    
    if have_keys:
        windows_df['hand'] = pd.Series(np.full(len(windows), np.NaN))
        windows_df['finger'] = pd.Series(np.full(len(windows), np.NaN))
        windows_df['keypressed'] = pd.Series(window_key_labels)
    else:
        windows_df['hand'] = pd.Series(window_hand_labels)
        windows_df['finger'] = pd.Series(window_finger_labels)
        windows_df['keypressed'] = pd.Series(np.full(len(windows), np.NaN))
    
    windows_df['id'] = pd.Series(np.full(len(windows), data['id'][0]))
    windows_df['mode'] = pd.Series(np.full(len(windows), data['mode'][0]))
    
    return windows_df

def merge_data(directory, channels, filter_data=True, file_regex='*.txt'):
    """
    Combines all datasets in 'directory' into a single DataFrame.
    Optionally filters the data.
    inputs:
        directory   (string)
        filter_data (boolean)
     outputs:
        out         (DataFrame)
        windows     (DataFrame)
    """
    #Set up which files and channels to merge
    files = sorted(glob(directory + '/' + file_regex))

    #Merge dataframes from files
    big_data = pd.DataFrame()
    windows = pd.DataFrame()
    for i in range(0, len(files), 2):
        print("Appending trial with labels:", files[i])
        data = append_labels(files[i+1], files[i], channels)
        
        if data.empty:
            print("Not in air, skipping file!")
            continue
        
        #Filter data
        if filter_data: 
            data = filter_dataframe(data)
        
        #Window data
        w = label_window(data)
        
        #Add data/windows to larger dataframe
        big_data = big_data.append(data)
        windows = windows.append(w)
            
        print("Adding windows with shape:", str(w.shape) + ". Current total size:", str(windows.shape))
        print("Adding data with shape:", str(data.shape) + ". Current total size:", str(big_data.shape))
    
    #Reindex datarames before returning
    big_data.reset_index(inplace=True)
    windows.reset_index(inplace=True)
    
    return big_data, windows

if __name__ == '__main__':
    #Testing code
    channels = [1,2,3,4,5,6,7,8]
    
#    markers = '../data/2020-02-23/002-trial1-both-guided-2020-02-23-18-16-45-254.txt'
#    fname = '../data/2020-02-23/002-trial1-both-guided-OpenBCI-RAW-2020-02-23_18-14-32.txt'
#    test = append_labels(fname, markers, channels)
#    out = label_window(test)
    
    directory = '../data/2020-02-23/'
    labelled_raw, good_windows = merge_data(directory, channels)
#     windows.to_csv('windows-2020-02-23.csv', index=False)
#     windows.to_pickle('windows-2020-02-23.pkl')

#    w = pd.read_pickle('windows-2020-02-23.pkl')