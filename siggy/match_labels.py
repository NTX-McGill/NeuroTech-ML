import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy import signal
from constants import LABEL_MAP, HAND_MAP, HAND_FINGER_MAP, MODE_MAP, SAMPLING_FREQ
import json


# copied from real_time filter.py
def test_filter(windows, fs=250, order=2, low=20, high=120):
    result = []
    nyq = fs / 2
    bb, ba = signal.butter(order, [low/nyq, high/nyq], 'bandpass')
    bz = signal.lfilter_zi(bb, ba)

    notch_freq = 60.0
    bp_stop = notch_freq + 3.0 * np.array([-1,1])
    nb, na = signal.iirnotch(notch_freq, notch_freq / 10, fs)
    nz = signal.lfilter_zi(nb, na)

    for w in windows:
        w, nz = signal.lfilter(nb, na, w, zi=nz)
        w, bz = signal.lfilter(bb, ba, w, zi=bz)
        result.append(w)
    
    return result


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

def filter_signal(arr, notch=True, filter_type='original_filter'):
    """
        Apply butterworth (and optionally notch) filter to a signal. Outputs the filtered signal.
        inputs:
            arr   (ndarray)
            notch (boolean)
            filter_type (string)
        outputs:
            (ndarray)
    """
    if filter_type=='original_filter':
        if notch:
            nb, na = notch_filter()
            arr = signal.lfilter(nb, na, arr)
        
        bb, ba = butter_filter()
        return signal.lfilter(bb, ba, arr)
    
    elif filter_type=='real_time_filter':
        fs,order,low,high,notch_freq = 250,2,5,120,60.0
        nyq = fs / 2
        bb, ba = signal.butter(order, [low/nyq , high/nyq], 'bandpass')
        bz = signal.lfilter_zi(bb,ba)
        
        # bp_stop = notch_freq + s3.0*np.array([-1,1])
        nb, na = signal.iirnotch(notch_freq, notch_freq / 10, fs)
        nz = signal.lfilter_zi(nb,na)
        
        filtered_signal, nz = signal.lfilter(nb, na, arr, zi=nz)
        filtered_signal, bz = signal.lfilter(bb, ba, filtered_signal, zi=bz)
        return filtered_signal
        
    else:
        print('\nfilter type not recognised, enter valid filter type!')
        raise Exception


def filter_dataframe(df,filter_type='original_filter'):
    """
        Filters the signals in a dataframe.
        inputs:
            df          (DataFrame)
            filter_type (String)
        outputs:
            filtered_df (DataFrame)
    """
    filtered_df = df.copy()
    
    for col in df.columns:
        if 'channel' in col:
            filtered_df[col] = filter_signal(np.array(df[col]),filter_type=filter_type)
        
    return filtered_df

def get_metadata(label_file):
    """
    Converts the JSON at the beginning of input to a dict, and converts id/prompts values to int and list, respectively.
    
    Parameters
    ----------
    label_file : str
        Name of file containing metadata for data trial

    Returns
    -------
    meta : dict
        The metadata about the data associated to the input
    """
    
    #As a convention the first 9 lines of each labels file is a JSON containing metadata about the trial
    #Read the metadata lines and convert them into a dict
    with open(label_file) as f:
        meta_string = ''.join(f.readlines()[:9])
        meta = json.loads(meta_string)
    
    #Convert subject id from STR to INT
    meta['id'] = int(meta['id']) 
    
    #Convert prompts from STR to LIST
    meta['prompts'] = [s.strip() for s in meta['prompts'].split(',')]
    
    return meta

def init_labelled_df(data, names):
    """
    Pre-allocates DataFrame for labelled datapoints, values are by default set to np.NaN.

    Parameters
    ----------
    data : ndarray
        Numpy ndarry of values read from data file.
    names : list
        List of columns names for output DataFrame.

    Returns
    -------
    labelled_data : DataFrame
        Pandas DataFrame with shape (# of data points, # of channels + # of labels).
    """
    
    labelled_data = pd.DataFrame(data=np.full((len(data), len(names)), np.NaN), columns=names)
    labelled_data.iloc[:, :9] = data
    
    return labelled_data
    
def to_hand(inp):
    """
    Computes (uninteresting, cheap math trick) encoding for hand used to press key, given which key was pressed.

    Parameters
    ----------
    inp : str, int, or np.NaN
        Key that was pressed.

    Returns
    -------
    int
        Encoding of hand that pressed key.

    """

    if type(inp) == str:
        return (LABEL_MAP[inp] - 1) // 5 + 1
    elif type(inp) == int:
        return (inp - 1) // 5 + 1
    elif np.isnan(inp):
        return inp
    else:
        print("Unhandled type:", type(inp))
        raise Exception

def append_labels(data_file, label_file, channels=[1,2,3,4,5,6,7,8]):
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
                      usecols=channels + [13])
    labels = pd.read_csv(label_file, 
                         skiprows= 10,
                         sep=", ", 
                         names=['timestamp(datetime)', 'timestamp(ms)', 'type', 'hand', 'finger', 'keypressed'], 
                         header=None, 
                         engine='python')
    
    #Get useful metadata
    meta = get_metadata(label_file)
    subject_id, trial_mode = meta['id'], MODE_MAP[meta['mode']]
    
    #Get timestamps and labels
    data_timestamps = data[:, -1]
    label_timestamps = labels['timestamp(ms)']
    hands = labels['hand']
    fingers = labels['finger']
    keys = labels['keypressed']
    
    #Pre-allocate new DataFrame
    names = ['channel {}'.format(i) for i in channels] + ['timestamp(ms)', 'hand', 'finger', 'keypressed', 'id', 'mode']
    labelled_data = init_labelled_df(data, names)
    
    #Initialize values for id, mode
    labelled_data['id'][:] = subject_id
    labelled_data['mode'][:] = trial_mode
    
    #Append each label to nearest timestamp in data
    for i in range(len(label_timestamps)):
        ind = closest_time(data_timestamps, label_timestamps[i])
        
        #If there are key presses, ignore "prompt_end" lines, otherwise only use "prompt_end" lines
        #... prompt_end, left, index finger,   <-- Example of labels in "prompt_end" line
        #... keystroke, , , k                  <-- Example of labels in non-"prompt_end" line
        if any(keys.notnull()):
            if keys[i]: 
                    labelled_data['hand'][ind] = to_hand(keys[i])
                    labelled_data['finger'][ind] = LABEL_MAP[keys[i]]
                    labelled_data['keypressed'][ind] = keys[i]
        else:
            labelled_data['hand'][ind] = HAND_MAP[hands[i]]
            labelled_data['finger'][ind] =  HAND_FINGER_MAP[hands[i]][fingers[i][:-1]]

    return labelled_data

def get_window_label(labels, win_start, win_len):
    """
    Gets the label in closest to the middle index of a window, returns np.NaN if there are no events in the window

    Parameters
    ----------
    labels : Series
        Labels of the events in the window (either finger or key press label).
    win_labels : Series
        DESCRIPTION.
    win_start : int
        Starting index of the window.
    win_len : int
        Length of the window.

    Returns
    -------
    int (if labels if a list of ints ) or str (if labels is a list of strings) <- I know this is bad practice, will change
        DESCRIPTION.

    """
    
    if len(labels) == 0:
        return np.NaN
    else:
        mid_ind = (2*win_start + win_len)//2
        indices = np.array(labels.index)
        
        return labels.iloc[np.argmin(np.abs(indices - mid_ind))]

def label_window(data, length=1, shift=0.1, offset=2, take_everything=False):
    """
        Combines data points from data into labelled windows
        inputs:
            data    (DataFrame)indices = np.array([np.where(key_labels.iloc[i:i+length] == l)[0][0] for l in w_key_labels])
            length  (int)
            shift   (int)
            offset  (int)
        output:
            windows (DataFrame) 
    """

    #Convert arguments
    length, shift, offset = int(length*SAMPLING_FREQ), int(shift*SAMPLING_FREQ), int(offset*SAMPLING_FREQ)
    
    #If any key press labels are not null, label windows using key presses - otherwise label using fingers
    label_col = 'keypressed' if any(data['keypressed'].notnull()) else 'finger'

    #Find starting index
    if take_everything:
        start = 0
        end = len(data)
    else:
        start = np.where(data[label_col].notnull())[0][0]
        end = np.where(data[label_col].notnull())[0][-1]
    
    #Find how many channels are used in the dataframe by looking at the names of the columns
    ch_ind = []
    for i in range(len(data.columns)):
        if 'channel' in data.columns[i]: 
            ch_ind.append(i)
    
    #Only focus on the part of the emg data between start and end
    emg = data.iloc[start - offset:end + offset, ch_ind]
    
    #Create and label windows
    windows = []
    labels = data[label_col][start - offset: end + 1 + offset]
    window_labels = []
    for i in range(0, emg.shape[0], shift):
        #Handle windowing the data
        #Want w to be a list of ndarrays representing channels, 
        #so w = np.array(emg[i: i+length, :]) doesn't work (it gives array with shape (250, 8))
        w = []
        for j in range(emg.shape[1]):
            channel = emg.iloc[i:i+length, j]
            w.append(np.array(channel))
        
        #Only use windows with enough data points
        if len(w[0]) != length: continue
        
        #
        windows.append(w)
    
        #Get all not-null labels in the window (if any) and choose which one to use for the window
        w_labels = labels[i: i + length][labels[i: i + length].notnull()]
        window_labels.append(get_window_label(labels, w_labels, i, length))
    
    #Put everything into a DataFrame
    channel_names = [data.columns[i] for i in ch_ind]
    windows_df = pd.DataFrame(windows, columns=channel_names)
    
    window_labels_series = pd.Series(window_labels)
    if label_col == 'keypressed':
        windows_df['hand'] = window_labels_series.apply(to_hand)   #Map key presses to hand
        windows_df['finger'] = window_labels_series.map(LABEL_MAP) #Map key presses to fingers
        windows_df['keypressed'] = window_labels_series            #Keep labels as they are
    else:
        windows_df['hand'] = window_labels_series.apply(to_hand)            #Map finger to hand
        windows_df['finger'] = window_labels_series                         #Kepp labels as they are
        windows_df['keypressed'] = pd.Series(np.full(len(windows), np.NaN)) #No key presses
    
    #All the windows have the same id and mode as labelled data
    windows_df['id'] = pd.Series(np.full(len(windows), data['id'][0]))
    windows_df['mode'] = pd.Series(np.full(len(windows), data['mode'][0]))
    
    return windows_df

def merge_data(directory, channels, filter_type='original_filter', file_regex='*.txt'):
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
    # Set up which files and channels to merge
    files = sorted(glob(directory + '/' + file_regex))

    # Merge dataframes from files
    big_data = pd.DataFrame()
    windows = pd.DataFrame()
    for i in range(0, len(files), 2):
        print("Appending trial with labels:", files[i])
        data = append_labels(files[i+1], files[i], channels)
        
        # Filter data
        data = filter_dataframe(data,filter_type=filter_type)
        
        # Window data
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

#Can still abstract pre-allocating and initilizing DataFrames, will do that later if time permitting

if __name__ == '__main__':
    #Testing code
    channels = [1,2,3,4,5,6,7,8]
    
    markers = '../data/2020-02-23/002-trial1-both-guided-2020-02-23-18-16-45-254.txt'
    fname = '../data/2020-02-23/002-trial1-both-guided-OpenBCI-RAW-2020-02-23_18-14-32.txt'
    test = append_labels(fname, markers, channels)
    out = label_window(test)
    
    # directory = '../data/2020-02-23/'
    # labelled_raw, good_windows = merge_data(directory, channels)    
#     windows.to_csv('windows-2020-02-23.csv', index=False)
    # good_windows.to_pickle('windows-2020-02-23_not_real_time.pkl')

#    w = pd.read_pickle('windows-2020-02-23.pkl')