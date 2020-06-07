import numpy as np
import pandas as pd
from scipy import signal
from constants import LABEL_MAP, HAND_MAP, HAND_FINGER_MAP, MODE_MAP, SAMPLING_FREQ
import json

def get_power(arr):
    
    # compute PSD
    nperseg = arr.shape[-1]/2
    nfft = max(nperseg*2, 50)
    freqs, Pxx = signal.welch(arr, fs=SAMPLING_FREQ, window='hanning', nperseg=nperseg, nfft=nfft, scaling='density')
    
    # keep only values for frequencies in a certain range
    limits = [10, 50]
    indices = [i for i in range(len(freqs)) if (freqs[i] > limits[0] and freqs[i] < limits[1])]
    freqs = freqs[indices]
    Pxx = Pxx[:, indices]
    
    # mean PSD for each channel
    Pxx_mean = np.mean(Pxx, axis=-1)

    # return max mean channel PSD
    return np.max(Pxx_mean)

def split_window(w, size, overlap=0.5, axis=-1):
    
    size = int(size)
    
    if size >= w.shape[axis]:
        return [w]
    
    shift = int(overlap*size)
    subwindows = []
    
    for i in range(0, w.shape[axis], shift):
        subwindow = w[:, i:i+size]
        if subwindow.shape[axis] == size:
            subwindows.append(subwindow)
    
    return subwindows

def is_baseline(w, len_subwindow, method='power', threshold=5):
    
    functions = {'power':get_power}
    
    w = np.stack(w, axis=0)
    
    subwindows = split_window(w, len_subwindow)
    
    for subwindow in subwindows:
        if len(subwindow) == 0:
            continue
        if functions[method](subwindow) > threshold:
            return False
    
    return True

# not the same as the one in train.py
def sample_baseline(df, baseline_label=np.NaN, method='mean', drop_rest=False, baseline_sample_factor=1, seed=7):
    """
    Select a subset of the baseline: convert the selected rows' label from NaN to 0
    runs in-place

    Parameters
    ----------
    df : pd.DataFrame
    labels : list
    baseline_sample_factor : int
        default 1 -> same number of baseline samples as single class
        represents the amount to multiply the number of samples

    Returns
    -------
    Modified DataFrame

    """
        
    if baseline_label is np.NaN:
        df_baseline = df[np.logical_not(df['finger'].notnull())]
    elif baseline_label == 0:
        df_baseline = df.loc[df['finger'] == 0]
        df = df.loc[df['finger'] != 0] # remove baseline rows from DataFrame
    else:
        raise ValueError('Invalid argument for baseline_label: {}'.format(baseline_label))
    
    fingers = df['finger'].loc[df['finger'].notnull()]
    
    # take the maximum of all existing classes (excluding NaN and 0), then multiply by the sample factor
    # if this maximum exceeds the number of NaN rows, uses all NaN rows
    if method == 'max':
        n_baseline_samples = min(len(df_baseline), np.max(fingers.value_counts())) * baseline_sample_factor
    
    # other option: take the mean instead
    elif method == 'mean':
        n_baseline_samples = min( len(df_baseline), int( fingers.count()/fingers.nunique() ) ) * baseline_sample_factor
    
    # other option : take a fixed amount of baseline samples
    elif method == 'determined amount':
        n_baseline_samples = 5000
    
    # other option : take the entire baseline
    elif method == 'everything':
        n_baseline_samples = len(df_baseline)
    
    else:
        raise ValueError('Invalid method: {}. Accepted methods are \'max\' and \'mean\''.format(method))
        
    baseline_samples = df_baseline.sample(n=n_baseline_samples, replace=False, random_state=seed)
    if baseline_label != 0:
        df.loc[baseline_samples.index, ['finger']] = 0
    else:
        df = df.append(baseline_samples)
    
    # drop all rows where 'finger' is NaN
    if drop_rest:
        df = df[df['finger'].notnull()]
        df.reset_index(drop=True, inplace=True)
    
    return df
    
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

def init_labeled_df(data, names):
    """
    Pre-allocates DataFrame for labeled datapoints, values are by default set to np.NaN.

    Parameters
    ----------
    data : ndarray
        Numpy ndarry of values read from data file.
    names : list
        List of columns names for output DataFrame.

    Returns
    -------
    labeled_data : DataFrame
        Pandas DataFrame with shape (# of data points, # of channels + # of labels).
    """
    
    labeled_data = pd.DataFrame(data=np.full((len(data), len(names)), np.NaN), columns=names)
    labeled_data.iloc[:, :data.shape[1]] = data
    
    return labeled_data
    
def to_hand(input_val):
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

    if input_val == 'baseline':
        return np.NaN
    elif input_val == 0:
        return np.NaN
    elif type(input_val) == str:
        return (LABEL_MAP[input_val] - 1) // 5 + 1
    elif type(input_val) == int:
        return (input_val - 1) // 5 + 1
    elif np.isnan(input_val):
        return input_val
    elif type(input_val) == float:
        return (int(input_val) - 1) // 5 + 1
    else:
        print("Unhandled type:", type(input_val))
        raise Exception

def load_data(data_file, label_file, channels=[1,2,3,4,5,6,7,8]):
    """
        Append ASCII values of labels in keyboard markings file to nearest (in terms of time) 
        data point from data set
        inputs:
          data_file     (string)
          labels_file   (string)
        outputs:
          labeled_data (DataFrame)
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
    labeled_data = init_labeled_df(data, names)
    
    #Initialize values for id, mode
    labeled_data.loc[:, 'id'] = subject_id
    labeled_data.loc[:, 'mode'] = trial_mode
    
    #Append each label to nearest timestamp in data
    for i in range(len(label_timestamps)):
        ind = closest_time(data_timestamps, label_timestamps[i])
        
        #If there are key presses, ignore "prompt_end" lines, otherwise only use "prompt_end" lines
        #... prompt_end, left, index finger,   <-- Example of labels in "prompt_end" line
        #... keystroke, , , k                  <-- Example of labels in non-"prompt_end" line
        if any(keys.notnull()):
            if keys[i]: 
                    try:
                        labeled_data.loc[ind, 'hand'] = to_hand(keys[i])
                        labeled_data.loc[ind, 'finger'] = LABEL_MAP[keys[i]]
                        labeled_data.loc[ind, 'keypressed'] = keys[i]
                    except KeyError:
                        pass
        else:
            labeled_data.loc[ind, 'hand'] = HAND_MAP[hands[i]]
            labeled_data.loc[ind, 'finger'] =  HAND_FINGER_MAP[hands[i]][fingers[i][:-1]]

    return labeled_data

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