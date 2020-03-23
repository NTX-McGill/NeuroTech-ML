import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy import signal
from constants import LABEL_MAP, HAND_MAP, HAND_FINGER_MAP, MODE_MAP, SAMPLING_FREQ
import json
import os
import re
import pickle

# not the same as the one in train.py
def sample_baseline(df, method='max', drop_rest=False, baseline_sample_factor=1, seed=7):
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
    
    fingers = df['finger']
    
    # take the maximum of all existing classes (excluding NaN), then multiply by the sample factor
    # if this maximum exceeds the number of NaN rows, uses all NaN rows
    if method == 'max':
        n_baseline_samples = min(len(df[np.logical_not(df['finger'].notnull())]), np.max(fingers.value_counts())) * baseline_sample_factor
    
    # other option: take the mean instead
    elif method == 'mean':
        n_baseline_samples = int( fingers.count()/fingers.nunique() ) * baseline_sample_factor
    
    else:
        raise ValueError('Invalid method: {}. Accepted methods are \'max\' and \'mean\''.format(method))
        
    
    baseline_samples = df[np.logical_not(df['finger'].notnull())].sample(n=n_baseline_samples, replace=False, random_state=seed)
    df.loc[baseline_samples.index, ['finger']] = 0
    
    # drop all rows where 'finger' is NaN
    if drop_rest:
        df = df[df['finger'].notnull()]
        df.reset_index(drop=True, inplace=True)
    
    return df

# copied from real_time filter.py
def test_filter(windows, fs=250, order=2, low=5.0, high=50.0):
    result = []
    nyq = fs / 2
    bb, ba = signal.butter(order, [low/nyq, high/nyq], 'bandpass')
    bz = signal.lfilter_zi(bb, ba)

    notch_freq = 60.0
    bp_stop = notch_freq + 3.0 * np.array([-1,1])
    nb, na = signal.iirnotch(notch_freq, notch_freq / 6, fs)
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

def notch_filter(freq=60.0, fs=250):
    """
        Design notch filter. Outputs numerator and denominator polynomials of iir filter.
        inputs:
            freq  (float)
            order (int)
            fs    (int)
        outputs:
            (ndarray), (ndarray)
    """
    return signal.iirnotch(freq, freq / 6, fs=fs)
    
def butter_filter(low=5.0, high=50.0, order=4, fs=250):
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

def filter_signal(arr, notch=True, filter_type='original_filter', start_of_overlap=25):
    """
        Apply butterworth (and optionally notch) filter to a signal. Outputs the filtered signal.
        inputs:
            arr   (ndarray)
            notch (boolean)
            filter_type (string)
        outputs:
            (ndarray)
    """
    
    bb, ba = butter_filter()
    nb, na = notch_filter() 
    
    if filter_type=='original_filter':
        if notch:
            arr = signal.lfilter(nb, na, arr)
        
        return signal.lfilter(bb, ba, arr)
    
    elif filter_type=='real_time_filter':        
        #First index at which two subsequent windows overlap, same shift as 
        
        #Initial conditions of filters
        nz = signal.lfilter_zi(nb, na)
        bz = signal.lfilter_zi(bb, ba)
        
        #Filter each window sample-by-sample
        results = []
        for window in arr:
            #Initialize filtered window
            w = np.zeros(len(window))
            
            #Set intial conditions to those of the start of the window
            temp_nz, temp_bz = nz, bz
            
            #Notch filter
            for i, datum in enumerate(window):
                #signal.lfilter returns a list, so we save to a temp list to avoid a list of lists
                filtered_sample, temp_nz = signal.lfilter(nb, na, [datum], zi=temp_nz)
                w[i] = (filtered_sample[0]) 
                
                #Save initial condition for next window
                if i == start_of_overlap - 1: nz = temp_nz
            
            #Bandpass filter
            for i, datum in enumerate(w):
                filtered_sample, temp_bz = signal.lfilter(bb, ba, [datum], zi=temp_bz)
                w[i] = filtered_sample[0]
                
                #Save intial condition for next window
                if i == start_of_overlap - 1: bz = temp_bz
                
            results.append(w)
            
        return results
        
    else:
        print('\nfilter type not recognised, enter valid filter type!')
        raise Exception
        

def filter_dataframe(df,filter_type='original_filter', start_of_overlap=25):
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
            filtered_df[col] = filter_signal(np.array(df[col]), filter_type=filter_type, start_of_overlap=start_of_overlap)
        
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

    if type(input_val) == str:
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

def create_windows(data, length=1, shift=0.1, offset=2, take_everything=False, filter_type='real_time_filter', drop_rest=True, sample=True):
    """
        Combines data points from data into labeled windows
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
    ch_ind = [i for i in range(len(data.columns)) if 'channel' in data.columns[i]]
    
    #Only focus on the part of the emg data between start and end
    emg = data.iloc[start - offset:end + offset, ch_ind]
    labels = data.loc[start - offset: end + 1 + offset, label_col]
    
    #Create and label windows
    windows = []
    window_labels = []
    for i in range(0, emg.shape[0], shift):
        #Handle windowing the data
        #Want w to be a list of ndarrays representing channels, 
        #so w = np.array(emg[i: i+length, :]) doesn't work (it gives array with shape (250, 8))
        w = []
        for j in range(emg.shape[1]):
            channel = np.array(emg.iloc[i:i+length, j])
            w.append(channel)
        
        #Only use windows with enough data points
        if len(w[0]) != length: continue
        
        windows.append(w)
    
        #Get all not-null labels in the window (if any) and choose which one to use for the window
        w_labels = labels[i: i + length][labels[i: i + length].notnull()]
        window_labels.append(get_window_label(w_labels, i, length))
    
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
    
    #All the windows have the same id and mode as labeled data
    windows_df['id'] = pd.Series(np.full(len(windows), data['id'][0]))
    windows_df['mode'] = pd.Series(np.full(len(windows), data['mode'][0]))
    
    #Real-time filter dataframe
    # filtered_windows_df = windows_df
    if filter_type == 'real_time_filter':
        windows_df = filter_dataframe(windows_df, filter_type=filter_type, start_of_overlap=shift)
    
    # add finger=0 for random subset of baseline samples
    if sample:
        windows_df = sample_baseline(windows_df, drop_rest=drop_rest)
    
    return windows_df

def create_dataset(directory, channels, filter_type='original_filter', file_regex='*.txt'):
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
        data = load_data(files[i+1], files[i], channels)
        
        # Filter data
        if filter_type == 'original_filter':
            data = filter_dataframe(data,filter_type=filter_type)
        
        # Window data
        w = create_windows(data, filter_type=filter_type)
        
        #Add data/windows to larger dataframe 
        big_data = big_data.append(data) 
        windows = windows.append(w)
        
        print("Adding windows with shape:", str(w.shape) + ". Current total size:", str(windows.shape))
        print("Adding data with shape:", str(data.shape) + ". Current total size:", str(big_data.shape))
    
    #Reindex datarames before returning
    big_data.reset_index(drop=True, inplace=True)
    windows.reset_index(drop=True, inplace=True)
    
    return big_data, windows

def select_files(path_data, dates=None, subjects=None, modes=None):
    """
    Selects data files according to specifications.
    Specifically, keeps only files in the intersection of requested dates, subjects and modes

    Parameters
    ----------
    path_data : string
        Path to data directory.
    dates : list of requested dates as strings in 'YYYY-MM-DD' format, optional
        If None, no filtering is done for the dates. The default is None.
    subjects : list of requested subject IDs as strings in 'XXX' format, optional
        If None, no filtering is done for the subjects. The default is None.
    modes : list of requested modes as integers or single digit strings, optional
        If None, no filtering is done for the modes. The default is None.

    Returns
    -------
    selected_files : list of (file_data, file_log) tuples (of filenames)

    """
    
    r_date = '\d{4}-\d{2}-\d{2}'
    r_subject = '\d{3}'
    
    # input validation: check that requested dates/subjects are formatted correctly
    invalid_dates = []
    if dates:
        invalid_dates = [d for d in dates if not re.fullmatch(r_date, d)]
    invalid_subjects = []
    if subjects:
        invalid_subjects = [s for s in subjects if not re.fullmatch(r_subject, s)]
        
    # input validation: check that modes are formatted correctly (can be int or numerical string)
    invalid_modes = []
    if modes:
        for i in range(len(modes)):
            try:
                modes[i] = int(modes[i])
                
                if modes[i] < 1 or modes[i] > len(MODE_MAP.keys()):
                    invalid_modes.append(modes[i])
            except ValueError:
                invalid_modes.append(modes[i])
            
    # raise exception if invalid input
    if invalid_dates:
        raise ValueError('Invalid date(s): {}. Must be a list of strings in \'YYYY-MM-DD\' format.'.format(invalid_dates))
    if invalid_subjects:
        raise ValueError('Invalid subject ID(s): {}. Must be a list of strings in \'XXX\' format (three digits).'.format(invalid_subjects))
    if invalid_modes:
        raise ValueError('Invalid mode(s): {}. Available modes are the following: {}.'.format(
            invalid_modes, {v:k for k,v in MODE_MAP.items()}))
        
    # convert req_subjects into list of strings (ex: '001' -> 1) because of the way get_metadata() works
    if subjects:
        subjects_int = [int(s) for s in subjects]
    
    # get all available dates
    dates_all = [f for f in os.listdir(path_data) if re.fullmatch(r_date, f)]
    
    # remove 2020-02-09 because log file uses old data format from CLI tool
    try:dates_all.remove('2020-02-09')
    except:pass
    
    # keep only requested dates
    if dates:
        dates_all = [d for d in dates_all if d in dates]
    
    # get all data files
    files_all = []
    for date in dates_all:
        files = glob(os.path.join(path_data, date, '*.txt'))
        files_all.extend(files)
            
    # separate files into lists of datafiles and logfiles
    files_data = []
    files_log = []
    for (i, file) in enumerate(files_all):
        try:
            get_metadata(file)      # datafiles don't have JSON header so will raise JSONDecodeError (ValueError)
            files_log.append(file)
            
        except ValueError:
            files_data.append(file)
            
    # sort files so that files in same position contain data from same trial
    # (this assumes consistent file naming)
    files_data.sort()
    files_log.sort()
            
    # make sure that separation makes sense
    if not (len(files_data) == len(files_log)):
        raise Exception('Number of data files ({}) and number of log files ({}) do not match'.format(len(files_data), len(files_log)))
    
    # filter files by requested subjects/modes
    selected_files = []
    for i in range(len(files_log)):
        
        metadata = get_metadata(files_log[i])
        to_add = True
        
        if subjects and not metadata['id'] in subjects_int:
            to_add = False
        if modes and not MODE_MAP[metadata['mode']] in modes:
            to_add = False
        
        if to_add:
            selected_files.append((files_data[i], files_log[i]))
            
    # message
    print('Selected {} trials with these specifications:\n'.format(len(selected_files)) +
          '\tdates: {}\n'.format(dates if dates else 'all') + 
          '\tsubjects: {}\n'.format(subjects if subjects else 'all') + 
          '\tmodes: {}'.format(modes if modes else 'all'))
    
    return selected_files

def get_aggregated_windows(path_data, channels=[1,2,3,4,5,6,7,8], 
                           dates=None, subjects=None, modes=None, 
                           save=False, path_out='.', filter_type='real_time_filter'):
    """
    Selects trials based on dates/subjects/modes, 
    then creates windows and aggregates them together.
    Optionally saves windows in pickle file.

    Parameters
    ----------
    path_data : string
        Path to data folder.
    channels : list of integers, optional
        Channels to include in windows. The default is [1,2,3,4,5,6,7,8].
    dates, subjects, modes : parameters passed to select_files()
    path_out : string, optional
        DESCRIPTION. The default is '.'.
    save : boolean, optional
        If True, will save windows as a pickle file in location given by path_out. The default is False.

    Returns
    -------
    windows_all : pandas.DataFrame
        DataFrame with one row per window. Contains one column for each channel, 
        and also 'hand', 'finger', 'keypressed', 'id', and 'mode'

    """
    
    # get relevant data/log files
    selected_files = select_files(path_data, dates=dates, subjects=subjects, modes=modes)
    
    # make empty dataframe where windows from each file will be appended
    windows_all = pd.DataFrame()
    
    # for each trial
    for (file_data, file_log) in selected_files:
        try:
            # add windows
            print('\nAdding windows for trial with following files:\n' + 
              '\tdata: {}\n'.format(file_data) + 
              '\tlog: {}'.format(file_log))
            
            data = load_data(file_data, file_log, channels)
            windows = create_windows(data)# returns filtered windows
            # filtered_windows = test_filter(windows,shift=1.0)
            
            windows_all = windows_all.append(windows)
        except ValueError as e:
            print('An error occured while adding the windows from this file')
            print(e)
            print('moving on to the next one...\n')
        
    # save windows as pickle file
    if save:
        
        # generate filename based on requested dates/subjects/modes
        to_add = []
        for (i, l) in enumerate((dates, subjects, modes)):
            if l:
                to_add.append('_'.join(map(str, l)))
            else:
                to_add.append('all')
        filename = 'windows_date_{}_subject_{}_mode_{}.pkl'.format(
            to_add[0], to_add[1], to_add[2])
        
        # get full path to output file
        filename = os.path.join(path_out, filename)
        
        # write pickle file
        with open(filename, 'wb') as f_out:
            pickle.dump(windows_all, f_out)
            print('Saved windows to file {}'.format(filename))
        
    return windows_all

if __name__ == '__main__':
    #Testing code
    # channels = [1,2,3,4,5,6,7,8]
    
    # markers = '../data/2020-02-23/002-trial1-both-guided-2020-02-23-18-16-45-254.txt'
    # fname = '../data/2020-02-23/002-trial1-both-guided-OpenBCI-RAW-2020-02-23_18-14-32.txt'
    # test = load_data(fname, markers, channels)
    # out = create_windows(test)
    
    path_data = '../data'
    # w = get_aggregated_windows(path_data, modes=[1,2])
    w = get_aggregated_windows(path_data, modes=[3], save=True, path_out='windows')
    
    # directory = '../data/2020-02-23/'
    # labeled_raw, good_windows = create_dataset(directory, channels)    
#     windows.to_csv('windows-2020-02-23.csv', index=False)
    # good_windows.to_pickle('windows-2020-02-23_not_real_time.pkl')

#    w = pd.read_pickle('windows-2020-02-23.pkl')