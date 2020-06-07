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

from filtering import *
from windowing_helpers import *

def notch_filter(freq=60.0, fs=250, Q=60):
    return notch_filter_helper(freq, fs, Q)
    
def butter_filter(low=5.0, high=50.0, order=4, fs=250):
    return butter_filter_helper(low, high, order, fs)

def filter_signal(arr, notch=True, filter_type='original_filter', start_of_overlap=25):
    return filter_dataframe_helper(arr, notch, filter_type, start_of_overlap)
        
def filter_dataframe(df,filter_type='original_filter', start_of_overlap=25):
    return filter_dataframe_helper(df, filtter_type, start_of_overlap)

def create_windows(data, length=1, shift=0.1, offset=2, take_everything=False, 
                   filter_type='real_time_filter', drop_rest=True, sample=True,
                   baseline_sample_factor=1, method='mean', labelling_method='old'):
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
    
    #Put everything into a DataFrame
    channel_names = [data.columns[i] for i in ch_ind]
    windows_df = pd.DataFrame(windows, columns=channel_names)
    
    #Real-time filter dataframe
    if filter_type == 'real_time_filter':
        windows_df = filter_dataframe(windows_df, filter_type=filter_type, start_of_overlap=shift)
    else:
        print('Not filtering!')
        
    # label windows
    window_labels = []
    
    # margin for labels
    margin = max(0, (0.3 * SAMPLING_FREQ) - length) // 2
    margin = int(margin)
    # print(margin)
    
    for i_window, window in windows_df.iterrows():
        
        if labelling_method == 'old':
            
            baseline_label = np.NaN
            
            i = int(i_window * shift)
                
            #Get all not-null labels in a range around the window (if any)
            #and choose which one to use for the window
                
            w_labels = labels[i:i+length][labels[i:i+length].notnull()]
            window_labels.append(get_window_label(w_labels, i, length))
            
        else:
                
            baseline_label = 0
            
            # if window is baseline, label accordingly (0 or 'baseline')
            if is_baseline(window, 0.1*SAMPLING_FREQ, method=labelling_method):
                
                if label_col == 'finger':
                    window_labels.append(0)
                elif label_col == 'keypressed':
                    window_labels.append('baseline')
            
            # else find a label using 1s range centered around window
            else:
                
                i = int(i_window * shift)
                
                #Get all not-null labels in a range around the window (if any)
                #and choose which one to use for the window
                
                i_start = max(i - margin, 0)
                i_end = i + length + margin
                
                w_labels = labels[i_start:i_end][labels[i_start:i_end].notnull()]
                window_labels.append(get_window_label(w_labels, i, length))
    
    window_labels_series = pd.Series(window_labels)
    if label_col == 'keypressed':
        windows_df['hand'] = window_labels_series.apply(to_hand)   #Map key presses to hand
        windows_df['finger'] = window_labels_series.map(LABEL_MAP) #Map key presses to fingers
        windows_df['keypressed'] = window_labels_series            #Keep labels as they are
    else:
        windows_df['hand'] = window_labels_series.apply(to_hand)            #Map finger to hand
        windows_df['finger'] = window_labels_series                         #Keep labels as they are
        windows_df['keypressed'] = pd.Series(np.full(len(windows), np.NaN)) #No key presses
    
    #All the windows have the same id and mode as labeled data
    windows_df['id'] = pd.Series(np.full(len(windows), data['id'][0]))
    windows_df['mode'] = pd.Series(np.full(len(windows), data['mode'][0]))
    
    # add finger=0 for random subset of baseline samples
    if sample:
        windows_df = sample_baseline(windows_df, 
                                     baseline_label=baseline_label,   # either 0 or np.NaN
                                     drop_rest=drop_rest,
                                     baseline_sample_factor=baseline_sample_factor,
                                     method=method)
    
    return windows_df

def select_files(path_data, path_trials_json='.', dates=None, subjects=None, modes=None, trial_groups=None):
    """
    Selects data files according to specifications.
    Specifically, keeps only files in the intersection of requested dates, subjects and modes

    Parameters
    ----------
    path_data : string
        Path to data directory.
    path_trials_json: string
        Path to directory containing trials.json file
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
                
    # input validation: check that trial_groups has only accepted values
    valid_groups = ['bad', 'ok', 'good']
    invalid_groups = []
    if trial_groups:
        invalid_groups = [g for g in trial_groups if g not in valid_groups]
            
    # raise exception if invalid input
    if invalid_dates:
        raise ValueError('Invalid date(s): {}. Must be a list of strings in \'YYYY-MM-DD\' format.'.format(invalid_dates))
    if invalid_subjects:
        raise ValueError('Invalid subject ID(s): {}. Must be a list of strings in \'XXX\' format (three digits).'.format(invalid_subjects))
    if invalid_modes:
        raise ValueError('Invalid mode(s): {}. Available modes are the following: {}.'.format(
            invalid_modes, {v:k for k,v in MODE_MAP.items()}))
    if invalid_groups:
        raise ValueError('Invalid trial group(s): {}. Available trial groups are the following: {}'.format(
            invalid_groups, valid_groups))
        
    # convert req_subjects into list of strings (ex: '001' -> 1) because of the way get_metadata() works
    if subjects:
        subjects_int = [int(s) for s in subjects]
        
    # get lists of good/bad/ok trials
    if trial_groups:
        with open(os.path.join(path_trials_json, 'trials.json')) as file_json:
            trials = json.load(file_json)
            
        included_trials = []
        
        for trial_group in trials.keys():
            
            # add trial (full path)
            if trial_group in trial_groups:
                for trial_path in trials[trial_group]:
                    included_trials.append(os.path.join(path_data, trial_path[0], trial_path[1]))
                        
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
        if trial_groups and not files_data[i] in included_trials:
            to_add = False
        
        if to_add:
            selected_files.append((files_data[i], files_log[i]))
            
    # message
    print('Selected {} trials with these specifications:\n'.format(len(selected_files)) +
          '\tdates: {}\n'.format(dates if dates else 'all') + 
          '\tsubjects: {}\n'.format(subjects if subjects else 'all') + 
          '\tmodes: {}\n'.format(modes if modes else 'all') + 
          '\ttrial groups: {}'.format(trial_groups if trial_groups else 'all'))
    
    return selected_files

def get_aggregate_baseline_windows(path_data,channels=[1,2,3,4,5,6,7,8],
                         dates=None,subjects=None,modes=None,
                         save=False,path_out='.',filter_type='real_time_filter'):
    """
    Selects trials based on dates/subjects/modes, 
    then creates windows and aggregates them together, only selects baseline windows.
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
    # notes : hopefully the modes sholud not affect the baseline, perhaps investigate this
    
    # make empty dataframe where windows from each file will be appended
    baseline_all = pd.DataFrame()
    
    # for each trial 
    for (file_data, file_log) in selected_files:
        try:
            # add windows
            print('\nAdding windows for trial with following files:\n' + 
              '\tdata: {}\n'.format(file_data) + 
              '\tlog: {}'.format(file_log))
            data = load_data(file_data, file_log, channels)
            baseline = create_windows(data,method='everything')# the baseline sample factor determines how much of the baseline you sample
            
            # drop everthing that is not the baseline
            baseline = baseline[baseline['finger']==0]
            
            baseline_all = baseline_all.append(baseline)
        except ValueError as e:
            print('An error occured while adding the windows from this file')
            print(e)
            print('moving on to the next one...\n')
            
    # save baseline windows to a pickle file
    if save:
        # generate filename based on requested dates/subjects/modes
        to_add = []
        for (i,l) in enumerate((dates,subjects,modes)):
            if l:
                to_add.append('_'.join(map(str,l)))
            else:
                to_add.append('all')
        filename = 'baseline_windows_date_{}_subject_{}_mode_{}.pkl'.format(
            to_add[0],to_add[1],to_add[2])
        
        # get full path to output file
        filename = os.path.join(path_out,filename)
        
        # write pickle file
        with open(filename,'wb') as f_out:
            pickle.dump(baseline_all, f_out)
            print('Saved baseline windows to file {}'.format(filename))
            
    return baseline_all

def get_aggregated_windows(path_data, path_trials_json='.', channels=[1,2,3,4,5,6,7,8], 
                           dates=None, subjects=None, modes=None, trial_groups=None,
                           length=1, shift=0.1,
                           save=False, path_out='.', append='',
                           filter_type='real_time_filter',
                           method='mean',
                           labelling_method='old'):
    """
    Selects trials based on dates/subjects/modes, 
    then creates windows and aggregates them together.
    Optionally saves windows in pickle file.

    Parameters
    ----------
    path_data : string
        Path to data folder.
    path_trials_json : string
        Path to JSON file for good/bad/ok trials. The default is '.'.
    channels : list of integers, optional
        Channels to include in windows. The default is [1,2,3,4,5,6,7,8].
    dates, subjects, modes, trial_groups : parameters passed to select_files()
    length, shift: parameters passed to create_windows()
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
    selected_files = select_files(path_data, path_trials_json=path_trials_json, dates=dates, subjects=subjects, modes=modes, trial_groups=trial_groups)
    
    # make empty dataframe where windows from each file will be appended
    windows_all = pd.DataFrame()
    
    n_files = len(selected_files)
    
    # for each trial
    for i_file, (file_data, file_log) in enumerate(selected_files):
        try:
            # add windows
            print('\nAdding windows for trial {} of {}:\n'.format(i_file+1, n_files) + 
              '\tdata: {}\n'.format(file_data) + 
              '\tlog: {}'.format(file_log))
            
            data = load_data(file_data, file_log, channels)
            windows = create_windows(data, length=length, shift=shift,
                                     method=method, filter_type=filter_type, # returns filtered windows
                                     labelling_method=labelling_method)
            
            windows_all = windows_all.append(windows)
        except ValueError as e:
            print('An error occured while adding the windows from this file')
            print(e)
            print('moving on to the next one...\n')
        
    # save windows as pickle file
    if save:
        
        # generate filename based on requested dates/subjects/modes/groups
        to_add = []
        for (i, l) in enumerate((dates, subjects, modes, trial_groups)):
            if l:
                to_add.append('_'.join(map(str, l)))
            else:
                to_add.append('all')
                
        filename = 'windows_date_{}_subject_{}_mode_{}_groups_{}_{}ms_{}{}{}.pkl'.format(
            to_add[0], to_add[1], to_add[2], to_add[3],
            int(length*1000),
            labelling_method if (labelling_method != 'old') else 'old_labelling',
            '_unfiltered' if not (filter_type == 'real_time_filter') else '',
            append)
        
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
    # filenames = select_files(path_data, modes=[4])
    w = get_aggregated_windows(path_data, modes=[1,2,4], trial_groups=['ok', 'good'], 
                               length=0.2, shift=0.1,
                               save=True, path_out='windows', append='',
                               filter_type='real_time_filter',
                               labelling_method='power')

    # b = get_aggregate_baseline_windows(path_data,modes=[1],save=True,path_out='windows')
    
    
