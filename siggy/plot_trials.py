#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:14:20 2020

@author: michellewang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import groupby

import os
from pathlib import Path
import re
import pickle

from match_labels import select_files, load_data, create_windows
from constants import MODE_MAP

colours_map = {'channel 1':'grey', 'channel 2':'darkviolet', 'channel 3':'mediumblue', 'channel 4':'darkgreen',
               'channel 5':'gold', 'channel 6':'darkorange', 'channel 7':'red', 'channel 8':'saddlebrown'}

fingers_map = {1:'Right thumb', 2:'Right index', 3:'Right middle', 4:'Right ring', 5:'Right pinky',
               6:'Left thumb', 7:'Left index', 8:'Left middle', 9:'Left ring', 10:'Left pinky',
               0:'Baseline'}

mode_id_map = {v:k.lower() for (k, v) in MODE_MAP.items()}  # map from mode ID to mode name

seed = 3791

channels = [1, 2, 3, 4, 5, 6, 7, 8]
fingers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# text size
size_axis_title = 'large'
size_axis_labels = 'large'
size_axis_ticks = 'medium'
size_fig_title = 'x-large'

def sample_df(df, column_name, value, n):
    '''
    Returns a subsample of n rows in df where a column has a specific value

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    column_name : string
        Name of column to check.
    value : string
        Value of interest.
    n : int
        Number of samples to take.
        If this exceeds the number of rows where df[column_name] == value, 
        then the returned dataframe will contain all available samples

    Returns
    -------
    pd.DataFrame
        Sampled dataframe.

    '''
    
    # get all rows of df that satisfy condition
    df_value = df.loc[df[column_name] == value]
    max_n = len(df_value)
    
    # sample rows
    return df_value.sample(n=min(n, max_n), replace=False, random_state=seed)

def plot_window(ax, row_data, channel_names, center='zero'):
    '''
    Plots window timeseries for a specific row of a windows dataframe.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to be plotted.
    row_data : Series
        Row from windows dataframe containing channel columns.
    channel_names : list of strings
        Channels to be plotted.

    Returns
    -------
    None.

    '''
        
    ylim = [-200, 200] # y-axis limits
    
    legend = []
    
    # plot each channel
    for i_channel, channel_name in enumerate(channel_names):
        data = row_data[channel_name]
        
        if center == 'average':
            data = data - np.mean(data)
        
        ax.plot(data, color=colours_map[channel_name], alpha=0.8)
        legend.append(channel_names[i_channel].capitalize())

    # add legend of channel names
    ax.legend(legend, ncol=2)
    
    # set x/y-axis limits
    ax.set_xlabel('Time sample', fontsize=size_axis_labels)
    ax.set_xlim(left=0, right=len(data))
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
        
    return

def plot_windows(df, channel_names, col_titles, center='zero', title=None):
    '''
    Plots all windows in a windows dataframe, with one column per finger.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of windows.
    channel_names : list of strings
        Names of columns containing timeseries to be plotted.
    col_titles : dict of int:string entries
        Map from finger number to title of column containing windows for this finger.
    title : string, optional
        Title of figure. If None, no title is added. The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Plotted figure.

    '''
    
    # get list of used fingers and of associated number of samples
    used_fingers = np.unique(np.array(df['finger']), return_counts=True)
    n_samples = used_fingers[1]
    used_fingers = used_fingers[0]
    n_fingers = len(used_fingers)
            
    # start figure
    n_rows = int(np.max(n_samples))
    n_cols = int(n_fingers)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharey='row',
                             figsize=(n_cols*6, n_rows*3))
    
    # grid to keep track of which subplots are filled
    subplots_state = [[] for _ in range(n_fingers)] 
    
    for _, row_data in df.iterrows():
        
        # get position of subplot
        finger = row_data['finger']
        i_col = np.where(used_fingers == finger)[0][0]
        i_row = sum(subplots_state[i_col])
        
        # update grid
        subplots_state[i_col].append(1)
        
        # select correct subplot to fill
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[i_col]
        elif n_cols == 1:
            ax = axes[i_row]
        else:
            ax = axes[i_row][i_col]
            
        ax.tick_params(axis='both', which='major', labelsize=size_axis_ticks)
        
        # plot window
        plot_window(ax, row_data, channel_names, center=center)
        
        # add subplot title for first row
        if i_row == 0:
            ax.set_title(col_titles[finger], fontsize=size_axis_title)
        if i_col == 0:
            ax.set_ylabel('Amplitude (\u03BCV)', fontsize=size_axis_labels)
    
    # 'turn off' subplots that are empty
    for i_col in range(n_cols):
        i_row = sum(subplots_state[i_col])
        while i_row < n_rows:
            axes[i_row][i_col].axis('off')
            i_row += 1

    # add figure title
    if title is not None:
        left_space = 0.20/fig.get_size_inches()[0]
        top_space = 0.4/fig.get_size_inches()[1]
        fig.suptitle(title, fontsize=size_fig_title,
                     x=0.5+left_space, y=(1-top_space/2))
    else:
        left_space = 0
        top_space = 0
        
    fig.tight_layout(rect=[left_space, 0, 1, 1-top_space])
        
    return fig

def generate_col_titles(df_windows):
    # get total number of keypresses per finger
    finger_sequence = [g[0] for g in groupby(df_windows['finger'])]
    dict_finger_count = {f:finger_sequence.count(f) for f in np.unique(finger_sequence)}

    # generate column titles
    col_titles = {}
    for finger in np.unique(finger_sequence):
        if finger == 0:
            col_titles[finger] = fingers_map[finger] # baseline
        else:
            # col_titles[finger] = '{}'.format(fingers_map[finger])
            col_titles[finger] = '{} ({} keypresses total)'.format(fingers_map[finger], dict_finger_count[finger])
    return col_titles

def plot_trials(path_data, n, dates=None, subjects=None, modes=None, trial_groups=None, path_out='.', save=False, add_title=True, overwrite=False):
    '''
    Plots subset of windows for all trials with specific date/subject/mode.
    Optionally saves figure as .png file.

    Parameters
    ----------
    path_data : string
        Path to data directory
    n : int
        Maximnum number of samples for each finger.
    dates, subjects, mode, trial_groups : lists, optional
        Parameters for trial selection, passed to select_files()
    path_out : string, optional
        Path to output directory. The default is '.'.
    save : boolean, optional
        If True, plot is saved as .png file inside a subject directory. The default is False.
    add_title : boolean, optional
        If True, generates and adds a title for each trial, otherwise no title. The default is True.
    overwrite : boolean, optional
        If True, overwrites existing figures. The default is False.

    Returns
    -------
    None.

    '''

    filenames = select_files(path_data=path_data, dates=dates, subjects=subjects, modes=modes, trial_groups=trial_groups)

    for (filename_data, filename_log) in filenames:

        # regex
        r_date = '\d{4}-\d{2}-\d{2}'
        r_subject = '\/0\d{2}'
        r_trial = '[tT]rial\d+'

        # get date, subject, and trial number
        date = re.search(r_date, filename_data).group()
        subject_id = re.search(r_subject, filename_data).group()[1:]
        i_trial = re.search('\d+', re.search(r_trial, filename_data).group()).group()

        # path to saved figure
        path_dir_out = os.path.join(path_out, subject_id)
        filename_out = os.path.join(path_dir_out, 'subject_{}_{}_trial{}_{}.png'.format(subject_id, date, i_trial, n))

        # skip trial if already plotted
        if not overwrite and os.path.isfile(filename_out):
            print('Skipping: {}, subject {}, trial {}'.format(date, subject_id, i_trial))
            continue
        
        # make directory if it doesn't exist
        Path(path_dir_out).mkdir(parents=True, exist_ok=True)

        print('Now plotting: {}, subject {}, trial {}'.format(date, subject_id, i_trial))

        # get all windows for trial
        data = load_data(filename_data, filename_log, channels)
        df_windows_all = create_windows(data)

        # resample windows
        df_windows = pd.DataFrame()
        for i_finger, finger in enumerate(fingers):
            df_finger = sample_df(df_windows_all, 'finger', finger, n)
            df_windows = df_windows.append(df_finger)

        # get channel names
        column_names = list(df_windows)
        channel_names = [c for c in column_names if 'channel' in c]    

        # get mode
        mode_id = np.array(df_windows['mode'])[0]
        mode = mode_id_map[mode_id]

        col_titles = generate_col_titles(df_windows_all)

        if add_title:
            fig_title = 'Subject {}, {}, trial {} ({})'.format(subject_id, date, i_trial, mode)
        else:
            fig_title = None
            
        fig = plot_windows(df_windows, channel_names, col_titles, title=fig_title)

        # save
        if save:
            fig.savefig(filename_out, dpi=100)

def plot_pickled_windows(path_windows, n, center='zero', fig_title=None, path_out='.', save=False):
    '''
    Plots subset of windows from a pickled windows DataFrame.
    Optionally saves figure as .png file.

    Parameters
    ----------
    path_windows : string
        Path to pickled windows
    n : int
        Maximnum number of samples for each finger.
    fig_title : string
        Title of figure to be passed to plot_windows().
    path_out : string, optional
        Path to output directory. The default is '.'.
    save : boolean, optional
        If True, plot is saved as .png file inside a subject directory. The default is False.
    overwrite : boolean, optional
        If True, overwrites existing figures. The default is False.

    Returns
    -------
    None.

    '''
    
    with open(path_windows, 'rb') as file_in:
        df_windows_all = pickle.load(file_in)

    # resample windows
    df_windows = pd.DataFrame()
    for i_finger, finger in enumerate(fingers):
        df_finger = sample_df(df_windows_all, 'finger', finger, n)
        df_windows = df_windows.append(df_finger)

    # get channel names
    column_names = list(df_windows)
    channel_names = [c for c in column_names if 'channel' in c]    

    col_titles = generate_col_titles(df_windows_all)

    fig = plot_windows(df_windows, channel_names, col_titles, center=center, title=fig_title)

    # save
    if save:
        filename_windows = os.path.basename(path_windows)
        filename_out = 'plot_{}.png'.format(os.path.splitext(filename_windows)[0])
        path_out = os.path.join(path_out, filename_out)
        fig.savefig(path_out, dpi=100)
        print('Saved figure: {}'.format(path_out))
        
    return fig

if __name__ == '__main__':
    
    path_data = '../data'
    path_out = '../data/window_plots'
    n = 15
    
    # creates windows and plots them
    plot_trials(path_data, n=n, subjects=['013'], modes=[1, 2, 4], 
                path_out=path_out, save=True, add_title=True)
    
    # # plots windows that have already been saved
    # path_windows = 'windows/windows_date_all_subject_all_mode_1_2_4_groups_ok_good_200ms_test.pkl'
    # plot_pickled_windows(path_windows, n, path_out='windows', save=True)
    
    
    
    
    