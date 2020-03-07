#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:32:01 2020

@author: marley
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import seaborn as sns
import matplotlib.mlab as mlab

def silly(signal):
    if len(signal) < 20:
        print('aaaaaaahhhhhhhhh hell')
        return np.asarray([1,2,3])
    return np.asarray(signal[:3])

#Integrated IEMG
def iemg(signal):
    return np.sum(np.abs(signal))

#Mean absolute value
def mav(signal):
    return np.sum(np.abs(signal)) / len(signal)

#Modified mean absolute value
def mmav(signal):
    def weights(n, N):
        if n < 0.25*N: return 4*n/N
        elif 0.25*N <= n and n <= 0.75*N: return 1
        else: return 4*(n - N)/N
    
    w = np.array([weights(n, len(signal)) for n in range(len(signal))])
    return np.dot(w, np.abs(signal)) / len(signal)

#Modified mean abs value 2
def mmav2(signal):
    def weights(n,N):
        if n < .33*N or n > .67*N: return .33
        else: return .67
    w = np.array([weights(n, len(signal)) for n in range(len(signal))])
    return np.dot(w, np.abs(signal)) / len(signal)

#

#Variance
def var(signal):
    return np.sum(np.square(signal)) / (len(signal) - 1)

#Root mean square
def rms(signal):
    return np.sqrt(np.sum(np.square(signal)) / len(signal))


#In what follows, thresholds: 10-100mV
    

#Zero crossing
def zc(signal, threshold=40):
    signal = np.array(signal)
    signal -= np.mean(signal)
    
    shifted_signal = np.append(signal[1:], signal[-1])
    return len(signal[ (signal*shifted_signal < 0) & (np.abs(signal-shifted_signal) >= threshold)])

#Willison amplitude
def wamp(signal, threshold=20):
    shifted_signal = np.append(signal[1:], signal[-1])
    return len(signal[ np.abs(signal-shifted_signal) >= threshold])

def wl(signal):
    signal = np.array(signal)
    shifted_signal = np.append(signal[1:], signal[-1])
    return np.sum(np.abs(signal-shifted_signal))

# slope sign change
def ssch(signal):
    def is_neg(a):
        if a<0: return 1
        else: return 0
    return np.sum([is_neg(signal[i]*signal[i+1]) for i in range(len(signal)-1)])

#The Length of the Waveform Per Unit
def wfl(signal):
    return np.sum([np.abs(signal[i]-signal[i+1]) for i in range(len(signal)-1)]) / len(signal)

#Spectrogram

# helper function returns just the trimmed power spectrum - we don't care really about very low frequencies (below 5hz)
def get_psd(signal):
    psd,freqs = mlab.psd(signal,NFFT=256,window=mlab.window_hanning,Fs=250,noverlap=0)
    return psd

# some basic freq domain features
def freq_low(signal):
    return np.sum(get_psd(signal)[5:20])

def freq_20_40_abs(signal):
    return np.sum(get_psd(signal)[20:40])

def freq_40_60_abs(signal):
    return np.sum(get_psd(signal)[40:60])

def freq_60_80_abs(signal):
    return np.sum(get_psd(signal)[60:80])

def freq_80_100_abs(signal):
    return np.sum(get_psd(signal)[80:100])

def freq_100_120_abs(signal):
    return np.sum(get_psd(signal)[100:120])

# some more freq domain features
def freq_var(signal):
    if len(signal) < 250:
        print('aaaaaaahhhhhhhhh hell')
        return np.asarray([1,2,3])
    psd = get_psd(signal)
    return np.asarray([np.mean(psd[:40]),np.mean(psd[40:80]),np.mean(psd[80:])])


def freq_ssch(signal):
    return ssch(get_psd(signal)[:])
def freq_mav(signal):
    return mav(get_psd(signal)[:])

# counts the number of .20 crossings



### subinterval features
#Root mean squared subwindows 1 of 3
def rms3_1(signal):
    # split signal into three parts
    signal_split = signal[:int(len(signal)/3)]
    return np.sqrt(np.sum(np.square(signal_split))) / len(signal_split)

#Root mean squared subwindows 2 of 3
def rms3_2(signal):
    # split signal into three parts
    signal_split = signal[int(len(signal)/3):int(2*len(signal)/3)]
    return np.sqrt(np.sum(np.square(signal_split))) / len(signal_split)

#Root mean squared subwindows 3 of 3
def rms3_3(signal):
    # split signal into three parts
    signal_split = signal[int(2*len(signal)/3):]
    return np.sqrt(np.sum(np.square(signal_split))) / len(signal_split)



#Still want WL and SSC
    
