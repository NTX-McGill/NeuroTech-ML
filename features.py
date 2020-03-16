#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:32:01 2020

@author: marley
"""

import numpy as np
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

# abs value varience
def var_abs(signal):
    return np.sum(np.square(np.abs(signal) - np.array(np.mean(np.abs(signal))*len(signal)))) / (len(signal) - 1)

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
def freq_feats(signal):
    if len(signal) < 250:
        print('we need to fix this : the window is too short')
        return [1,2,3,4,5,6]
    psd = get_psd(signal)
    return [np.mean(psd[5:20]),np.mean(psd[20:40]),np.mean(psd[40:60]),np.mean(psd[60:80]),np.mean(psd[80:100]),np.mean(psd[100:120])]    

# some more freq domain features
def freq_var(signal):
    if len(signal) < 250:
        print('aaaaaaahhhhhhhhh hell : the window is too short')
        return np.asarray([1,2,3])
    psd = get_psd(signal)
    return np.asarray([np.mean(np.abs(psd[:40] - np.array([np.mean(psd[:40])]*40))),
                       np.mean(np.abs(psd[40:80] - np.array([np.mean(psd[40:80])]*20))),
                               np.mean(np.abs(psd[80:] - np.array([np.mean(psd[80:])]*len(psd[80:]))))])

# more freq domain features
def freq_misc(signal):
    if len(signal) < 250:
        print('window too short')
        return [1,2,3,4]
    psd = get_psd(signal)
    return [ssch(psd),mav(psd),mmav(psd),zc(psd-[.5]*len(psd))]


### subinterval features
#Root mean squared subwindows
def rms_3(signal):
    signal_split = [signal[:int(len(signal)/3)],signal[int(len(signal)/3):int(2*len(signal)/3)],signal[int(2*len(signal)/3):]]
    return [np.sqrt(np.sum(np.square(i))) / len(signal_split[0]) for i in signal_split]



#Still want WL and SSC
    
## Unhelpful feature: power spectral density
""" # FIX DIMENSIONS LATER
def psd(signal):
    shift = 0.1
    fs_Hz = 250
    NFFT = 256
    overlap = NFFT - int(shift * fs_Hz)
    
    # Pxx - 1D array for power spectrum values
    # freq - 1D array corresponding to Pxx values
    Pxx, freq = mlab.psd(np.squeeze(signal),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   )
    # Make it size 130 so we can splice it
    Pxx.append(Pxx[-1])
    # Bin it by taking average power of every 10 hz
    Pxx_bins = np.reshape(Pxx,(10,-1)).mean();
    
    return Pxx_bins
"""