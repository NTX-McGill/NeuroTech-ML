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
    return np.sum([is_neg(signal(i)*signal(i+1)) for i in range(len(signal)-1)])

#The Length of the Waveform Per Unit
def wfl(signal):
    return np.sum([np.abs(signal(i)-signal(i+1)) for i in range(len(signal)-1)]) / len(signal)


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
    
