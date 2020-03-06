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

#Still want WL and SSC
    
