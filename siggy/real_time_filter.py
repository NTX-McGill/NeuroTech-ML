#!/usr/bin/env python
# coding: utf-8

# In[87]:


from scipy import zeros, signal, random
from match_labels import append_labels, label_window, filter_signal
# from ipynb.fs.full.features import filter_signal
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


def filter_sbs(n=10000, split=5000):
    data = random.random(n)
    result = []
    b = signal.firwin(150, 0.004)
    z = signal.lfilter_zi(b, 1)
    for i in range(n // split):
        test_data = data[split*i:split*(i + 1)]
        test_result = zeros(test_data.size)
        for i, x in enumerate(test_data):
            test_result[i], z = signal.lfilter(b, 1, [x], zi=z)
        result.append(test_result)
    
    return data, result

# if __name__ == '__main__':
#     data, result = filter_sbs()


# In[82]:


# plt.plot(result[0])
# plt.plot(result[1])
# plt.show()
# plt.plot(np.append(result[0], result[1]))


# In[116]:


def butter_filter(windows, fs=250, order=2, low=20, high=120, notch=True):
    result = []
    nyq = fs / 2
    b, a = signal.butter(order, [low/nyq, high/nyq], 'bandpass')
    z_butter = signal.lfilter_zi(b, a)
    for i, w in enumerate(windows):
        if i == 0 and notch:
            w, z_notch = notch_filter(w)
        elif notch:
            w, z_notch = notch_filter(w, z=z_notch)
        
        w, z_butter = signal.lfilter(b, a, w, zi=z_butter)
        result.append(w)
    
    return result

def notch_filter(arr, fs=250, z=[]):
    freqs = np.array([60.0])
    nyq = fs / 2
    for f in np.nditer(freqs):
        bp_stop_f = f + 3.0 * np.array([-1,1])
        b, a = signal.butter(3, bp_stop_f / nyq, 'bandstop')

        if len(z) == 0:
            z = signal.lfilter_zi(b, a)

    return signal.lfilter(b, a, arr, zi=z)

def test_filter(windows, fs=250, order=2, low=20, high=120):
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


# In[85]:


if __name__ == '__main__':
    markers = '../data/2020-02-16/001_trial2_right_air_2020-02-16-19-15-12-263.txt'
    fname = '../data/2020-02-16/001_trial2_right_air_OpenBCI-RAW-2020-02-16_19-14-29.txt'
    channels = [1,2,3,4,13]

    labelled_raw = append_labels(fname, markers, channels)
    labelled_raw['keypressed'] = np.NaN
    out = label_window(labelled_raw, shift=1, offset=0, take_everything=True)
    
#     plt.plot(labelled_raw.iloc[:,0])
    
    filtered_all = filter_signal(labelled_raw['channel 1'])
    input_windows = list(out.iloc[:10,0])
    results = test_filter(input_windows)
    for num, window in enumerate(results):
        plt.figure()
        plt.plot(filtered_all[num * 250:(num+1)*250])
        plt.plot(window)
        plt.show()

