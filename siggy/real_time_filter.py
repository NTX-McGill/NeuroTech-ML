#!/usr/bin/env python
# coding: utf-8

# In[87]:


from scipy import zeros, signal, random
from match_labels import *
# from ipynb.fs.full.features import filter_signal
import matplotlib.pyplot as plt
import numpy as np
import time

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


# def butter_filter(windows, fs=250, order=2, low=5, high=120, notch=True):
#     result = []
#     nyq = fs / 2
#     b, a = signal.butter(order, [low/nyq, high/nyq], 'bandpass')
#     z_butter = signal.lfilter_zi(b, a)
#     for i, w in enumerate(windows):
#         if i == 0 and notch:
#             w, z_notch = notch_filter(w)
#         elif notch:
#             w, z_notch = notch_filter(w, z=z_notch)
        
#         w, z_butter = signal.lfilter(b, a, w, zi=z_butter)
#         result.append(w)
    
#     return result

# def notch_filter(arr, fs=250, z=[]):
#     freqs = np.array([60.0])
#     # nyq = fs / 2
#     nyq = fs / 2
#     for f in np.nditer(freqs):
#         bp_stop_f = f + 3.0 * np.array([-1,1])
#         b, a = signal.butter(3, bp_stop_f / nyq, 'bandstop')

#         if len(z) == 0:
#             z = signal.lfilter_zi(b, a)

#     return signal.lfilter(b, a, arr, zi=z)

def test_filter(windows, shift, fs=250, order=2, low=5.0, high=50.0):
    num_shifted = int(shift * fs)
    splits = [i for i in range(0, 250, num_shifted)]
    result = []
    nyq = fs / 2
    bb, ba = signal.butter(order, [low/nyq, high/nyq], 'bandpass')
    bz = signal.lfilter_zi(bb, ba)

    notch_freq = 60.0
    nb, na = signal.iirnotch(notch_freq, notch_freq / 6, fs)
    nz = signal.lfilter_zi(nb, na)
    for w in windows:
        # w, nz = signal.lfilter(nb, na, w, zi=nz)
        # w, bz = signal.lfilter(bb, ba, w, zi=bz)
        
        temp_nz, temp_bz = nz, bz
        for i, s in enumerate(w):
            # w[i], nz = signal.lfilter(nb, na, [s], zi=nz)
            w[i], temp_nz = signal.lfilter(nb, na, [s], zi=temp_nz)
            
            if i == num_shifted - 1:
                nz = temp_nz
                
        for i, s in enumerate(w):
            # w[i], bz = signal.lfilter(bb, ba, [s], zi=bz)
            w[i], temp_bz = signal.lfilter(bb, ba, [s], zi=temp_bz)
            
            if i == num_shifted - 1:
                bz = temp_bz
            
        result.append(w)
    
    return result

# def foo_filter(x, y):
#     nyq = fs / 2
#     bb, ba = signal.butter(order, [low/nyq, high/nyq], 'bandpass')
#     bz = signal.lfilter_zi(bb, ba)

#     notch_freq = 60.0
#     nb, na = signal.iirnotch(notch_freq, notch_freq / 6, fs)
#     nz = signal.lfilter_zi(nb, na)
    
#     for s in 

# In[85]:


if __name__ == '__main__':
    markers = '../data/2020-02-16/001_trial2_right_air_2020-02-16-19-15-12-263.txt'
    fname = '../data/2020-02-16/001_trial2_right_air_OpenBCI-RAW-2020-02-16_19-14-29.txt'
    
    # load the raw info 
    labelled_raw = load_data(fname, markers)
    
    out = create_windows(labelled_raw, shift=0.1, offset=0, filter_type='original_filter', take_everything=False, sample=False)
    input_windows = np.copy(out.loc[:14, 'channel 1'].to_numpy())
    results_low_20_high_120 = test_filter(input_windows, 0.1, low=20.0, high=120.0)
    
    out = create_windows(labelled_raw, shift=0.1, offset=0, filter_type='original_filter', take_everything=True, sample=False)
    input_windows = np.copy(out.loc[:14, 'channel 1'].to_numpy())
    results_low_5_high_50 = test_filter(input_windows, 0.1, low=5.0, high=50.0)

    
    #Offline filter
    # filtered_all = filter_signal(labelled_raw['channel 1'])
    
    
    # results_low_5_high_120 = test_filter(input_windows, 0.1, low=5.0)
    
    num_plots = 10
    for i in range(num_plots):
        plt.figure()
        plt.plot(filtered_all[i * 250:(i+1)*250],label='OG filter Low: 20, High: 120')
        plt.plot(results_low_20_high_120[i], label='Low: 20, High: 120')
        plt.plot(results_low_5_high_50[i], label='Low: 5, High: 50')
        plt.title('window number ' + str(i))
        plt.legend()
        plt.show()

