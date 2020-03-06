# methods for using in spectrogram - marley's methods adapted 
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os,random,math

import display_methods
import load_raw

SAMPLING_FREQ = 250

def get_spectral_content(ch, fs_Hz, shift=.1):
	NFFT = fs_Hz*2
	overlap = NFFT - int(shift * fs_Hz)
	spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                                     NFFT=NFFT,
                                                     window=mlab.window_hanning,
                                                     Fs=fs_Hz,
                                                     noverlap=overlap
                                                     ) # returns PSD power per Hz
	# convert the units of the spectral data
	spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
	return spec_t, spec_freqs, spec_PSDperBin # dB re: 1 uV


def plot_specgram(spec_freqs, spec_PSDperBin, title, shift,no_chan, i=1,cmap='terrain'):
	f_lim_Hz = [0, 30]   # frequency limits for plotting
	#plt.figure(figsize=(10,5))# assume figure obj already initialised
	spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
	plt.subplot(no_chan,1,i)
	plt.title(title)
	plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin),cmap=cmap)  # dB re: 1 uV
	plt.clim([-25,26])
	plt.xlim(spec_t[0], spec_t[-1]+1)
	plt.ylim(f_lim_Hz)
	plt.xlabel('Time (sec)')
	plt.ylabel('Frequency (Hz)')
	plt.subplots_adjust(hspace=.5)

def disp1(channel=[1,2,3,4],save_as='spec1.png'):
	lr = load_raw.load_dta(channel=channel+[13])
	# trim data
	keypressed = lr['keypressed']
	start_idx = keypressed.first_valid_index()
	stop_idx = keypressed.last_valid_index()
	lr_trimmed = lr.truncate(before=start_idx-750,after=stop_idx+750)
	only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]
	
	# plot multiple channels
	markings = np.array([only_keypressed.iloc[i].name for i in range(len(only_keypressed))])
	labels = np.asarray(only_keypressed['keypressed'])
	markings_labels = np.asarray([markings,labels]).T
	
	ch_names = [i for i in lr.columns.values if i not in ('timestamp(ms)','keypressed')]
	eeg = np.asarray([display_methods.filter_signal(np.asarray(lr[i])) for i in ch_names])
	
	fig = plt.figure(figsize=(10,int(5*len(ch_names))),dpi=80, facecolor='w', edgecolor='k')
	for idx,ch in enumerate(eeg):
		stop_plot_idx = start_idx + 10000
		t, spec_freqs, spec_PSDperBin = get_spectral_content(ch[start_idx:stop_plot_idx], SAMPLING_FREQ)
		plot_specgram(spec_freqs, spec_PSDperBin, 'channel = {}'.format(ch_names[idx]),
					shift=0,no_chan=len(ch_names),i=idx+1,cmap='terrain')
		# trim the markings and labels to only those that are within the specified range
		markings_labels_trimmed = [(m,l) for m,l in markings_labels if m<stop_plot_idx and m>start_idx]
		for mark,label in markings_labels_trimmed:
			plt.text((mark-start_idx)/250,.2,label,color='white',fontsize=20)
	plt.savefig(save_as)






