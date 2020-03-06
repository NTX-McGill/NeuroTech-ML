import pandas as pd
import numpy as np

import datetime # for get_ms helper
import pytz # for get_ms helper

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# convolving signals etc
import seaborn as sns
import pyarrow.parquet as pq
import random

# smoothing
from statsmodels.nonparametric.smoothers_lowess import lowess

SAMPLING_FREQ = 250

# checked
def load_dta(markers='./001_trial1_right_log_18-09-46-931825.txt',
			 fname='./001_trial1_right_OpenBCI-RAW-2020-02-09_17-59-22.txt',
			 channel = [1,2,3,4]):
	try:# for the first type of labels data, try to load it
		df_labels = pd.read_csv(markers)
		print('labels dataframe',list(df_labels.columns))#trace
		input('press enter (trace)')#trace
		start = df_labels['timestamp(ms)'].iloc[0]
		end = df_labels['timestamp(ms)'].iloc[-1]
	except:# for the later kind of labeled data
		df_labels = pd.read_csv(markers,names=['datetime(ms)','int_time(ms)',
											   'prompt','leftright','finger','keypressed'])
		print('labels dataframe',list(df_labels.columns))#trace
		input('press enter (trace)')#trace
		start = df_labels['int_time(ms)'].iloc[0]
		end = df_labels['int_time(ms)'].iloc[-1]

	channel.append(13)
	data = np.loadtxt(fname,delimiter=',',skiprows=7,usecols=channel)
	eeg = data[:,:-1]
	timestamps = data[:,-1]

	return eeg,timestamps,start,end,df_labels# watch out over-loaded, returns two types of df_labels


# change this so that it takes a date
# checked
"""
Convert timestamp in keyboard markings file to unix milliseconds
inputs:
  str_time (string)
outputs:
  milliseconds (float)
"""
def get_ms(str_time, str_date='2020-02-09 '):# overload this to deal with two differnet types of database
	#if str_date[-1]!=' ': 
	#	str_date+=' '# must have a space!
	try:# first type of data
		date_time_str = str_date + str_time
		date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S:%f')

		timezone = pytz.timezone('America/New_York')
		timezone_date_time_obj = timezone.localize(date_time_obj)
		return timezone_date_time_obj.timestamp() * 1000
	except:
	# second type of data, here the str_time is actually already in the correct format
		return str_time
		

    

# return the date from data in labels data frame in string format
#def get_str_date():

### marley's filtering methods for the spectrogram
def notch_mains_interference(data):
    notch_freq_Hz = np.array([60.0])  # main + harmonic frequencies
    for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])  # set the stop band
        b, a = signal.butter(3, bp_stop_Hz / (SAMPLING_FREQ / 2.0), 'bandstop')
        arr = signal.lfilter(b, a, data, axis=0)
        print("Notch filter removing: " +
              str(bp_stop_Hz[0]) +
              "-" +
              str(bp_stop_Hz[1]) +
              " Hz")
    return arr

def filter_signal(arr, lowcut=4.0, highcut=60.0, order=3, notch=True):
    if notch:
        arr = notch_mains_interference(arr)
    nyq = 0.5 * SAMPLING_FREQ
    b, a = signal.butter(1, [lowcut / nyq, highcut / nyq], btype='band')
    for i in range(0, order):
        arr = signal.lfilter(b, a, arr, axis=0)
    return arr

def get_spectral_content(ch, fs_Hz, shift=0.1):
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
    return spec_t, spec_freqs, spec_PSDperBin  # dB re: 1 uV

def plot_specgram(spec_freqs, spec_PSDperBin,title,shift,i=1):
    f_lim_Hz = [0, 30]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
    plt.subplot(3,1,i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)
### marley's filtering methods for the spectrogram - end

### convolutions and filters etc. start

## simple convolution, rectangle block
def conv_1(sig,window):
	conv = np.repeat([0.,1.,0.], window)
	filtered = signal.convolve(sig,conv,mode='same') / window # normalizing factor
	return filtered

# MMAV2
def mmav2(sig,window):
	conv = np.repeat([0.5,1.,0.5], window)
	filtered = signal.convolve(sig, conv, mode='same') / (2*window)
	return filtered

## mavs
def mavs(sig,window):
	ar1 = np.asarray([4*k/ window for k in range(window//4)])
	ar2 = np.ones(window//2)
	ar3 = np.asarray([np.float(1 - 4*k/window) for k in range(window//4)])
	conv = np.concatenate([ar1,ar2,ar3])
	normalize = np.sum(conv)
	
	filtered = signal.convolve(sig, conv, mode='same') / normalize
	return filtered

# threshold
def thresh(x,a):
	if x>a: return 1
	else: return 0

# the willison convolution, not very efficient runs slowly
def willisen(sig,window):
	convolved = []
	for i in range(len(sig)-window):
		sa = np.sum([thresh(np.abs(sig[i+j] - sig[i+j+1]),0.0005) for j in range(window)])
		convolved.append(sa / window)
	return np.asarray(convolved)


### convolutions and filters etc. end


# uses plot_specgram and plots it based only on the filenames loaded with load method
def display_spec(markers='./001_trial1_right_log_18-09-46-931825.txt',
				 fname='./001_trial1_right_OpenBCI-RAW-2020-02-09_17-59-22.txt',
				 channel = [1,2,3,4]):
	eeg,timestamps,start,end,df_labels = load_dta(markers,fname,channel)
	start_idx = np.where(timestamps > get_ms(start))[0][0]
	end_idx = np.where(timestamps > get_ms(end))[0][0]

	print('labels dataframe columnnames\n',list(df_labels.columns))
	input()#trace
	try:# for the first type of data
		markings = [get_ms(val) for val in df_labels['timestamp(ms)'].values[::2]]# notice the 2 here... cause duplicate
		labels = df_labels.values[:,1]
	except:# for the second kind of data
		markings = [val[0] for val in df_labels[['int_time(ms)','keypressed']] if val[1] != None]
		print(markings[:5])
		input('trace 5')
		markings = [float(i) for i in markings]
		labels = df_labels.values[:,-1]
		labels = [i for i in labels if i!=None]# drop the none values

	for idx,ch in enumerate(eeg.T):
		ch = filter_signal(ch)
		t, spec_freqs, spec_PSDperBin = get_spectral_content(ch[start_idx:start_idx + 10000], 250)
		fig=plt.figure(figsize=(15,12), dpi= 80, facecolor='w', edgecolor='k')
		plot_specgram(spec_freqs, spec_PSDperBin,'channel {}'.format(idx + 1), 0)
		# trace
		print('type mark, type start')
		#print(type(mark),type(start))
		#input('enter trace')#trace
		try:
			for mark, label in zip(markings, labels):
				plt.text((mark - get_ms(start)-750)/1000,10,label, color='white')
		except:
			for mark,label in zip(markings,labels):
				print(type(mark),type(start),'\n',mark,start)
				input('enter trace')#trace
				plt.text((mark - start)/1000,10,label,color='white')
				


# just plots the graphs with only markers, fnam and channels
# timerange is a tuple of values 0 < start < end < 1    ---   (strt, fin)
def display_ts(markers='./001_trial1_right_log_18-09-46-931825.txt',
			   fname='./001_trial1_right_OpenBCI-RAW-2020-02-09_17-59-22.txt',
			   channel = [1,2,3,4],
			   savefig = True,
			   timerange = None):
	eeg,timestamps,start,end,df_labels = load_dta(markers,fname,channel)
	start_idx = np.where(timestamps > get_ms(start))[0][0]
	end_idx = np.where(timestamps > get_ms(end))[0][0]
	
	# trim the indices furthur for display purposes if the timerange is specified
	if timerange:
		strt,fin = timerange
		#if strt >= fin : raise Exception('start must be  < fin')
		new_start = int((1 - strt)*start_idx + strt * end_idx)
		new_end = int((1 - fin) * start_idx + fin * end_idx)
		start_idx = new_start
		end_idx = new_end
	
	time_interval = timestamps[start_idx:end_idx]
	
	markings = [get_ms(val) for val in df_labels['timestamp(ms)'].values[::2]]# 2 to account for duplicate labels
	labels = df_labels.values[:,1]
	
	# plot each signal
	plt.figure(figsize=(12,15))
	n = len(eeg.T)
	for idx,ch in enumerate(eeg.T):

		# the data is like buggy, i think every dta point appears twice !!!!

		# first trim the channel but not fully
		mmav2_ch = mmav2(ch,1200)
		mavs_ch = mavs(ch,1200)

		trimmed_ch = ch[start_idx:end_idx]
		trimmed_mmav2 = mmav2_ch[start_idx:end_idx]
		trimmed_mavs = mavs_ch[start_idx:end_idx]
		#lowess_ch,x = lowess(endog = trimmed_ch, exog = np.arange(len(trimmed_ch))).T

		print('len ch, len time_interval',len(trimmed_ch),len(time_interval))# trace
		plt.subplot(n,1,idx+1)
		plt.plot(time_interval,trimmed_ch,label='raw signal',color='blue',alpha=0.15)
		plt.plot(time_interval,trimmed_mmav2,label='mmav2',color='red',alpha=0.4)
		plt.plot(time_interval,trimmed_mavs,label='mavs',color='green',alpha=0.4)
		#plt.plot(time_interval,lowess_ch,label='lowess smoothing',color=(.7,.3,.5),alpha=0.4)
		#plt.plot(time_interval[::3],trimmed_ch[::3],label='changed thing',color='purple')
		plt.title('channel '+str(idx+1))
		plt.legend()
		plt.grid()
	plt.show()
	if savefig:
		savename = 'times_series_channels_'+str(channel)+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'.png'
		plt.savefig(savename)
		print('figure saved')
	#plt.savefig()
	#input()
		









