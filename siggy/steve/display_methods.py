import load_raw # rolands loading method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pyarrow.parquet as pq
from scipy import signal

import os

SAMPLING_FREQ = 250

### SMOOTHING
# make some convolutions
# simple convolution, takes the signal and window size, returns filtered signal
def conv_1(sig, window):
    conv = np.repeat([0.,1.,0.], window)
    filtered = signal.convolve(sig, conv, mode='same') / window # normalising factor
    return filtered

# MMAV1
def mmav2(sig,window):
    conv = np.repeat([0.5,1.,0.5], window)
    filtered = signal.convolve(sig, conv, mode='same') / (2*window)
    return filtered

# MAVS - the slopey one
def mavs(sig,window):
    ar1 = np.asarray([4*k/ window for k in range(window//4)])
    ar2 = np.ones(window//2)
    ar3 = np.asarray([np.float(1 - 4*k/window) for k in range(window//4)])
    conv = np.concatenate([ar1,ar2,ar3])
    normalize = np.sum(conv)
    
    filtered = signal.convolve(sig, conv, mode='same') / normalize
    return filtered

def thresh(x,a):
    if x>a: return 1
    else: return 0

def willisen(sig,window):
    convolved = []
    for i in range(len(sig)-window):
        sa = np.sum([thresh(np.abs(sig[i+j] - sig[i+j+1]),0.0005) for j in range(window)])
        convolved.append(sa / window)
    return np.asarray(convolved)

# takes signal (array), output array of same length
def rms(interval, halfwindow):
    """ performs the moving-window smoothing of a signal using RMS """
    n = len(interval)
    rms_signal = np.zeros(n)
    for i in range(n):
        small_index = max(0, i - halfwindow)  # intended to avoid boundary effect
        big_index = min(n, i + halfwindow)    # intended to avoid boundary effect
        window_samples = interval[small_index:big_index]

        # here is the RMS of the window, being attributed to rms_signal 'i'th sample:
        rms_signal[i] = np.sqrt(sum([s**2 for s in window_samples])/len(window_samples))

    return rms_signal




# notch mains
def notch_mains_interference(data):
	notch_freq_Hz = np.array([60.0]) # main + harmonic frequencies
	for  freq_Hz in np.nditer(notch_freq_Hz):# loop over each target freq
		bp_stop_Hz = freq_Hz + 3.0 * np.array([-1,1])# set the stop band
		b,a = signal.butter(3,bp_stop_Hz / (SAMPLING_FREQ /2.0),'bandstop')
		arr = signal.lfilter(b,a,data,axis=0)
		print('notch filter removing:'+str(bp_stop_Hz[0]) + '-'+str(bp_stop_Hz[1])+"Hz")
	return arr

# filter signal function, low and high pass filters
def filter_signal(arr,lowcut=5.0,highcut=120.0,order=4,notch=True):
	nyq = 0.5 * SAMPLING_FREQ
	if notch:
		arr = notch_mains_interference(arr)
	b,a = signal.butter(order,[lowcut/nyq, highcut / nyq], btype='band')
	return signal.lfilter(b,a,arr)

# filter signal with low and high pass filters and add notch
# def filter_signal_notch(arr,lowcut=5.0,highcut=120.0,order=4):
	


# plot the signal
def plot_ts(markers='./001_trial1_right_keyboard_2020-02-16-19-09-10-309.txt',
			fname='./001_trial1_right_keyboard_OpenBCI-RAW-2020-02-16_18-59-08.txt',
			channel=[1,2,3,4],figsize=(8,15),time_interval=None):#no_dta_pts=None
	"""
	plot labelled time series data
	time interval if specified must be a two-tuple with values (x,y) 0<x<y<1 
	"""
	# load labelled raw data
	channel.append(13)
	lr = load_raw.load_dta(markers,fname,channel)

	# trim the data frame for plotting
	keypressed = lr['keypressed']
	start_idx = keypressed.first_valid_index()
	stop_idx = keypressed.last_valid_index()
	lr_trimmed = lr.truncate(before=start_idx-500,after=stop_idx+500)
	only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# trim more if user wants it to be trimmed
	if time_interval:
		start_idx_new = int(start_idx + time_interval[0]*(stop_idx-start_idx))
		stop_idx_new = int(start_idx + time_interval[1]*(stop_idx-start_idx))
		lr_trimmed = lr.truncate(before=start_idx_new,after=stop_idx_new)
		only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# plot the data
	channels = [i for i in lr.columns if i not in ('timestamp(ms)','keypressed')]
	count=0
	plt.figure(figsize=figsize)
	for channel_name in channels:
		count+=1
		ch = lr_trimmed[channel_name].values
		timestamps = lr_trimmed['timestamp(ms)'].values
		
		###
		ch_keypress = only_keypressed[channel_name].values
		ts_keypress = only_keypressed['timestamp(ms)'].values
		
		np.unique(only_keypressed['keypressed'].values)
		
		keys_pressed_unique = np.unique(only_keypressed['keypressed'].values)
		ts_keypress = []
		ch_i_keypress = []
		labels = []
		for letter in keys_pressed_unique:
		    labels.append('$'+letter+'$')
		    ch_i_keypress.append(only_keypressed.loc[only_keypressed['keypressed']==letter][channel_name].values)
		    ts_keypress.append(only_keypressed.loc[only_keypressed['keypressed']==letter]['timestamp(ms)'].values)
		    
		# make the plot - raw data
		plt.subplot(len(channels),1,count)
		plt.title(channel_name,fontsize=15)
		plt.plot(timestamps,ch)
		

		for x,y,lab in zip(ts_keypress,ch_i_keypress,labels):
		    plt.plot(x,y+np.ones(len(y))*300,marker=lab,color='red',linestyle='None')


	plt.savefig('asdf.png')
	plt.show()

### same thing but with no x axis
def plot_ts_2(markers='./001_trial1_right_keyboard_2020-02-16-19-09-10-309.txt',
			fname='./001_trial1_right_keyboard_OpenBCI-RAW-2020-02-16_18-59-08.txt',
			channel=[1,2,3,4],figsize=(8,15),time_interval=None):#no_dta_pts=None
	"""
	plot labelled time series data
	time interval if specified must be a two-tuple with values (x,y) 0<x<y<1 
	"""
	# load labelled raw data
	channel.append(13)
	lr = load_raw.load_dta(markers,fname,channel)

	# trim the data frame for plotting
	keypressed = lr['keypressed']
	start_idx = keypressed.first_valid_index()
	stop_idx = keypressed.last_valid_index()
	lr_trimmed = lr.truncate(before=start_idx-500,after=stop_idx+500)
	only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# trim more if user wants it to be trimmed
	if time_interval:
		start_idx_new = int(start_idx + time_interval[0]*(stop_idx-start_idx))
		stop_idx_new = int(start_idx + time_interval[1]*(stop_idx-start_idx))
		lr_trimmed = lr.truncate(before=start_idx_new,after=stop_idx_new)
		only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# plot the data
	channels = [i for i in lr.columns if i not in ('timestamp(ms)','keypressed')]
	count=0
	plt.figure(figsize=figsize)
	for channel_name in channels:
		count+=1
		ch = lr_trimmed[channel_name].values
		timestamps = lr_trimmed['timestamp(ms)'].values
		
		###
		ch_keypress = only_keypressed[channel_name].values
		ts_keypress = only_keypressed['timestamp(ms)'].values
		
		np.unique(only_keypressed['keypressed'].values)
		
		keys_pressed_unique = np.unique(only_keypressed['keypressed'].values)
		ts_keypress = []
		ch_i_keypress = []
		labels = []
		for letter in keys_pressed_unique:
		    labels.append('$'+letter+'$')
		    ch_i_keypress.append(only_keypressed.loc[only_keypressed['keypressed']==letter][channel_name].values)
		    ts_keypress.append(only_keypressed.loc[only_keypressed['keypressed']==letter]['timestamp(ms)'].values)
		    
		# make the plot - raw data
		plt.subplot(len(channels),1,count)
		plt.title(channel_name,fontsize=15)
		plt.plot(ch,label='raw')
		
		# filter the signal and plot smoothed
		filt_conv_1 = conv_1(ch,100)
		plt.plot(np.arange(len(filt_conv_1))[100:-100],filt_conv_1[100:-100],label='conv filter')
		
		# frequency filter
		freq_filtered = filter_signal(ch)
		baseline = np.mean(ch)
		plt.plot(np.arange(len(freq_filtered))[150:],freq_filtered[150:]+np.ones(len(freq_filtered)-150)*baseline, label='frequency filter',alpha=0.3)

		l = list(lr_trimmed['timestamp(ms)'].values)
		for x,y,lab in zip(ts_keypress,ch_i_keypress,labels):
			x_new = [l.index(i) for i in x]
			plt.plot(x_new,y+np.ones(len(y))*300,marker=lab,color='red',linestyle='None')

		plt.legend()

	#plt.savefig('djjd.png')
	plt.show()

### same thing but with no x axis
def plot_ts_filtered(markers='./001_trial1_right_keyboard_2020-02-16-19-09-10-309.txt',
			fname='./001_trial1_right_keyboard_OpenBCI-RAW-2020-02-16_18-59-08.txt',
			channel=[1,2,3,4],figsize=(8,15),time_interval=None,explicit_interval=None,
			save_fig=None,disp=False):
	"""
	plot labelled time series data
	time interval if specified must be a two-tuple with values (x,y) 0<x<y<1 
	explicit_interval is just indices (start, stop)
	"""
	# load labelled raw data
	channel.append(13)
	lr = load_raw.load_dta(markers,fname,channel)

	# trim the data frame for plotting
	
	keypressed = lr['keypressed']
	try:
		start_idx = keypressed.first_valid_index()
		stop_idx = keypressed.last_valid_index()
		# by default trim like this - to be rectified in following conditional statments if needed
		lr_trimmed = lr.truncate(before=start_idx-500,after=stop_idx+500)
	except:
		start_idx = 0
		stop_idx = 50000
		lr_trimmed = lr.truncate(before=start_idx,after=stop_idx)
	only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# trim more if user wants it to be trimmed
	if time_interval:
		start_idx_new = int(start_idx + time_interval[0]*(stop_idx-start_idx))
		stop_idx_new = int(start_idx + time_interval[1]*(stop_idx-start_idx))
		lr_trimmed = lr.truncate(before=start_idx_new,after=stop_idx_new)
		only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# trim according to an explicit interval - specially used when saveing pngs
	elif explicit_interval:
		start_idx = explicit_interval[0]
		stop_idx = explicit_interval[1]
		lr_trimmed = lr.truncate(before=start_idx , after=stop_idx)
		only_keypressed = lr_trimmed[lr_trimmed['keypressed'].notna()]

	# plot the data
	the_channels = [i for i in lr.columns if i not in ('timestamp(ms)','keypressed')]
	count=0
	fig = plt.figure(figsize=figsize)
	for channel_name in the_channels:
		count+=1
		ch = lr_trimmed[channel_name].values
		timestamps = lr_trimmed['timestamp(ms)'].values
		
		###
		ch_keypress = only_keypressed[channel_name].values
		ts_keypress = only_keypressed['timestamp(ms)'].values
		
		np.unique(only_keypressed['keypressed'].values)
		
		keys_pressed_unique = np.unique(only_keypressed['keypressed'].values)
		ts_keypress = []
		ch_i_keypress = []
		labels = []
		for letter in keys_pressed_unique:
		    labels.append('$'+letter+'$')
		    ch_i_keypress.append(only_keypressed.loc[only_keypressed['keypressed']==letter][channel_name].values)
		    ts_keypress.append(only_keypressed.loc[only_keypressed['keypressed']==letter]['timestamp(ms)'].values)
		    
		# make the plot - raw data
		chop_idx = 250
		plt.subplot(len(the_channels),1,count)
		print('channel_name',channel_name)#channel_name
		plt.title(channel_name,fontsize=17)
		
		# frequency filter
		freq_filtered = filter_signal(ch)
		plt.plot(np.arange(len(freq_filtered))[chop_idx:],freq_filtered[chop_idx:], label='frequency notch filter',color='green',alpha=.3)
		
		
		# smooth the filtered with convolution
		freq_filtered_conv = mmav2(freq_filtered,window=16)
		freq_filtered_conv_abs = mmav2(np.abs(freq_filtered),window=10)
		freq_filtered_rms_15 = rms(freq_filtered,15)
		freq_filtered_rms_5 = rms(freq_filtered,5)
		freq_filtered_rms_30 = rms(freq_filtered,30)
		freq_diff_smooth = mmav2(np.abs(np.diff(freq_filtered)),window=10)

		plt.plot(np.arange(len(freq_filtered_conv))[chop_idx:],freq_filtered_conv[chop_idx:],label='conv',alpha=.75)
		plt.plot(np.arange(len(freq_filtered_conv_abs))[chop_idx:],freq_filtered_conv_abs[chop_idx:],label='conv abs',alpha=.75)

		plt.plot(np.arange(len(freq_filtered_rms_15))[chop_idx:],freq_filtered_rms_15[chop_idx:],label='rms 15',alpha=.5)
		plt.plot(np.arange(len(freq_filtered_rms_5))[chop_idx:],freq_filtered_rms_5[chop_idx:],label='rms 5',alpha=.5)
		plt.plot(np.arange(len(freq_filtered_rms_30))[chop_idx:],freq_filtered_rms_30[chop_idx:],label='rms 30',alpha=.6)

		plt.plot(np.arange(len(freq_diff_smooth))[chop_idx:],freq_diff_smooth[chop_idx:],label='freq abs difference',color='red',alpha=.35)

		l = list(lr_trimmed['timestamp(ms)'].values)
		for x,y,lab in zip(ts_keypress,ch_i_keypress,labels):
			x_new = [l.index(i) for i in x]
			plt.plot(x_new,np.ones(len(y))*75.0,marker=lab,color='red',linestyle='None')
		
		plt.legend(fontsize=12,loc=1,framealpha=.1)
		
		# set the y axis max min if i'm taking pictures
		if explicit_interval:
			x1,x2,y1,y2 = plt.axis()
			plt.axis((x1,x2,-65,175))
	if save_fig: plt.savefig(save_fig)
	if disp: plt.show()
	plt.close(fig)



# method to get the intervals surrounding each specific letter
def get_explicit_letter_intervals(markers='./001_trial1_right_keyboard_2020-02-16-19-09-10-309.txt',
								fname='./001_trial1_right_keyboard_OpenBCI-RAW-2020-02-16_18-59-08.txt',
								channel=[1,2,3,4]):
	# load the data and find where they all are
	channel.append(13)
	channel1 = channel[:]

	lr = load_raw.load_dta(markers,fname,channel)
	key = lr[lr['keypressed'].notna()]
	keypress_indices = [key.iloc[i].name for i in range(len(key))]
	# the intervals should go from -350 till +500, it gets trimmed by chop_idx later which should be set to 250
	intervals = [(i-600,i+600) for i in keypress_indices]
	letters = key['keypressed'].values
	return intervals,letters



# method to geneate a bunch of pictures of the setup. 
"""
input
takes a list of tuples : [(markers,fname),(markers,fname),...]
takes the channels to include : channel=[1,2,3,4]
takes the figsize
takes the name of a folder to create and save all the pngs in
takes the 

saves pictures of each letter from each of the specified files to the specified folder using a conventional naming scheme
"""
def take_pictures(fname_dta=[('./001_trial1_right_keyboard_2020-02-16-19-09-10-309.txt',
							'./001_trial1_right_keyboard_OpenBCI-RAW-2020-02-16_18-59-08.txt')],
				channel=[1,2,3,4],
				figsize=(10,12),
				folder_name='pictures'):
	channel1 = channel[:]
	# make a folder
	os.mkdir(folder_name)# throws exception if file already exists
	
	# for each of the specified files load the data and take pictures
	count1=0
	for abc in fname_dta:
		count1+=1
		markers = abc[0]
		fname = abc[1]
		# load the data, and get intervals information
		lr = load_raw.load_dta(markers,fname,channel+[13])
		explicit_intervals,letters = get_explicit_letter_intervals(markers=markers,fname=fname,channel=channel)
		count2=0
		for i,j in zip(explicit_intervals,letters):
			count2+=1
			channel=channel1[:]
			try:
				plot_ts_filtered(markers=markers,fname=fname,channel=channel,figsize=figsize,
								explicit_interval=i,
				save_fig='./'+folder_name+'/'+j+'_'+str(count1)+'_'+str(count2)+'_'+markers[2:-4]+'.png',
				disp=False)
			except: continue


# actually take the pictures, thsi method is just temporary it's to make execution faster so i dont have to go thorugh jnb

def big_shooting(channel=[1,2,3,4]):
	import os
	txt = [i for i in os.listdir() if '.txt' in i]
	dta_fnames = [i for i in os.listdir() if '.txt' in i and 'OpenBCI' in i]
	markers = [i for i in os.listdir() if '.txt' in i and 'OpenBCI' not in i]
	dta_fnames.sort()
	markers.sort()
	len(dta_fnames),len(markers)
	fname_dta = [(i,j) for i,j in zip(markers,dta_fnames)]
	fname_dta[:2]
	
	take_pictures(channel=channel, figisize=(10,12), fname_dta=fname_dta, folder_name='channels_'+str(channel))
	
	#take_pictures(channel=[1,2],figsize=(10,12),fname_dta=fname_dta,folder_name='channels_1_2')
	#take_pictures(channel=[3,4],figsize=(10,12),fname_dta=fname_dta,folder_name='channels_3_4')
	#take_pictures(channel=[1,2,3,4],figsize=(10,15),fname_dta=fname_dta,folder_name='channels_1234')
	#take_pictures(channel=[1],figsize=(10,10),fname_dta=fname_dta,folder_name='channel_1')
	#take_pictures(channel=[2],figsize=(10,10),fname_dta=fname_dta,folder_name='channel_2')
	#take_pictures(channel=[3],figsize=(10,10),fname_dta=fname_dta,folder_name='channel_3')
	#take_pictures(channel=[4],figsize=(10,10),fname_dta=fname_dta,folder_name='channel_4')







