from featurize import compute_features, all_names
from features import * # important to import these, but you still need to redefine some individually for plotting purposes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
import pickle
from datetime import datetime
import matplotlib.mlab as mlab
from siggy.constants import *


ALL_FEATURES = ['iemg','mav','mmav','var','rms','rms_3','zc','wamp','wl','ssch','wfl','freq_feats','freq_var']
LABEL_MAP = {'k': 3, ';':5, 'j': 2, 'l': 4, 'p': 5, 'u': 2, 'o':4, '.': 4,
          'm':2, 'n': 2, '[':5, ']': 5, "'": 5, 'h': 2, '/':5, '\\':5,
          'a':10, 'c': 8, 'f': 7, 's': 9, 'd':8, 'e':8, 'g':7, 'q':10, 'r':7, 't':7, 'v':7, 'w':9, 'x':9, 'z':10
          }
SEED = 7


def load_windows(filename, channels):
    channel_names = ['channel {}'.format(i) for i in channels]
    # reads file
    df = pd.read_pickle(filename)
    df.reset_index(inplace=True)
    # set labels
    df['keypressed'] = df['keypressed'].map(LABEL_MAP)
    print("Key press values: {}".format(df['keypressed'].unique()))
    return df


#%%
def display_subplots(df,chname='channel 3',hand=1.0,finger=5.0):
    '''
	a function that plots a bunch of little windows
	'''
    # refine the data-frame so that it's only the ones relating to a certain finger
    channels = df[(df['hand']==hand) & (df['finger']==finger)]
    fig = plt.subplots(figsize=(20,60))
    for i in range(40):
        a = channels[chname].iloc[i+1]
        plt.subplot(20,4,2*i+1)
        plt.plot(a)
        plt.title('number '+str(i)+' time dom',fontsize=12)

        psd,freqs = get_fft(a)
        plt.subplot(20,4,2*i+2)
        plt.plot(freqs,psd)
        plt.title('number '+str(i)+' freq dom',fontsize=12)
    plt.show()
	
    


#%% 
### some utility funcitons:

# function that returns boolean, returns True if the signal is a real signal, 
# returns False if the signal is just noise
# it's important to throw things away, so if there is any doubt that it's just noise, label it as noise...
def has_content(signal):
    m,r,v = mav(signal),rms(signal),var(signal)
    # the thresholds have been determined empirically (see jupyter notebook visualise pickle files.ipynb)
    # i cut it so that they were approx 3 stds away from being bad
    # thesh mav = 2.8 ; thresh rms = 4.0 ; thresh var = 6.5
    if m < 3.5 or r < 10.0 or v < 120 : return False
    return True# DEPRICATED, THESE VALUES ARE NO GOOD, ONLY WORK IN SPECIFIC CIRCUMSTANCES

# takes a data frame and chucks rows which are baseline
# returns data frame without baseline rows
def df_drop_baseline(df):
    return df[df['finger']!=0 & df['keypressed'].isnull()]
    
# takes a windows data fram and chucks rows which are labelled 
# so that what remains is only baseline
def keep_only_baseline(df):
    return df[(df['keypressed'].isnull()) & (df['finger']==0) & (df['hand'].isnull())]



# calculate the mavs and vars for each of the channels 
# then get mean and std of these, store in new df

# takes two data-frames and returns meta information
def get_meta(baseline,no_baseline):
    channels=[1,2,3,4,5,6,7,8]
    meta_no_base,meta_baseline = pd.DataFrame(),pd.DataFrame()
    for i in channels:
        channel = 'channel '+str(i)
        # baseline
        mavs_i = [mav(w) for w in baseline[channel]]
        vars_i = [var(w) for w in baseline[channel]]
        meta_baseline[channel]=[np.mean(mavs_i),np.std(mavs_i),np.mean(vars_i),np.std(vars_i)]

        # no basline
        mavs_i = [mav(w) for w in no_baseline[channel]]
        vars_i = [var(w) for w in no_baseline[channel]]
        meta_no_base[channel]=[np.mean(mavs_i),np.std(mavs_i),np.mean(vars_i),np.std(vars_i)]

    # rename the rows
    meta_baseline.rename(index={0:'mav mean',1:'mav std',
                       2:'var mean',3:'var std'},inplace=True)
    meta_no_base.rename(index={0:'mav mean',1:'mav std',
                       2:'var mean',3:'var std'},inplace=True)
    return meta_baseline,meta_no_base



#%%
# get_psd is already imported from features
# single features for plotting purposes, so is get_bands() 
def freq_band_0(signal):
    return np.mean(get_bands(get_psd(signal))[0])
def freq_band_1(signal):
    return np.mean(get_bands(get_psd(signal))[1])
def freq_band_2(signal):
    return np.mean(get_bands(get_psd(signal))[2])
def freq_band_3(signal):
    return np.mean(get_bands(get_psd(signal))[3])
def freq_band_4(signal):
    return np.mean(get_bands(get_psd(signal))[4])
def freq_band_5(signal):
    return np.mean(get_bands(get_psd(signal))[5])


def get_fft(signal):
    psd,freqs = mlab.psd(signal,NFFT=256,window=mlab.window_hanning,Fs=250,noverlap=0)
    return psd,freqs


#%%
### the main plotting funciton, please don't judge my variable names, the only thing you need to know is what goes in and what comes out... 
def display_by_subject_id(df,fn,str_fn,fig,modes=[1,2,4],
                          mean=True,thresh=15,
                          valid_ids=[1,2,3,4,5,6,7,8,9,10,11,12]):
    baseline = keep_only_baseline(df)
    """
    parameters
   		df : a data-frame contianting windowed data (i use it for baseline data)
   		fn : the feature function
   		str_fn : a string representation of the feature function for title of subplot
   		fig : an instance of pytplot.plot.figure
   		mean : True if you plot the mean, if False will plot the median value
   		thresh : some of the data has HIGH uncertainty, the threshold is a tolarance level for this uncertainty, without it things would get very ugly
   		valid_ids : plots only those ids which are listed, this is so that we can add new subject's data when it comes in
   
   	output
   		doesn't return anything but i think the correct 'compsci' way to say it is it edits the attributes of figure, basically you call it once and it'll plot something in one of your subplots; if you want to plot a bunch of plots in a grid of subplots you can call it many times.
   	"""
    
    # filter the df by modes and baseline
    baseline = baseline[baseline['mode'].isin(modes)]
    
    ids = np.unique(baseline['id'].values)
    print(ids)
    
    
    means_by_channel = []
    medians_by_channel = []
    std_by_channel = []
    for ch in [1,2,3,4,5,6,7,8]:
        means_ch = []
        medians_ch = []
        std_ch = []
        for i in valid_ids:
            things = [fn(w) for w in baseline[baseline['id']==i]['channel '+str(ch)].values]
            means_ch.append(np.mean(things))
            # the output is continuous, so we discretise
            medians_ch.append(np.median([int(t) for t in things]))
            uncert = np.std(things) / np.sqrt(len(things))
            std_ch.append(uncert)# uncertainty in the mean
            
            print('\nthe uncertainty in the mean for channel '+str(ch)+' id '+str(i)+' is',uncert)
            print('and the number of data points is ',len(things))
            
            # only plot things that are relatively certain
            for n,i in enumerate(std_ch):
                if i>thresh:
                    means_ch[n]=0
                    medians_ch[n]=0
                    std_ch[n]=0
        means_by_channel.append(means_ch)
        medians_by_channel.append(medians_ch)
        std_by_channel.append(std_ch)
        if mean==True:
            plt.errorbar(valid_ids,means_ch,
                         yerr=std_ch,fmt='x',label='ch='+str(ch))
        else:
            plt.errorbar(valid_ids,medians_ch,
                         yerr=std_ch,fmt='x',label='ch='+str(ch))
            
    
    plt.legend()
    tit='medians'# beg of title
    if mean==True:tit='means'
    plt.title(tit+' '+str_fn+' in baseline by subject id',fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
    plt.grid()
    plt.ylabel(tit+' '+str_fn+' by channel',fontsize=12)
    plt.xlabel('subject id',fontsize=12)
    




if __name__ == '__main__':
	window_size = 2
	channels = [1,2,3,4,5,6,7,8]
	filename = 'windows_date_all_subject_all_mode_1_2_4.pkl'
	file_prefix = filename.split(".")[0]
	channel_names = ['channel {}'.format(i) for i in channels]
	
	df = load_windows(filename, channels)
	baseline = keep_only_baseline(df)
	no_baseline = df_drop_baseline(df)
	
	# plot a nice figure display some useful information
	fig,ax = plt.subplots(3,3,figsize=(15,15))
	
	plt.subplot(3,3,1)
	display_by_subject_id(df=baseline,fn=mav,str_fn='mav',fig=fig,thresh=15)
	
	plt.subplot(3,3,2)
	display_by_subject_id(df=baseline,fn=var,str_fn='var',fig=fig,thresh=700)
	
	plt.subplot(3,3,3)
	display_by_subject_id(df=baseline,fn=rms,str_fn='rms',fig=fig,thresh=5)
	
	plt.subplot(3,3,4)
	display_by_subject_id(df=baseline,fn=freq_band_0,str_fn='freq band 0',fig=fig,thresh=5)
	
	plt.subplot(3,3,5)
	display_by_subject_id(df=baseline,fn=freq_band_1,str_fn='freq band 1',fig=fig,thresh=5)
	
	plt.subplot(3,3,6)
	display_by_subject_id(df=baseline,fn=freq_band_2,
		                  str_fn='freq band 2',fig=fig,thresh=5)
	
	plt.subplot(3,3,7)
	display_by_subject_id(df=baseline,fn=freq_band_3,
		                  str_fn='freq band 3',fig=fig,thresh=5)
	
	plt.subplot(3,3,8)
	display_by_subject_id(df=baseline,fn=freq_band_4,
		                  str_fn='freq band 4',fig=fig,thresh=5)
	
	plt.subplot(3,3,9)
	display_by_subject_id(df=baseline,fn=freq_band_5,
		                  str_fn='freq band 5',fig=fig,thresh=5)
	
	
	fig.tight_layout()
	# plt.savefig('asdfasdfasdfdjdkdkkddk.png')
	plt.show()
	
	
	
