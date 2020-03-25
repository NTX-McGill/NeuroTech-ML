import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
SAMPLING_FREQ = 250

# display spectrogram
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


# helper method - converts string time to python datetime obj
def get_ms(str_time):
  """
    Convert timestamp in keyboard markings file to unix milliseconds
    inputs:
      str_time (string)
    outputs:
      milliseconds (float)
  """
  date_time_str = '2020-02-09 ' + str_time
  date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S:%f')

  timezone = pytz.timezone('America/New_York')
  timezone_date_time_obj = timezone.localize(date_time_obj)
  return timezone_date_time_obj.timestamp() * 1000


def display_spectrogram(markers = '001_trial1_right_log_18-09-46-931825.txt',fname = '001_trial1_right_OpenBCI-RAW-2020-02-09_17-59-22.txt'):
    # load data
    df = pd.read_csv(markers)
    # find start to end
    start = df['timestamp(ms)'].iloc[0]
    end = df['timestamp(ms)'].iloc[-1]
    channel = (1,2,3,4,13)
    data = np.loadtxt(fname,
	              delimiter=',',
	              skiprows=7,
	              usecols=channel)
    eeg = data[:,:-1]
    timestamps = data[:,-1]

	start_idx = np.where(timestamps > get_ms(start))[0][0]
	end_idx = np.where(timestamps > get_ms(end))[0][0]
	markings = [get_ms(val) for val in df['timestamp(ms)'].values[::2]]
	labels = df.values[:,1]

	for idx, ch in enumerate(eeg.T):
		ch = filter_signal(ch)
		t, spec_freqs, spec_PSDperBin = get_spectral_content(ch[start_idx:start_idx + 10000], 250)
		fig=plt.figure(figsize=(8,8), dpi= 80, facecolor='w', edgecolor='k')
		plot_specgram(spec_freqs, spec_PSDperBin,'channel {}'.format(idx + 1), 0)
		for mark, label in zip(markings, labels):
		    plt.text((mark - get_ms(start))/1000,10,label, color='white')





















