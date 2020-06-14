import numpy as np
from scipy import signal

def notch_filter(freq=60.0, fs=250, Q=60):
    """
        Design notch filter. Outputs numerator and denominator polynomials of iir filter.
        inputs:
            freq  (float)
            order (int)
            fs    (int)
        outputs:
            (ndarray), (ndarray)
    """
    return signal.iirnotch(freq, freq / Q, fs=fs)
    
def butter_filter(low=5.0, high=50.0, order=4, fs=250):
    """
        Design butterworth filter. Outputs numerator and denominator polynomials of iir filter.
        inputs:
            low   (float)
            high  (float)
            order (int)
            fs    (int)
        outputs:
            (ndarray), (ndarray)
    """
    nyq = fs / 2
    return signal.butter(order, [low / nyq, high / nyq], 'bandpass')

def filter_signal(arr, low=5.0, high=50.0, order=4, fs=250,
                  freq=60.0, Q=60, notch=True, 
                  filter_type='original_filter', start_of_overlap=25):
    """
        Apply butterworth (and optionally notch) filter to a signal. Outputs the filtered signal.
        inputs:
            arr   (ndarray)
            notch (boolean)
            filter_type (string)
        outputs:
            (ndarray)
    """
    
    bb, ba = butter_filter(low, high, order, fs)
    nb, na = notch_filter(freq, fs, Q) 
    
    if filter_type=='original_filter':
        if notch:
            arr = signal.lfilter(nb, na, arr)
        
        return signal.lfilter(bb, ba, arr)
    
    elif filter_type=='real_time_filter':        
        #First index at which two subsequent windows overlap, same shift as 
        
        #Initial conditions of filters
        nz = signal.lfilter_zi(nb, na)
        bz = signal.lfilter_zi(bb, ba)
        
        #Filter each window sample-by-sample
        results = []
        for window in arr:
            #Initialize filtered window
            w = np.zeros(len(window))
            
            #Set intial conditions to those of the start of the window
            temp_nz, temp_bz = nz, bz
            
            #Notch filter
            for i, datum in enumerate(window):
                #signal.lfilter returns a list, so we save to a temp list to avoid a list of lists
                filtered_sample, temp_nz = signal.lfilter(nb, na, [datum], zi=temp_nz)
                w[i] = (filtered_sample[0]) 
                
                #Save initial condition for next window
                if i == start_of_overlap - 1: nz = temp_nz
            
            #Bandpass filter
            for i, datum in enumerate(w):
                filtered_sample, temp_bz = signal.lfilter(bb, ba, [datum], zi=temp_bz)
                w[i] = filtered_sample[0]
                
                #Save intial condition for next window
                if i == start_of_overlap - 1: bz = temp_bz
                
            results.append(w)
            
        return results
        
    else:
        print('\nfilter type not recognised, enter valid filter type!')
        raise Exception
        

def filter_dataframe(df, low=5.0, high=50.0, order=4, fs=250, 
                     freq=60.0, Q=60, notch=True, 
                     filter_type='original_filter', start_of_overlap=25):
    """
        Filters the signals in a dataframe.
        inputs:
            df          (DataFrame)
            filter_type (String)
        outputs:
            filtered_df (DataFrame)
    """
    filtered_df = df.copy()
    
    for col in df.columns:
        if 'channel' in col:
            filtered_df[col] = filter_signal(np.array(df[col]),
                                             low=low, high=high, order=order, fs=fs,
                                             freq=freq, Q=Q, notch=notch,
                                             filter_type=filter_type, start_of_overlap=start_of_overlap)
        
    return filtered_df
