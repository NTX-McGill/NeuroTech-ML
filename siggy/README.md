# A comprehensive list of the features used

To do : get rid of redundant features and normalised the others so that their values are all within the same ball park, i think this is much better for training. First plot a confusiong matrix, then add normalization coeficients and then plot a second confusion matrix to see if it makes significant progress, (using same data)

plot the features to figure out which ones are any good etc. as well as normalizing factor

### Time-domain features
* iemg : the sum of the absolute value of the signal - almost idential to mav
* mav : the average absolute value of the signal
* mmav : a trapezium looking weights for multiplying the absolute value of the signal - convolution like
* mmav2 : a hat shaped weights - convolution like
* var : the mean is assumed to be 0 - so this is essentially just the mean of the squares, which is just the square of rms, they are not independent!
* var_abs : this is the varience of the absolute values
* zc :  zero-crossings - the number of times it crosses the zero and the distance between the two points either side of the crossing is greater than some critical value - this function takes a threshold, we need to determine what a good threshold is
* wamp : the willison amplitude,  - for this features also we need to determine what constitutes a good threshold is
* wl : the sum of the absolute value of the differences between two successive terms
* ssch : the number of times the sign of the slope changes
* wfl : exactly the same as wl but scaled by a factor of length of window (we can get rid of this one)
* rms_3 : splits the signal into 3 equal parts and computes 3 features, rms for each part


### Frequency domain features
* freq_feats : the means of truncated intervals 5:20 , 20:40 , 40:60 , 60:80 , 80:100 , 100:120
* freq_var : the varience in fft for low, medium and high frequency wavelengths :40 , 40:80 , 80:
* freq_misc : ssch, mav, mmav, zc applied to the fourier transform







