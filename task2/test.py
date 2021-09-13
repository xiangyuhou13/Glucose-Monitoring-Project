import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import argrelextrema
from sklearn import svm
import pickle

test_data = pd.read_csv('test.csv',header=None, low_memory=False)
test_data_to_list = test_data.to_numpy().tolist()

def extract_features(n):
    feature_set = []

    for x in n:
        max_idx = np.argmax(x[1:], axis=0) + 1

        time_diff = max_idx

        dg = (x[max_idx] - x[0])/x[0]

        fft_data = np.abs(fft(np.asarray(x)))
        peaks = np.asarray(argrelextrema(fft_data, np.greater))

        if len(peaks[0]) >= 2:
            first_peak_idx = peaks[0][0]
            second_peak_idx = peaks[0][1]
            first_p = fft_data[first_peak_idx]
            second_p = fft_data[second_peak_idx]

        elif len(peaks[0] == 1):
            first_peak_idx = peaks[0][0]
            second_peak_idx = 0
            first_p = fft_data[first_peak_idx]
            second_p = 0
        else:
            first_peak_idx = 0
            second_peak_idx = 0
            first_p = 0
            second_p = 0

        dy = np.diff(x)
        dd = np.diff(dy)
        d_avg = max(dy)
        dd_avg = max(dd)


        feature_set.append([time_diff,dg,first_p,first_peak_idx,second_p,second_peak_idx,d_avg,dd_avg])
    return feature_set



feature_set = np.asarray(test_data_to_list)

clf = pickle.load(open('model', 'rb'))
result = clf.predict(feature_set)

df = pd.DataFrame(result)
df.to_csv(r'Result.csv', index=False, header=False)
