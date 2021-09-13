import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.signal import argrelextrema
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

RawInsulinData = pd.read_csv('InsulinData.csv', low_memory=False)
RawCGMData = pd.read_csv('CGMData.csv', low_memory=False)
RawPatientInsulin = pd.read_csv('Insulin_patient2.csv', low_memory=False)
RawPatientInsulin['Date'] = RawPatientInsulin['Date'].str.replace(' 00:00:00', '')
RawPatientCGM = pd.read_csv('CGM_patient2.csv', low_memory=False)
RawPatientCGM['Date'] = RawPatientCGM['Date'].str.replace(' 00:00:00', '')

def format_insulin(df):

    df['Date and Time'] = df['Date'] + "-" + df['Time']
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    df = df[df['Date and Time'].notnull()]
    InsulinData = df[df['BWZ Carb Input (grams)'].notnull() & (df['BWZ Carb Input (grams)'] != 0)]
    a =InsulinData.copy()
    return a

def format_CGM(df):

    b = df[df['Sensor Glucose (mg/dL)'].notnull()].copy()
    b['Date and Time'] = b['Date'] + "-" + b['Time']
    b['Date and Time'] = pd.to_datetime(b['Date and Time'])
    b = b[b['Date and Time'].notnull()]
    return b


Insulin_data = format_insulin(RawInsulinData)
CGM_data = format_CGM(RawCGMData)
PatientInsulin_data = format_insulin(RawPatientInsulin)
PatientCGM_data = format_CGM(RawPatientCGM)

#find_meal_data
def find_meal_data(Insulin_df, CGM_df):
    two_hour = pd.to_timedelta('2:00:00', unit='h')
    half_hour = pd.to_timedelta('00:30:00', unit='h')

    mealDate = []
    mealDataSet = []

    for i in range(len(Insulin_df)-1, -1,-1):
        if i == 0:
            mealDate.append(Insulin_df['Date and Time'].iloc[0])
        elif Insulin_df['Date and Time'].iloc[i] + two_hour < Insulin_df['Date and Time'].iloc[i-1]:
            mealDate.append(Insulin_df['Date and Time'].iloc[i])

    for k in mealDate:
        CGMData = []
        CGMDataSet = CGM_df.loc[(CGM_df['Date and Time'] > k-half_hour) & (CGM_df['Date and Time'] < k+two_hour)]
        for l in CGMDataSet['Sensor Glucose (mg/dL)']:
            CGMData.append(l)

        mealDataSet.append(CGMData)

    return mealDataSet

## find no meal data

def find_nomeal_data(RawInsulin_df,CGM_df):
    two_hour = pd.to_timedelta('2:00:00', unit='h')
    noMealDataSet = []

    j = RawInsulin_df['Date and Time'].iloc[-1]

    while j <= RawInsulin_df['Date and Time'].iloc[0]:
        noMealData = []
        end_time = j+two_hour

        df = RawInsulin_df[(RawInsulin_df['Date and Time'] >= j) & (RawInsulin_df['Date and Time'] < end_time)]
        df2 = df[(df['BWZ Carb Input (grams)'].notnull()) & (df['BWZ Carb Input (grams)'] != 0)]

        if len(df2) == 0:
            CGMDataSet = CGM_df[(CGM_df['Date and Time'] >= j) & (CGM_df['Date and Time'] <= end_time)]

            for x in CGMDataSet['Sensor Glucose (mg/dL)']:
                noMealData.append(x)

            noMealDataSet.append(noMealData)
            j = end_time

        else:
            idx = df2.index[0]
            j = RawInsulin_df['Date and Time'].iloc[idx] + two_hour

    return noMealDataSet

meal_mat = find_meal_data(Insulin_data,CGM_data)
nomeal_mat = find_nomeal_data(RawInsulinData,CGM_data)
patient_meal_mat = find_meal_data(PatientInsulin_data, PatientCGM_data)
patient_nomeal_mat = find_nomeal_data(RawPatientInsulin, PatientCGM_data)

def format_Data(n,m):

    n = [a[6:30] for a in n if len(a) >= 30]
    m = [b[:24] for b in m if len(b) >= 24]

    n2 = [x[::-1] for x in n]
    m2 = [y[::-1] for y in m]
    return n2,m2

format_meal_mat,format_nomeal_mat = format_Data(meal_mat,nomeal_mat)
patient_format_meal_mat, patient_format_nomeal_mat = format_Data(patient_meal_mat,patient_nomeal_mat)


##test = (np.abs(fft(np.asarray(format_Meal[10]))))**2
##peaks = argrelextrema(test,np.greater)

##print(test[peaks[0]], test[3])


##print(test)
##print(peaks)

##xx = fftfreq(len(test),1/30)

##plt.figure(1)
##plt.plot(test)
##plt.show()

'''
def extract_features(n):
    feature_set = []

    for x in n:
        row_feature_set = []

        if len(x) == 30:
            max_idx = np.argmax(x[7:], axis=0) + 7
            time_diff = (max_idx - 6)
            dg = (x[max_idx] - x[6]) / x[6]
            dy = np.diff(x[6:])
            dd = np.diff(dy)
            d_avg = max(dy)
            dd_avg = max(dd)
        else:
            max_idx = np.argmax(x[1:], axis=0) + 1
            time_diff = (max_idx)
            dg = (x[max_idx] - x[0]) / x[0]
            dy = np.diff(x)
            dd = np.diff(dy)
            d_avg = max(dy)
            dd_avg = max(dd)

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

        row_feature_set = [time_diff,dg,first_p,first_peak_idx,second_p,second_peak_idx,d_avg,dd_avg]

        if len(row_feature_set) == 8:
            feature_set.append(row_feature_set)

    return feature_set


meal_feature_set = extract_features(format_meal_mat)
nomeal_feature_set = extract_features(format_nomeal_mat)
patient_meal_feature_set = extract_features(patient_format_meal_mat)
patient_nomeal_feature_set = extract_features(patient_format_nomeal_mat)
'''
def create_set(meal_df,nomeal_df,patient_meal_df, patient_nomeal_df):
    meal_df = np.asarray(meal_df)
    nomeal_df = np.asarray(nomeal_df)
    patient_meal_df = np.asarray(patient_meal_df)
    patient_nomeal_df = np.asarray(patient_nomeal_df)

    meal_label = np.ones(len(meal_df))
    nomeal_label = np.zeros(len(nomeal_df))
    patient_meal_label = np.ones(len(patient_meal_df))
    patient_nomeal_label = np.zeros(len(patient_nomeal_df))

    feature_set = np.vstack((meal_df,patient_meal_df,nomeal_df,patient_nomeal_df))
    label_set = np.hstack((meal_label,patient_meal_label,nomeal_label,patient_nomeal_label))


    return feature_set,label_set

x,y = create_set(format_meal_mat,format_nomeal_mat,patient_format_meal_mat,patient_format_nomeal_mat)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


clf = svm.SVC(kernel='poly',gamma='auto',C=1)
#clf = MLPClassifier(random_state=0,max_iter=300)
clf.fit( x_train, y_train)
predict = clf.predict(x_test)
train = clf.score(x_train, y_train)
test = clf.score(x_test,y_test)
f1 = f1_score(y_test,predict)
precision = precision_score(y_test,predict)
recall = recall_score(y_test, predict)
pickle.dump(clf, open('model', 'wb'))