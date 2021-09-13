import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
from numpy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

RawInsulinData = pd.read_csv('InsulinData.csv', low_memory=False)
RawCGMData = pd.read_csv('CGMData.csv', low_memory=False)

def format_insulin(df):
    df['Date'] = df['Date'].str.replace(' 00:00:00', '')
    df['Date and Time'] = df['Date'] + "-" + df['Time']
    df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%m/%d/%Y-%H:%M:%S')
    df = df[df['Date and Time'].notnull()]
    InsulinData = df[df['BWZ Carb Input (grams)'].notnull() & (df['BWZ Carb Input (grams)'] != 0)]
    a =InsulinData.copy()
    return a

def format_CGM(df):
    df['Date'] = df['Date'].str.replace(' 00:00:00', '')
    b = df[df['Sensor Glucose (mg/dL)'].notnull()].copy()
    b['Date and Time'] = b['Date'] + "-" + b['Time']
    b['Date and Time'] = pd.to_datetime(b['Date and Time'],format='%m/%d/%Y-%H:%M:%S')
    b = b[b['Date and Time'].notnull()]
    return b

Insulin_data = format_insulin(RawInsulinData)
CGM_data = format_CGM(RawCGMData)

def find_meal_data(Insulin_df, CGM_df):
    two_hour = pd.to_timedelta('2:00:00', unit='h')
    half_hour = pd.to_timedelta('00:30:00', unit='h')

    mealDate = []
    mealDataSet = []
    mealAmountData = []
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

        meal_df = Insulin_df[Insulin_df['Date and Time'] == k]
        mealAmountData.append(meal_df['BWZ Carb Input (grams)'].iloc[0])

    return mealDataSet,mealAmountData

meal_mat, mealAmountData = find_meal_data(Insulin_data,CGM_data)

def format_Data(n):

    n = [a[:30] for a in n if len(a) >= 30]
    n2 = [x[::-1] for x in n]

    return n2

format_meal_mat = format_Data(meal_mat)

def max_min(df):
    max_Insulin = df['BWZ Carb Input (grams)'].max()
    min_Insulin = df['BWZ Carb Input (grams)'].min()

    return max_Insulin, min_Insulin

max_Insulin,min_Insulin = max_min(Insulin_data)
mealAmountData = mealAmountData[::-1]

def create_bin(l,max_Insulin,min_Insulin):
    bin_numbers = math.ceil(max_Insulin-min_Insulin/20)
    truth = []
    for x in l:
        if (x >= min_Insulin) and (x <= 20+min_Insulin):
            truth.append(1)
        elif (x > 20+min_Insulin) and (x <= 40+min_Insulin):
            truth.append(2)
        elif (x > 60+min_Insulin) and (x <= 60+min_Insulin):
            truth.append(3)
        elif (x > 60+min_Insulin) and (x <= 80+min_Insulin):
            truth.append(4)
        elif (x > 80+min_Insulin) and (x <= 100+min_Insulin):
            truth.append(5)
        else:
            truth.append(6)

    return truth

ground_truth = create_bin(mealAmountData,max_Insulin,min_Insulin)

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

        row_feature_set = [time_diff,dg,first_p,first_peak_idx,d_avg,dd_avg]

        if len(row_feature_set) == 6:
            feature_set.append(row_feature_set)

    return feature_set

def format_data(l):
    meal = []
    for x in l:
        if len(x) >= 6:
            meal.append(x[:6])

    for y in meal:
        for i in range(len(y)):
            y[i] = (y[i]-min(y))/(max(y)-min(y))

    return meal

meal = format_data(extract_features(format_meal_mat))
meal = np.asarray(meal)

def find_measures(truth,clusters):

    df = {}
    total_entropy = 0
    total_purity = 0

    for i in range(len(clusters)):
        if clusters[i] not in df:
            df[clusters[i]] = [truth[i]]
        else:
            df[clusters[i]].append(truth[i])

    for y in df:
        df2 = []
        for j in range(len(df)):
            a = df[y].count(j)

            df2.append(a)

        g = max(df2)

        total_entropy += entropy(df2,base=2)*(len(df[y])/len(clusters))
        total_purity += g/len(clusters)

    return total_entropy, total_purity

def find_d_measures(meal,truth):
    meal = StandardScaler().fit_transform(meal)

    dbscan = DBSCAN(eps=0.13, min_samples=1).fit(meal)
    labels = dbscan.labels_.tolist()
    df = {}
    centroids = {}
    dbscan_SSE = 0

    for i in range(len(labels)):
        if labels[i] in df:
            df[labels[i]].append(i)
        else:
            df[labels[i]] = [i]

    for y in df:
        a,b,c,d,e,f,g,h = 0,0,0,0,0,0,0,0
        for z in df[y]:
            a += meal[z][0]
            b += meal[z][1]
            c += meal[z][2]
            d += meal[z][3]
            e += meal[z][4]
            f += meal[z][5]
            #g += meal[z][6]
            #h += meal[z][7]

        centroids[y] = [a/len(df[y]),b/len(df[y]),c/len(df[y]),d/len(df[y]),e/len(df[y]),f/len(df[y])]

    for h in df:
        for idx in df[h]:
            dbscan_SSE += (euclidean(meal[idx],centroids[h])**2)

    dbscan_entropy, dbscan_purity = find_measures(truth, labels)

    return dbscan_entropy, dbscan_purity, dbscan_SSE


kmeans = KMeans(n_clusters=6, random_state=0).fit(meal)
k_labels = kmeans.labels_.tolist()
kmeans_entropy, kmeans_purity = find_measures(ground_truth, k_labels)
kmeans_SSE = kmeans.inertia_

dbscan_entropy, dbscan_purity, dbscan_SSE = find_d_measures(meal, ground_truth)


print(kmeans_SSE,kmeans_entropy,kmeans_purity)
print(dbscan_SSE,dbscan_entropy,dbscan_purity)

#result = [[kmeans_SSE,dbscan_SSE,kmeans_entropy,dbscan_entropy,kmeans_purity,dbscan_purity]]
#df = pd.DataFrame(result)
#df.to_csv(r'Result.csv', index=False, header=False)

