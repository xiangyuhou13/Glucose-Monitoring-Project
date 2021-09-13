import pandas as pd
import numpy as np

CGMData = pd.read_csv('CGMData.csv', low_memory=False,)
InsulinData = pd.read_csv('InsulinData.csv', low_memory=False)

filtered_CGMData = CGMData[CGMData['Sensor Glucose (mg/dL)'].notnull()]

def find_auto_swtich_time(data):

    auto_switch_time = data[data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']
    auto_switch_copy = auto_switch_time.copy()
    auto_switch_copy["Date and Time"] = auto_switch_time['Date'] + "-" + auto_switch_time['Time']
    auto_switch_copy['Date and Time'] = pd.to_datetime(auto_switch_copy['Date and Time'], format='%m/%d/%Y-%H:%M:%S')

    return auto_switch_copy.iloc[-1]['Date and Time']


def find_date(data):

    CGMData_by_date = {}
    removed_date = []

    for date in data['Date']:
        if date in CGMData_by_date:
            CGMData_by_date[date] += 1
        else:
            CGMData_by_date[date] = 1

    for date in CGMData_by_date:
        if CGMData_by_date[date] < 26:
            removed_date.append(date)

    return removed_date, CGMData_by_date

removed_date, CGMData_by_date = find_date(filtered_CGMData)
auto_switch_date = find_auto_swtich_time(InsulinData)

CGMData_with_removed_date = filtered_CGMData[~filtered_CGMData['Date'].isin(removed_date)]

a = CGMData_with_removed_date.copy()
a['Date and Time'] = a['Date'] + "-" + a['Time']
a['Date and Time'] = pd.to_datetime(a['Date and Time'], format='%m/%d/%Y-%H:%M:%S')
a['Time'] = pd.to_datetime(a['Time'], format='%H:%M:%S').dt.time

day_start = pd.to_datetime('6:00:00', format='%H:%M:%S').time()

auto_data = a[a['Date and Time'] >= auto_switch_date]
manual_data = a[a['Date and Time'] < auto_switch_date]
auto_data_day = auto_data[auto_data['Time'] >= day_start]
auto_data_night = auto_data[auto_data['Time'] < day_start]
manual_data_day = manual_data[manual_data['Time'] >= day_start]
manual_data_night = manual_data[manual_data['Time'] < day_start]

'''
def extract_data(data,total_data):
    data.set_index('Date and Time', inplace=True)
    hyperglycemia_df = data[data['Sensor Glucose (mg/dL)'] > 180.0]
    hyperglycemia_critical_df = data[data['Sensor Glucose (mg/dL)'] > 250.0]
    in_range_df = data[(data['Sensor Glucose (mg/dL)'] >= 70.0) & (data['Sensor Glucose (mg/dL)'] <= 180.0)]
    in_range_secondary_df = data[(data['Sensor Glucose (mg/dL)'] >= 70.0) & (data['Sensor Glucose (mg/dL)'] <= 150.0)]
    hypoglycemia_level1_df = data[data['Sensor Glucose (mg/dL)'] < 70.0]
    hypoglycemia_level2_df = data[data['Sensor Glucose (mg/dL)'] < 54.0]

    hyperglycemia = []
    hyperglycemia_critical = []
    in_range = []
    in_range_secondary = []
    hypoglycemia_level1 = []
    hypoglycemia_level2 = []

    data_date = []

    for date in total_data['Date']:
        if date not in data_date:
            data_date.append(date)

    for date in data_date:
        if date in hyperglycemia_df['Date']:
            hyperglycemia.append((hyperglycemia_df[date]['Sensor Glucose (mg/dL)'].count())/288)
        else:
            hyperglycemia.append(0)
    for date in data_date:
        if date in hyperglycemia_critical_df['Date']:
            hyperglycemia_critical.append((hyperglycemia_critical_df[date]['Sensor Glucose (mg/dL)'].count())/288)
        else:
            hyperglycemia_critical.append(0)
    for date in data_date:
        if date in in_range_df['Date']:
            in_range.append((in_range_df[date]['Sensor Glucose (mg/dL)'].count())/288)
        else:
            in_range.append(0)
    for date in data_date:
        if date in in_range_secondary_df['Date']:
            in_range_secondary.append((in_range_secondary_df[date]['Sensor Glucose (mg/dL)'].count())/288)
        else:
            in_range_secondary.append(0)
    for date in data_date:
        if date in hypoglycemia_level1_df['Date']:
            hypoglycemia_level1.append((hypoglycemia_level1_df[date]['Sensor Glucose (mg/dL)'].count())/288)
        else:
            hypoglycemia_level1.append(0)
    for date in data_date:
        if date in hypoglycemia_level2_df['Date']:
            hypoglycemia_level2.append((hypoglycemia_level2_df[date]['Sensor Glucose (mg/dL)'].count())/288)
        else:
            hypoglycemia_level2.append(0)


    return np.mean(hyperglycemia),np.mean(hyperglycemia_critical),np.mean(in_range),np.mean(in_range_secondary),np.mean(hypoglycemia_level1),np.mean(hypoglycemia_level2)
'''
def extract_data2(data):
    data.set_index('Date and Time', inplace=True)

    hyperglycemia = {}
    hyperglycemia_critical = {}
    in_range = {}
    in_range_secondary = {}
    hypoglycemia_level1 = {}
    hypoglycemia_level2 = {}


    for date in data['Date']:
        if date not in hyperglycemia:
            hyperglycemia[date] = 0
            hyperglycemia_critical[date] = 0
            in_range[date] = 0
            in_range_secondary[date] = 0
            hypoglycemia_level1[date] = 0
            hypoglycemia_level2[date] = 0

    for date in hyperglycemia:
        for x in data.loc[date,'Sensor Glucose (mg/dL)']:
            if x > 180.0:
                hyperglycemia[date] += 1
            if x > 250.0:
                hyperglycemia_critical[date] += 1
            if x >= 70.0 and x <= 180.0:
                in_range[date] += 1
            if x >= 70.0 and x <= 150.0:
                in_range_secondary[date] += 1
            if x < 70.0:
                hypoglycemia_level1[date] += 1
            if x < 54.0:
                hypoglycemia_level2[date] += 1

    for date in hyperglycemia:
        hyperglycemia[date] = hyperglycemia[date] / 288.0 * 100
        hyperglycemia_critical[date] = hyperglycemia_critical[date] / 288.0 * 100
        in_range[date] = in_range[date] / 288.0 * 100
        in_range_secondary[date] = in_range_secondary[date] / 288.0 * 100
        hypoglycemia_level1[date] = hypoglycemia_level1[date] / 288.0 * 100
        hypoglycemia_level2[date] = hypoglycemia_level2[date] / 288.0 * 100

    hyperglycemia_sum = 0
    hyperglycemia_critical_sum = 0
    in_range_sum = 0
    in_range_secondary_sum = 0
    hypoglycemia_level1_sum = 0
    hypoglycemia_level2_sum = 0

    for date in hyperglycemia:
        hyperglycemia_sum += hyperglycemia[date]
        hyperglycemia_critical_sum += hyperglycemia_critical[date]
        in_range_sum += in_range[date]
        in_range_secondary_sum += in_range_secondary[date]
        hypoglycemia_level1_sum += hypoglycemia_level1[date]
        hypoglycemia_level2_sum += hypoglycemia_level2[date]

    hyperglycemia_mean = hyperglycemia_sum / len(hyperglycemia)
    hyperglycemia_critical_mean = hyperglycemia_critical_sum/ len(hyperglycemia_critical)
    in_range_mean = in_range_sum / len(in_range)
    in_range_secondary_mean  = in_range_secondary_sum / len(in_range_secondary)
    hypoglycemia_level1_mean  = hypoglycemia_level1_sum / len(hypoglycemia_level1)
    hypoglycemia_level2_mean  = hypoglycemia_level2_sum / len(hypoglycemia_level2)

    return hyperglycemia_mean,hyperglycemia_critical_mean,in_range_mean,in_range_secondary_mean,hypoglycemia_level1_mean,hypoglycemia_level2_mean



whole_day_auto_hyperglycemia,whole_day_auto_hyperglycemia_critical,whole_day_auto_in_range,whole_day_auto_in_range_secondary,whole_day_auto_hypoglycemia_level1,whole_day_auto_hypoglycemia_level2 = extract_data2(auto_data)
whole_day_manual_hyperglycemia,whole_day_manual_hyperglycemia_critical,whole_day_manual_in_range,whole_day_manual_in_range_secondary,whole_day_manual_hypoglycemia_level1,whole_day_manual_hypoglycemia_level2 = extract_data2(manual_data)
day_auto_hyperglycemia,day_auto_hyperglycemia_critical,day_auto_in_range,day_auto_in_range_secondary,day_auto_hypoglycemia_level1,day_auto_hypoglycemia_level2 = extract_data2(auto_data_day)
night_auto_hyperglycemia,night_auto_hyperglycemia_critical,night_auto_in_range,night_auto_in_range_secondary,night_auto_hypoglycemia_level1,night_auto_hypoglycemia_level2 = extract_data2(auto_data_night)
day_manual_hyperglycemia,day_manual_hyperglycemia_critical,day_manual_in_range,day_manual_in_range_secondary,day_manual_hypoglycemia_level1,day_manual_hypoglycemia_level2 = extract_data2(manual_data_day)
night_manual_hyperglycemia,night_manual_hyperglycemia_critical,night_manual_in_range,night_manual_in_range_secondary,night_manual_hypoglycemia_level1,night_manual_hypoglycemia_level2 = extract_data2(manual_data_night)

final_data = [[night_manual_hyperglycemia,night_manual_hyperglycemia_critical,night_manual_in_range,night_manual_in_range_secondary,night_manual_hypoglycemia_level1,night_manual_hypoglycemia_level2,\
               day_manual_hyperglycemia,day_manual_hyperglycemia_critical,day_manual_in_range,day_manual_in_range_secondary,day_manual_hypoglycemia_level1,day_manual_hypoglycemia_level2, \
               whole_day_manual_hyperglycemia,whole_day_manual_hyperglycemia_critical,whole_day_manual_in_range,whole_day_manual_in_range_secondary,whole_day_manual_hypoglycemia_level1,whole_day_manual_hypoglycemia_level2],\
              [night_auto_hyperglycemia,night_auto_hyperglycemia_critical,night_auto_in_range,night_auto_in_range_secondary,night_auto_hypoglycemia_level1,night_auto_hypoglycemia_level2, \
               day_auto_hyperglycemia,day_auto_hyperglycemia_critical,day_auto_in_range,day_auto_in_range_secondary,day_auto_hypoglycemia_level1,day_auto_hypoglycemia_level2, \
               whole_day_auto_hyperglycemia,whole_day_auto_hyperglycemia_critical,whole_day_auto_in_range,whole_day_auto_in_range_secondary,whole_day_auto_hypoglycemia_level1,whole_day_auto_hypoglycemia_level2]]

df = pd.DataFrame(final_data)

df.to_csv(r'Results.csv', index=False, header=False)
