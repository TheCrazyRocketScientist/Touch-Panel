import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt




def set_df(df):
    df.index = pd.to_datetime(df.timestamp)
    df.drop(labels='timestamp',axis=1,inplace=True)

    for col in df:
        df[col] = df[col]-np.mean(df[col])
        df[col] = df[col]/np.std(df[col])
    return df


def chop_label_ts(df,region,axis,window_size,overlap):

    sensor_taps = []
    stride = window_size*overlap


    
    std_dev = np.std(df[axis])
    df.loc[:, 'taps'] = (~((df[axis] <= 2*std_dev)&(df[axis] >= -2*std_dev))).astype(int) 

    old_time = df.index[0]
    new_time = df.index[0]+datetime.timedelta(seconds=window_size)

    my_data = []
    labels = []

    #label all points which are greater than some set multiple of standard deviation of the time series
    #create a 2d array which breaks the ts into window size with label
    #also capture residual artifact after tap event for given number of windows.

    while new_time < df.index[-1]:

        window = df[old_time:new_time]['taps']
        window_data = df[old_time:new_time][axis].to_list()

        if(window.empty):
            break

        if(1 in window.values):

            sensor_taps.append(old_time+datetime.timedelta(seconds=stride))
            label = region
        else:
            label = 0 # 0 is the label for noise
        
        my_data.append(window_data)
        labels.append(label)

        new_time += datetime.timedelta(seconds=stride)
        old_time += datetime.timedelta(seconds=stride)
    
    return (my_data,labels,sensor_taps)

def label_artifacts(labels,region,window_artifact_number):
    
    add_label = 0

    for x in range(len(labels)):
        
        if(labels[x] != 0):
            add_label = window_artifact_number

        elif add_label > 0:
            labels[x] = region
            add_label -= 1
        
    return labels

def get_time_intervals(sensor_taps):

    tap_intervals = []

    last_reading = sensor_taps[0]

    for x in range(1,len(sensor_taps)):

        readings_diff = sensor_taps[x]-sensor_taps[x-1]

        if(readings_diff > datetime.timedelta(seconds=stride)):

            interval = [last_reading,sensor_taps[x-1]]
            last_reading = sensor_taps[x]

            tap_intervals.append(interval)

    tap_intervals.append([last_reading, sensor_taps[-1]])

    return tap_intervals


def resample_to_50hz(df):
    # Resample to 50Hz → one sample every 20 ms
    df_resampled = df.resample('20ms').mean()

    # Optionally: interpolate missing values if needed
    df_resampled = df_resampled.interpolate(method='linear')

    return df_resampled


def get_train_data(window_size,overlap):

    train_data = []
    train_labels = []

    validation_data = []
    validation_labels = []

    test_data = []
    test_labels = []

    for region in range(1,5):

        df_sensor_one = pd.read_csv(rf"E:\Projects\Vibration Sensing Touch Panel\data\region{region}_extended\SENSOR1_data.csv")
        df_sensor_two = pd.read_csv(rf"E:\Projects\Vibration Sensing Touch Panel\data\region{region}_extended\SENSOR2_data.csv")
        df_sensor_three = pd.read_csv(rf"E:\Projects\Vibration Sensing Touch Panel\data\region{region}_extended\SENSOR3_data.csv")


        df_sensor_one = set_df(df_sensor_one)
        df_sensor_two = set_df(df_sensor_two)
        df_sensor_three = set_df(df_sensor_three)

        df_sensor_one = resample_to_50hz(df_sensor_one)
        df_sensor_two = resample_to_50hz(df_sensor_two)
        df_sensor_three = resample_to_50hz(df_sensor_three)

        recording_start = max(df_sensor_three.index[0],df_sensor_two.index[0],df_sensor_one.index[0])
        recording_end = min(df_sensor_three.index[-1],df_sensor_two.index[-1],df_sensor_one.index[-1])

        df_sensor_one = df_sensor_one[recording_start:recording_end]
        df_sensor_two = df_sensor_two[recording_start:recording_end]
        df_sensor_three = df_sensor_three[recording_start:recording_end]

        #perform chronological splitting
        duration = recording_end-recording_start
        train_time = 0.70*duration.total_seconds()
        test_time = 0.15*duration.total_seconds()

        df_train_one = df_sensor_one[recording_start:(recording_start+pd.Timedelta(seconds=train_time))].copy()
        df_train_two = df_sensor_two[recording_start:(recording_start+pd.Timedelta(seconds=train_time))].copy()
        df_train_three = df_sensor_three[recording_start:(recording_start+pd.Timedelta(seconds=train_time))].copy()
        recording_start = (recording_start+pd.Timedelta(seconds=train_time))

        df_validation_one = df_sensor_one[recording_start:(recording_start+pd.Timedelta(seconds=test_time))].copy()
        df_validation_two = df_sensor_two[recording_start:(recording_start+pd.Timedelta(seconds=test_time))].copy()
        df_validation_three = df_sensor_three[recording_start:(recording_start+pd.Timedelta(seconds=test_time))].copy()
        recording_start = (recording_start+pd.Timedelta(seconds=test_time))


        df_test_one = df_sensor_one[recording_start:recording_end].copy()
        df_test_two = df_sensor_two[recording_start:recording_end].copy()
        df_test_three = df_sensor_three[recording_start:recording_end].copy()

        data_train_sensor_one,train_labels_one,_ = chop_label_ts(df_train_one,region,'z',window_size,overlap)
        data_train_sensor_two,train_labels_two,_ = chop_label_ts(df_train_two,region,'z',window_size,overlap)
        data_train_sensor_three,train_labels_three,_ = chop_label_ts(df_train_three,region,'z',window_size,overlap)

        data_validation_sensor_one,validation_labels_one,_ = chop_label_ts(df_validation_one,region,'z',window_size,overlap)
        data_validation_sensor_two,validation_labels_two,_ = chop_label_ts(df_validation_two,region,'z',window_size,overlap)
        data_validation_sensor_three,validation_labels_three,_ = chop_label_ts(df_validation_three,region,'z',window_size,overlap)


        data_test_sensor_one,test_labels_one,_ = chop_label_ts(df_test_one,region,'z',window_size,overlap)
        data_test_sensor_two ,test_labels_two,_ = chop_label_ts(df_test_two,region,'z',window_size,overlap)
        data_test_sensor_three ,test_labels_three,_ = chop_label_ts(df_test_three,region,'z',window_size,overlap)


        train_data_one_two_three = np.stack([data_train_sensor_one,data_train_sensor_two,data_train_sensor_three],axis=-1)

        validation_data_one_two_three = np.stack([data_validation_sensor_one,data_validation_sensor_two,data_validation_sensor_three],axis=-1)

        test_data_one_two_three = np.stack([data_test_sensor_one,data_test_sensor_two,data_test_sensor_three],axis=-1)

        train_data_labels = [max(a,b,c) for a,b,c in zip(train_labels_one,train_labels_two,train_labels_three)] 
        validation_data_labels = [max(a,b,c) for a,b,c in zip(validation_labels_one,validation_labels_two,validation_labels_three)]
        test_data_labels = [max(a,b,c) for a,b,c in zip(test_labels_one,test_labels_two,test_labels_three)]


        train_data.append(train_data_one_two_three)
        train_labels.append(train_data_labels)

        validation_data.append(validation_data_one_two_three)
        validation_labels.append(validation_data_labels)

        test_data.append(test_data_one_two_three)
        test_labels.append(test_data_labels)

    train_data = np.concatenate(train_data,axis=0)
    train_labels = np.concatenate(train_labels,axis=0)

    validation_data = np.concatenate(validation_data,axis=0)
    validation_labels = np.concatenate(validation_labels,axis=0)

    test_data = np.concatenate(test_data,axis=0)
    test_labels = np.concatenate(test_labels,axis=0)

    return train_data,validation_data,test_data,train_labels,validation_labels,test_labels

if( __name__ == "__main__"):
    _,labels = get_train_data(window_size=0.6,overlap=1.0)
    print(np.unique_counts(labels))