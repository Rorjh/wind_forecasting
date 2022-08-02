import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from tensorflow import keras
from netcdf_to_csv import *


def datetimeToSignal(df):
    date_time = pd.to_datetime(df.pop('Date Time'), infer_datetime_format=True)

    def datetime_to_seconds(dates):
        return (float)((dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

    timestamp_s = date_time.map(datetime_to_seconds)

    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    return df

def to_supervised(data, n_input, n_out=1):
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[out_end-1, 5]) # 5 indicates 'windspeed10' parameter
        # move along one time step
        in_start += 1
    return array(X), array(y).reshape((len(y),1))

def data_prep(df, n_hours):

    train = df[df['Date Time']>='1994-01-01']
    train = train[train['Date Time']<'2002-01-01']
    test = df[df['Date Time']>='2002-01-01']
    test = test[test['Date Time']<'2003-01-01']

    train = datetimeToSignal(train)
    test = datetimeToSignal(test)

    values_train = train.values
    values_test = test.values

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values_train)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(values_train[:,5].reshape(len(values_train),1))

    values_train = scaler.transform(values_train)
    values_test = scaler.transform(values_test)

    X_train, y_train = to_supervised(values_train, n_input=24, n_out=n_hours)
    X_test, y_test = to_supervised(values_test, n_input=24, n_out=n_hours)

    return X_train, y_train, X_test, y_test, scaler_y

def compile_and_fit(model, X_train, y_train):
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=2,
                                                   mode='min')

    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stopping])

def evaluate(model, X_test, y_test, scaler_y, display = True, save_to = ''):
    # make predictions
    testPredictions = model.predict(X_test)

    # invert scalling
    testPredictions = scaler_y.inverse_transform(testPredictions)
    testYTrue = scaler_y.inverse_transform(y_test)

    # calculate RMSE
    testScore = sqrt(mean_squared_error(testYTrue, testPredictions))
    
    if display:
        print('Test Score: %.4f RMSE' % (testScore))
        length = 168
        start = random.randrange(len(testYTrue)-length)
        end = start+length
        plt.figure(figsize=[10,5])
        plt.xlabel('TimePoint in hours')
        plt.ylabel('$Windspeed_{10}[m/s]$')
        plt.grid()
        plt.plot(range(length), testYTrue[start:end],'k.')
        plt.plot(range(length),testPredictions[start:end],'r')
        plt.legend(['Actual','Predicted'])
        plt.savefig(save_to)

    return testScore, testPredictions, testYTrue

def cross_corr(model, X_test, y_test, scaler_y):
    # make predictions
    testPredictions = model.predict(X_test)

    # invert scalling
    testPredictions = scaler_y.inverse_transform(testPredictions)
    testYTrue = scaler_y.inverse_transform(y_test)
    testPredictions = testPredictions.reshape((testPredictions.shape[0]))
    testYTrue = testYTrue.reshape((testYTrue.shape[0]))

    plt.figure(figsize=[10,5])
    plt.xcorr(testYTrue, testPredictions, usevlines=False)
    plt.grid()
    plt.title("Cross correlation between predictions and actual values")


if __name__ == "__main__":
    # df = read_csv('data/data_single_loc.csv')
    df = load_netcdf('data/ERA5_single_location')
    
    X_train, y_train, X_test, y_test, scaler_y = data_prep(df, n_hours=6)

    dense = Sequential([
        Dense(units=24, activation='relu', input_shape=(24,11)),
        Dropout(0.2),
        Dense(units=24, activation='relu'),
        Dropout(0.2),
        Dense(units=24, activation='relu'),
        Dropout(0.2),
        Dense(1),
        Reshape([1,24]),
        Dense(1),
        Reshape([1])
    ])

    compile_and_fit(dense, X_train, y_train)
    testScore, testPredictions, testYTrue = evaluate(dense, X_test, y_test, scaler_y, display = True, save_to='workdir/figures/results.png')