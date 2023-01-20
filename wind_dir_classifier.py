import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from configparser import ConfigParser
from data_load import netCDF2df

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Reshape
from tensorflow import keras
from sklearn.metrics import confusion_matrix

def load_netcdf(path):
    
    config = ConfigParser()
    config.read('properties.cfg')
    latitude = config.get('dataSection', 'latitude')
    longitude = config.get('dataSection', 'longitude')
    trainFrom = int(config.get('dataSection', 'train.from'))
    trainTo = int(config.get('dataSection', 'train.to'))
    testFrom = int(config.get('dataSection', 'test.from'))
    testTo = int(config.get('dataSection', 'test.to'))
    
    years = list(range(trainFrom, trainTo+1)) + list(range(testFrom, testTo+1))
    df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.nc'):
            df = df.append(netCDF2df(path+'/'+filename),ignore_index=True)
    
    df['Date Time'] = pd.to_datetime(df.pop('Date Time'), infer_datetime_format=True)
    df['tp'] = df['tp'].fillna(0)

    df['windspeed_10'] = (df['u10']**2 + df['v10']**2)**(1/2)
    df['windspeed_100'] = (df['u100']**2 + df['v100']**2)**(1/2)

    return df

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

def to_supervised(data, n_input, n_out, y_index):
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
            y.append(data[out_end-1, y_index]) 
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y).reshape((len(y),1))

def data_prep(df, trainFrom, trainTo, testFrom, testTo, lookback, toFuture, y_featurename):

    train = df[df['Date Time']>='{}-01-01'.format(trainFrom)]
    train = train[train['Date Time']<'{}-01-01'.format(trainTo+1)]
    test = df[df['Date Time']>='{}-01-01'.format(testFrom)]
    test = test[test['Date Time']<'{}-01-01'.format(testTo+1)]

    train = datetimeToSignal(train)
    test = datetimeToSignal(test)

    y_index = train.columns.get_loc(y_featurename)

    values_train = train.values
    values_test = test.values

    y_train = values_train[:,y_index]
    y_test = values_test[:,y_index]

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values_train)
    # scaler_y = MinMaxScaler(feature_range=(0, 1))
    # scaler_y.fit(values_train[:,y_index].reshape(len(values_train),1))

    values_train = scaler.transform(values_train)
    values_test = scaler.transform(values_test)

    values_train[:,y_index] = y_train
    values_test[:,y_index] = y_test

    X_train, y_train = to_supervised(values_train, n_input=lookback, n_out=toFuture, y_index=y_index)
    X_test, y_test = to_supervised(values_test, n_input=lookback, n_out=toFuture, y_index=y_index)

    return X_train, y_train, X_test, y_test

def compile_and_fit(model, X_train, y_train, X_test, y_test):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=4,
                                                   mode='min')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], validation_data=(X_test,y_test), verbose=1)
    return history

if __name__ == "__main__":
    
    # parsing config file
    config = ConfigParser()
    config.read('properties.cfg')
    latitude = config.get('dataSection', 'latitude')
    longitude = config.get('dataSection', 'longitude')
    trainFrom = int(config.get('dataSection', 'train.from'))
    trainTo = int(config.get('dataSection', 'train.to'))
    testFrom = int(config.get('dataSection', 'test.from'))
    testTo = int(config.get('dataSection', 'test.to'))
    lookback = int(config.get('modelsSection', 'lookback'))
    hoursIntoFuture = int(config.get('modelsSection', 'noHoursPredicted'))
    include_dense = config.getboolean('modelsSection', 'dense')
    include_lstm = config.getboolean('modelsSection', 'lstm')
    path = config.get('dataSection', 'path')
    
    df = load_netcdf(path)

    df['wd_10'] = np.arctan2(df['v10'], df['u10'])
    df['wd_100'] = np.arctan2(df['v100'], df['u100'])
    df['wd_10'] = df['wd_10'].where(df['wd_10']>0, df['wd_10']+2*np.pi)
    df['wd_100'] = df['wd_100'].where(df['wd_100']>0, df['wd_100']+2*np.pi)

    bins = np.linspace(0, 2*np.pi,9)
    labels = range(8)
    df['wd_100_cat'] = pd.cut(df['wd_100'], bins=bins, include_lowest=True, labels=labels)

    df['wd_100_cat'] = df['wd_100_cat'].cat.add_categories(8)
    df['wd_100_cat'] = df['wd_100_cat'].where(df['windspeed_100']>2.5, 8)

    df.drop(labels=['u10', 'v10', 'u100', 'v100'], axis=1, inplace = True)

    # processing data for models training and evaluating
    X_train, y_train, X_test, y_test = data_prep(df, trainFrom=2014, trainTo=2020, testFrom=2021, testTo=2021, lookback=24, toFuture=1, y_featurename='wd_100_cat')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    n_outputs = y_train.shape[1]
    
    if include_dense:
        # defining models
        dense = Sequential([
            Dense(units=200, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(units=200, activation='relu'),
            Dropout(0.2),
            Dense(units=200, activation='relu'),
            Dense(n_outputs),
            Reshape([n_outputs*24]),
            Dense(n_outputs, activation='softmax')
        ])

        # training
        history = compile_and_fit(dense, X_train, y_train, X_test, y_test)

        # evaluation
        _, accuracy = dense.evaluate(X_test, y_test, batch_size=32, verbose=0)
        #Predict
        y_prediction = dense.predict(X_test)
        y_prediction = np.argmax (y_prediction, axis = 1)
        y_test_for_cm = np.argmax(y_test, axis=1)
        #Create confusion matrix and normalizes it over predicted (columns)
        cm = confusion_matrix(y_test_for_cm, y_prediction , normalize='pred')
        plot = sns.heatmap(cm, annot=True)
        plot.set(title = f'Accuracy = {accuracy}')
        fig = plot.get_figure()
        fig.savefig('figures/confusion_matrix_dense.png')

        # saving models to files
        dense.save('saved_models/wind_direction_dense.h5')

    if include_lstm:
        # defining models
        lstm = Sequential([
            LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.5),
            Dense(units=100, activation='relu'),
            Dense(n_outputs, activation='softmax')
        ])

        # training
        history = compile_and_fit(lstm, X_train, y_train, X_test, y_test)

        # evaluation
        _, accuracy = lstm.evaluate(X_test, y_test, batch_size=32, verbose=0)
        #Predict
        y_prediction = lstm.predict(X_test)
        y_prediction = np.argmax (y_prediction, axis = 1)
        y_test_for_cm = np.argmax(y_test, axis=1)
        #Create confusion matrix and normalizes it over predicted (columns)
        cm = confusion_matrix(y_test_for_cm, y_prediction , normalize='pred')
        plot = sns.heatmap(cm, annot=True)
        plot.set(title = f'Accuracy = {accuracy}')
        fig = plot.get_figure()
        fig.savefig('figures/confusion_matrix_lstm.png')

        # saving models to files
        dense.save('saved_models/wind_direction_lstm.h5')
        