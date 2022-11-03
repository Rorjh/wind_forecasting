import sys
import pandas as pd
from models import datetimeToSignal
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def data_prep(df):
    df = datetimeToSignal(df)
    values = df.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(values)
    scaler_y = MinMaxScaler(feature_range=(0,1))
    scaler_y.fit(values[:,5].reshape(len(values),1))
    values = scaler.transform(values)
    values = values.reshape(1, values.shape[0], values.shape[1])
    return values, scaler_y

if __name__ == "__main__":
    modelpath = sys.argv[1]
    filepath = sys.argv[2]
    
    df = pd.read_csv(filepath)
    
    values, scaler = data_prep(df)

    model = load_model(modelpath)

    prediction = model.predict(values)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction[0,0]

    print(prediction)

