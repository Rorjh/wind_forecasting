import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from calendar import isleap


def day_or_night(t,lat,lon):
    ''' Day or night - function generating additional input variable for ML model.
    Astral uses UTC times by default so no conversion is needed
    t - python datetime object
    lat - decimal latitude
    lon - decimal longitude
    '''
    city = LocationInfo('','','',lat, lon)
    s = sun(city.observer, t)
    sunrise = s['sunrise']
    sunset = s['sunset']
    if t.hour <= sunrise.hour:
        return 0
    elif (t.hour - sunrise.hour) == 1:
        if sunrise.minute < 30: return 1
        else: return 0
    elif t.hour <= sunset.hour:
        return 1
    elif (t.hour - sunset.hour) == 1:
        if sunset.minute >= 30: return 1
        else: return 0
    else:
        return 0

def year_len(year):
    ''' Checks if a year is leap, returns number of days.
    '''
    if isleap(year) == True:
        return 366
    else:
        return 365


def shiftdrop (df, n):
    # shift values of predicted columns - so true value after interval is avaialble. Drop incomplete rows
    for each in df.columns:
        if '+'+str(n)+'h' in each: df[each] = df[each].shift(-n)
    df.drop(df.tail(n).index,inplace=True) # drop last n rows that are incomplete
    return df