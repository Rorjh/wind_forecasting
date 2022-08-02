import numpy as np
import pandas as pd
import netCDF4

from netCDF4 import num2date
from functions import *

def netCDF2df (filename):
    f = netCDF4.Dataset(filename)
    var_names = ['t2m', 'd2m', 'msl', 'tp', 'u10', 'v10', 'u100', 'v100', 'ssrd'] #100m u,v and ssrd data for evalutation
    variables = {key: f.variables[key][:].flatten() for key in var_names} # dictionary of variable name and data, already flattened for pd
    
    # Extract variable - każdy parametr ma przypisane time, lat, lon
    t2m = f.variables['t2m']
    # Get dimensions assuming 3D: time, latitude, longitude
    time_dim, lat_dim, lon_dim = t2m.get_dims()
    time_var = f.variables[time_dim.name]
    times = num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    latitudes = f.variables[lat_dim.name][:]
    longitudes = f.variables[lon_dim.name][:]
    lat, lon = latitudes[0], longitudes[0]
    times_grid, latitudes_grid, longitudes_grid = [x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing='ij')]
    
    # additional processing of ssrd data- getting rid of negative values
    ssrd_d = f.variables['ssrd'][:].flatten()
    ssrd_d = np.where(ssrd_d < 10, 0, ssrd_d) / 3600 / 1000 #divide by 3600 s so [J/m2]-->[W/m2], divide by 1000 so finally [kW/m2]
    variables.update({'ssrd' : ssrd_d})
    
    #time variables transformation - rotating vector to u,v components    
    yd_i = [np.sin(t.timetuple()[7]*2*np.pi/year_len(t.year)) for t in times_grid]
    yd_j = [np.cos(t.timetuple()[7]*2*np.pi/year_len(t.year)) for t in times_grid]
    h_i = [np.sin(t.hour*np.pi/12) for t in times_grid]
    h_j = [np.cos(t.hour*np.pi/12) for t in times_grid]
    time = {'yday_i': yd_i, 'yday_j' : yd_j, 'hour_i' : h_i, 'hour_j' : h_j}
    
    #final df build
    df = pd.DataFrame({**{
        'Date Time': [t.isoformat() for t in times_grid],
        #'year' : [t.year for t in times_grid], #used for df division
        #'month' : [t.month for t in times_grid], #used for df trim when full_year = False
        #'yday' : [t.timetuple()[7] for t in times_grid],
        #'hour' : [t.hour for t in times_grid],
        #'dorn' : [day_or_night(t,lat,lon) for t in times_grid],
        #**time,
        **variables
        }})
    f.close()
    return df

def netcdf2csv():
    years = range(1950,2021)
    files = ['./data/ERA5/dane_[50,20]_12m_'+str(year)+'.nc' for year in years]
    df = pd.DataFrame()
    for file in files:
        df = df.append(netCDF2df(file),ignore_index=True)
    
    df['Date Time'] = pd.to_datetime(df.pop('Date Time'), infer_datetime_format=True)
    df['tp'] = df['tp'].fillna(0)

    df['windspeed_10'] = (df['u10']**2 + df['v10']**2)**(1/2)
    df['windspeed_100'] = (df['u100']**2 + df['v100']**2)**(1/2)
    df.drop(labels=['u10', 'v10', 'u100', 'v100'], axis=1, inplace = True)

    df.to_csv('data/table.csv', index = False)

def load_netcdf(path):
    years = range(1950,2021)
    files = [path+'/dane_[50,20]_12m_'+str(year)+'.nc' for year in years]
    df = pd.DataFrame()
    for file in files:
        df = df.append(netCDF2df(file),ignore_index=True)
    
    df['Date Time'] = pd.to_datetime(df.pop('Date Time'), infer_datetime_format=True)
    df['tp'] = df['tp'].fillna(0)

    df['windspeed_10'] = (df['u10']**2 + df['v10']**2)**(1/2)
    df['windspeed_100'] = (df['u100']**2 + df['v100']**2)**(1/2)
    df.drop(labels=['u10', 'v10', 'u100', 'v100'], axis=1, inplace = True)

    return df