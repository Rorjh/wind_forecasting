import time
import cdsapi

def request(year):
    return ({
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
            '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
            'mean_sea_level_pressure', 'surface_solar_radiation_downwards', 'total_precipitation',
        ],
        'year': [
            str(year),
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            51, 19, 49,
            21,
        ],
    })

c = cdsapi.Client(debug=True, wait_until_complete=False)

requests = [[year, c.retrieve('reanalysis-era5-single-levels',request(year))] for year in [2002,2003]]
sleep = 60

while True:
    for item in requests:
        r = item[1]
        r.update()
        reply = r.reply
        r.info("Request ID: %s, state: %s" % (reply['request_id'], reply['state']))
        if reply['state'] == 'completed':
            r.download('dane_[50,20]_12m_'+str(item[0])+'.nc')
            item[0] = 0
        elif reply['state'] in ('queued', 'running'):
            r.info("Request ID: %s, sleep: %s", reply['request_id'], sleep)
        elif reply['state'] in ('failed',):
            r.error("Message: %s", reply['error'].get('message'))
            r.error("Reason:  %s", reply['error'].get('reason'))
            for n in reply.get('error', {}).get('context', {}).get('traceback', '').split('\n'):
                if n.strip() == '':
                    item[0] = 0
                r.error("  %s", n)
            raise Exception("%s. %s." % (reply['error'].get('message'), reply['error'].get('reason')))
    requests = [x for x in requests if not x[0] == 0]
    if len(requests) == 0: break
    time.sleep(sleep)