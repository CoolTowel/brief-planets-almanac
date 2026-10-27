from astronomy import Time, Body, BodyCode, Observer, SearchRiseSet, Direction, SearchHourAngle, DefineStar, SearchAltitude
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

north40 = Observer(40, 116, 40) # Beijing
north24 = Observer(24, 116, 5) # Amoy
north52 = Observer(52, 4, 20) # Leiden

obs = north40
time_zone = time_zone = (obs.longitude)/15

year = datetime.date.today().year+1

time_origin = Time.Make(year, 1, 1, 12 - time_zone, 00, 00.0)

if calendar.isleap(year):
    days_in_year = 366
else:
    days_in_year = 365

rise_table = Table(
    meta={'info': 'rise time for solar system bodies in year of' + str(year)})
set_table = Table(
    meta={'info': 'set time for solar system bodies in year of' + str(year)})
transit_table = Table(
    meta={'info': 'transit time for solar system bodies in year of' + str(year)})

astro_twilight_table = Table(
    meta={'info': 'astro twilight end and start time in year of' + str(year)})

rise_table.add_column(np.arange(days_in_year).T + 1.0, name='Day')
set_table.add_column(np.arange(days_in_year).T + 1.0, name='Day')
transit_table.add_column(np.arange(days_in_year).T + 1.0, name='Day')
astro_twilight_table.add_column(np.arange(days_in_year).T + 1.0, name='Day')


def _search(table: Table, direction, body: Body, obs=obs, time_zone=time_zone):
    time_list = []

    for day in table['Day']:
        noon = Time(time_origin.ut+day-1)
        time = SearchRiseSet(body = body, observer= obs , direction = direction, startTime= noon, limitDays=1)
        if time is not None:
            time_hour = (time.ut-noon.ut)*24 - 12 
            time_list.append(time_hour)
        else:
            time_list.append(-999.0)

    time_list = np.array(time_list)

    table.add_column(time_list, name=body.name)

def search_rise_set(body: Body,
                    rise_table=rise_table,
                    set_table=set_table,
                    obs=obs,
                    time_zone=time_zone):
    _search(table=set_table,
            direction=Direction.Set,
            body=body,
            obs=obs,
            time_zone=time_zone)
    _search(table=rise_table,
            direction=Direction.Rise,
            body=body,
            obs=obs,
            time_zone=time_zone)
    
def search_transit(body: Body,
                    transit_table=transit_table,
                    obs=obs,
                    time_zone=time_zone):
    time_list = []

    for day in transit_table['Day']:
        noon = Time(time_origin.ut+day-1)
        transit = SearchHourAngle(body = body, observer= obs ,hourAngle = 0.0, direction = 1, startTime= noon)
        if transit.time.ut-noon.ut<1:
            time_hour = (transit.time.ut-noon.ut)*24 - 12 
            time_list.append(time_hour)
        else:
            time_list.append(-999.0)

    time_list = np.array(time_list)

    transit_table.add_column(time_list, name=body.name)

def search_twilight(table= astro_twilight_table, obs=obs, time_zone=time_zone):
    dusk_end_time_list = []
    dawn_start_time_list = []

    for day in table['Day']:
        noon = Time(time_origin.ut+day-1)
        dusk_end_time = SearchAltitude(body = Body.Sun, observer= obs , direction = Direction.Set, startTime= noon, limitDays=1, altitude=-18.0)
        if dusk_end_time is not None:
            time_hour = (dusk_end_time.ut-noon.ut)*24 - 12 
            dusk_end_time_list.append(time_hour)
        else:
            dusk_end_time_list.append(-999.0)

    dusk_end_time_list = np.array(dusk_end_time_list)


    for day in table['Day']:
        noon = Time(time_origin.ut+day-1)
        dawn_start_time = SearchAltitude(body = Body.Sun, observer= obs , direction = Direction.Rise, startTime= noon, limitDays=1, altitude=-18.0)
        if dawn_start_time is not None:
            time_hour = (dawn_start_time.ut-noon.ut)*24 - 12 
            dawn_start_time_list.append(time_hour)
        else:
            dawn_start_time_list.append(-999.0)

    dawn_start_time_list = np.array(dawn_start_time_list)

    table.add_column(dusk_end_time_list, name='Dusk')
    table.add_column(dawn_start_time_list, name='Dawn')

# define custom stars

DefineStar(body=Body.Star1, ra = 10+8/60, dec = 11+57/60, distanceLightYears=1000) # 轩辕十四
DefineStar(body=Body.Star2, ra = 14+15/60, dec = 19+9/60, distanceLightYears=1000) # 大角
DefineStar(body=Body.Star3, ra = 18+36/60, dec = 38+46/60, distanceLightYears=1000) # 织女
DefineStar(body=Body.Star4, ra = 22+57/60, dec = -29+37/60, distanceLightYears=1000) # 北落师门
DefineStar(body=Body.Star5, ra = 6+45/60, dec = -16+43/60, distanceLightYears=1000) # 天狼

# rise and set 
for name in ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Star1', 'Star2', 'Star3', 'Star4', 'Star5' ]: 
    search_rise_set(BodyCode(name))

rise_table.write('./data/'+str(year)+'/'+str(obs.latitude)+'/rise.fits', overwrite=True)
set_table.write('./data/'+str(year)+'/'+str(obs.latitude)+'/set.fits', overwrite=True)

# transit
for name in ['Mars', 'Jupiter', 'Saturn', 'Star1', 'Star2', 'Star3', 'Star4', 'Star5']:
    search_transit(BodyCode(name))
transit_table.write('./data/'+str(year)+'/'+str(obs.latitude)+'/transit.fits', overwrite=True)

# astro twilight
search_twilight()
astro_twilight_table.write('./data/'+str(year)+'/'+str(obs.latitude)+'/twilight.fits', overwrite=True)

if __name__ == '__main__':
    plt.plot(rise_table['Sun'], rise_table['Day'])
    plt.plot(set_table['Sun'], set_table['Day'])

    for col in rise_table.columns[3:]:
        plt.plot(rise_table[col], rise_table['Day'])
    for col in set_table.columns[3:]:
        plt.plot(set_table[col], set_table['Day'])
    ax = plt.gca()

    ax.set_xlim(-12, 12)
    ax.set_ylim(days_in_year, 1)
    plt.show()
