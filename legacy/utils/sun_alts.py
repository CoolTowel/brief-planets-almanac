import time
from astronomy import Equator, Horizon, Time, Body, Observer, Refraction
from astropy.coordinates import position_angle
import calendar
import datetime
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def sun_alts(year:int, obs: Observer):
    time_zone = (obs.longitude)/15

    EPH_DIR = os.path.join(DATA_DIR, str(year), str(obs.latitude))

    if calendar.isleap(year):
        days_in_year = 366
    else:
        days_in_year = 365

    def true_alt(body, times, obs):
        result = []
        for time in times.flatten():
            eq = Equator(body, time, obs, True, False)
            azalt = Horizon(time, obs, eq.ra, eq.dec, Refraction.Airless)
            result.append(azalt.altitude)
        result = np.array(result)
        result = result.reshape(times.shape)
        return np.array(result) # angle in deg

    mins_in_years = np.array([
        Time.Make(year, 1, 1, 12-time_zone, m, 00.0) # 直接从十二点起算，因为第一天的上午不在图中
        for m in np.arange(0, days_in_year * 24 * 60)
    ])

    backgroung_time = mins_in_years.reshape((days_in_year, 24 * 60))

    sun_alts = true_alt(Body.Sun, backgroung_time, obs=obs)

    np.save(os.path.join(EPH_DIR, 'sun_alts.npy'), sun_alts) 
