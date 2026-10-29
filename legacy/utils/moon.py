from astronomy import *
from astropy.coordinates import position_angle
import calendar
import datetime
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

year = datetime.date.today().year
if calendar.isleap(year):
    days_in_year = 366
else:
    days_in_year = 365

obs = Observer(40, 0, 100)


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
    Time.Make(year, 1, 1, 12, m, 00.0)
    for m in np.arange(0, days_in_year * 24 * 60)
])

backgroung_time = mins_in_years.reshape((365, 24 * 60))

# moon_alts = true_alt(Body.Moon, backgroung_time, obs)

# np.save("moon_alts.npy", moon_alts) 

moon_brightness_list = []
for time in backgroung_time.flatten():
    ill = Illumination(Body.Moon, time)
    moon_brightness_list.append(ill.mag)
moon_brightness = np.array(moon_brightness_list)

moon_brightness = np.power(10, -moon_brightness / 2.5) #mag to linear
moon_scale = np.array(moon_brightness / moon_brightness.max())

moon_scale = moon_scale.reshape(backgroung_time.shape)

np.save("moon_brightness.npy", moon_scale) 