import time
from astronomy import *
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2


def srgb_gamma(linear):
    if linear <= 0:  #gamut clip
        srgb = 0.0
    elif linear >= 1:
        srgb = 1.0
    else:
        a = 0.055
        gamma = 2.4

        if linear < 0.00304:
            srgb = 12.92 * linear
        else:
            srgb = (1 + a) * linear**(1 / gamma) - a
        
        srgb = linear**(1 / gamma)

    return srgb


vsrgb_gamma = np.vectorize(srgb_gamma)


def _srgb_degamma(srgb):
    if srgb < 0.04045:
        return srgb / 12.92
    else:
        linear = ((srgb + 0.055) / 1.055)**2.4
        return linear


srgb_degamma = np.vectorize(_srgb_degamma)

dark_night_color = [0, 0, 0.01298303]

year = datetime.date.today().year
if calendar.isleap(year):
    days_in_year = 366
else:
    days_in_year = 365
    
data_path = './data/' + str(year) + '/'

north40 = Observer(40, 120, 40)
obs = north40

def true_alt(body, times, obs):
    if isinstance(times, np.ndarray):
        result = []
        for time in times.flatten():
            eq = Equator(body, time, obs, True, False)
            azalt = Horizon(time, obs, eq.ra, eq.dec, Refraction.Airless)
            result.append(azalt.altitude)
        result = np.array(result)
        result = result.reshape(times.shape)
    else:
        eq = Equator(body, times, obs, True, False)
        azalt = Horizon(times, obs, eq.ra, eq.dec, Refraction.Airless)
        result = azalt.altitude
    return result  # angle in deg


def moon_true_alt(time, obs):
    eq = Equator(Body.Moon, time, obs, True, False)
    azalt = Horizon(time, obs, eq.ra, eq.dec, Refraction.Airless)
    return azalt.altitude  # angle in deg


def twilight_color(sun_alt):
    # R = R_func(sun_alt)
    # G = G_func(sun_alt)
    # B = B_func(sun_alt)

    factor = (sun_alt / 18 + 1)**2 *3
    R = factor * 0.0
    G = factor * 0.07227185
    B = factor * 0.22541454
    R[sun_alt <= -18] = 0
    G[sun_alt <= -18] = 0
    B[sun_alt <= -18] = 0
    R[sun_alt >= 0] = 1
    G[sun_alt >= 0] = 1
    B[sun_alt >= 0] = 1

    RGB = np.dstack([R, G, B])
    return RGB

def moon_twilight_color(moon_alt):
    color_adjustment = 0.8
    factor = abs(moon_alt / 18 + 1)
    R_max = 0.81484657 * color_adjustment
    B_max = 0.80695226 * color_adjustment
    G_max = 0.54572446 * color_adjustment

    R_max = 0. * color_adjustment
    B_max = 0.07227185 * color_adjustment
    G_max = 0.22541454 * color_adjustment
    R = factor * R_max
    G = factor * B_max
    B = factor * G_max
    R[moon_alt < -18] = 0
    G[moon_alt < -18] = 0
    B[moon_alt < -18] = 0
    R[moon_alt > 0] = R_max
    G[moon_alt > 0] = B_max
    B[moon_alt > 0] = G_max

    RGB = np.dstack([R, G, B])
    return RGB



def getbackground(obs, enlarging = 2, withmoon= False, y_scale_factor=14):
    sun_alts = np.load('./data/'+str(year)+'/'+str(obs.latitude)+'/sun_alts.npy')
    
    

    backgroung_color = np.tile(
        dark_night_color, (sun_alts.shape[0], sun_alts.shape[1], 1))
    twilight = twilight_color(sun_alts)
    

    
    if withmoon:
        moon_alts = np.load('./data/'+str(year)+'/'+str(obs.latitude)+'/moon_alts.npy')
        moon_bright_scale = np.load('./data/'+str(year)+'/'+str(obs.latitude)+'/moon_brightness.npy')
        moon_twilight = moon_twilight_color(moon_alts)
        moon_twilight = moon_twilight*np.dstack([moon_bright_scale, moon_bright_scale, moon_bright_scale])

        twilight += moon_twilight

    sky_color = backgroung_color + twilight
    sky_color = vsrgb_gamma(sky_color)

    sky_color_height = sky_color.shape[0]
    sky_color_width = sky_color.shape[1]
    aspect_factor = (days_in_year/14)/24
    sky_color = cv2.resize(sky_color, (sky_color_width, int(sky_color_width*aspect_factor)) )

    return sky_color

if __name__ == '__main__':
    plt.imshow(getbackground())
    plt.show()




# c_sky = '#dceaff'  # 220 234 255
# c_civ = '#88a5d4'  # 136 165 212
# c_nau = '#4673bc'  # 70 115 188
# c_ast = '#213c66'  # 33 60 102
# c_night = '#192029'  # 25 32 41

# degs = np.array([0, -3, -9, -15, -18])[::-1]
# R_values = np.array([220, 136, 70, 33,
#                      25])[::-1]  # corresponds to 0, 3, 9, 15, 18 deg
# G_values = np.array([234, 165, 115, 60, 32])[::-1]
# B_values = np.array([255, 212, 188, 102, 41])[::-1]

# R_func = interp1d(degs,
#                   R_values,
#                   kind='cubic',
#                   bounds_error=False,
#                   fill_value=(R_values[0], R_values[-1]))
# G_func = interp1d(degs,
#                   G_values,
#                   kind='cubic',
#                   bounds_error=False,
#                   fill_value=(G_values[0], G_values[-1]))
# B_func = interp1d(degs,
#                   B_values,
#                   kind='cubic',
#                   bounds_error=False,
#                   fill_value=(B_values[0], B_values[-1]))