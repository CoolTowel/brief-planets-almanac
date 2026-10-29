
import astronomy
from astropy.coordinates import position_angle
import time
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
def degToRad(deg):
    rad = deg / 180.0 * 3.1415926
    return rad


def radToDeg(rad):
    deg = rad / 3.1415926 * 180.0


moon = astronomy.Body(10)
scale = 0.0000116138


def moonpsa(m, d):
    onetime = astronomy.Time.Make(2023, m, d, 00, 00, 00.0)
    moonVector = astronomy.GeoVector(moon, onetime, False)
    moonAxis = astronomy.RotationAxis(moon, onetime)
    moonNorthVector = astronomy.Vector(moonVector.x + moonAxis.north.x * scale,
                                       moonVector.y + moonAxis.north.y * scale,
                                       moonVector.z + moonAxis.north.z * scale,
                                       onetime)
    moonEQ = astronomy.EquatorFromVector(moonVector)
    moonNorthEQ = astronomy.EquatorFromVector(moonNorthVector)
    psa = position_angle(degToRad(moonEQ.ra * 15), degToRad(moonEQ.dec), degToRad(moonNorthEQ.ra * 15 ), degToRad(moonNorthEQ.dec))
    if psa>= 0:
        return psa.deg
    else:
        return psa.deg + 360

start = time.time()
steps = 1
monthNum = 12
i = 1
while i<=steps:
    for m in range(13):
        for d in range(32):
          psa = moonpsa(m, d)
    i += 1
stop = time.time()
print("{} times done, in {} s".format(monthNum * 31 * steps, stop-start))

