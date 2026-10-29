"""Background colors from explicitly supplied altitude arrays.

The original coloring and gamma expressions are intentionally retained.
"""
import numpy as np


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


def getbackground(sun_alts, *, y_scale_factor=14):
    """Return RGB background from a (night count, 1440) solar altitude array.

    Altitudes are airless degrees. No date, location, or data path is inferred.
    """
    import cv2

    sun_alts = np.asarray(sun_alts)
    if sun_alts.ndim != 2 or sun_alts.shape[1] != 1440 or not sun_alts.shape[0]:
        raise ValueError("Solar altitude array must have shape (night count, 1440)")
    if not np.isfinite(sun_alts).all():
        raise ValueError("Solar altitude array contains non-finite values")
    background = np.tile([0, 0, 0.01298303], (*sun_alts.shape, 1))
    sky_color = np.vectorize(srgb_gamma)(background + twilight_color(sun_alts))
    width = sky_color.shape[1]
    aspect_factor = (sun_alts.shape[0] / y_scale_factor) / 24
    return cv2.resize(sky_color, (width, int(width * aspect_factor)))
