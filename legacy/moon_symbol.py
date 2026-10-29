from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from astronomy import Equator, Time, Body, Observer, MoonPhase, GeoVector, RotationAxis, Vector, EquatorFromVector
from astropy.coordinates import position_angle
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.transforms import Affine2D
import os

# path = os.path.dirname(__file__)
# datapath = os.path.dirname(path) + '/data/'

lms = []
for f in ['lm1.txt', 'lm2.txt', 'lm3.txt']:
    lms.append(np.loadtxt("./data/" + f))


def degToRad(deg):
    deg = deg % 360
    rad = deg / 180.0 * 3.1415926
    return rad


def moon_edge(resolution=500):
    '''
    The outline of moon. return an array of its pixel coordinates [X,Y] with 2 column
    '''
    t = np.linspace(0, np.pi * 2.0, resolution)
    t = t.reshape((len(t), 1))
    x = np.cos(t)
    y = np.sin(t)
    return np.hstack((x, y))


def moon_terminator(resolution=500, b=0.5, k=1.005):
    '''
    The ellipse of moon's terminator. A factor k was applying to the radius to make the final looks better. return an array of its pixel coordinates [X,Y] with 2 column. The points are not uniform in the arc. 
    '''
    t = np.linspace(0, np.pi * 2.0, resolution)
    t = t.reshape((len(t), 1))
    x = np.cos(t) * k
    y = np.sin(t) / b * k  #椭圆
    return np.hstack((x, y))


def moon_psa(time):
    scale = 0.0000116138
    moonVector = GeoVector(Body.Moon, time, False)
    moonAxis = RotationAxis(Body.Moon, time)
    moonNorthVector = Vector(moonVector.x + moonAxis.north.x * scale,
                             moonVector.y + moonAxis.north.y * scale,
                             moonVector.z + moonAxis.north.z * scale, time)
    moonEQ = EquatorFromVector(moonVector)
    moonNorthEQ = EquatorFromVector(moonNorthVector)
    psa = position_angle(degToRad(moonEQ.ra * 15), degToRad(moonEQ.dec),
                         degToRad(moonNorthEQ.ra * 15),
                         degToRad(moonNorthEQ.dec))
    return psa.rad


def draw_moon_symbol(ax,
                     time,
                     lms,
                     moon_phase,
                     day,
                     hour,
                     mares=False,
                     obs=Observer(40, 0, 100),
                     y_zoom_factor=1,
                     zoom_factor=1,
                     resolution=500,
                     light_color=np.array([1.0, 253 / 255, 230 / 255]),
                     shadow_color=0.15,
                     k=1.0005):

    # 0-360deg, 0 is the mew moon, and 180 is the full moon
    if mares == True:
        zoom_factor *= 1.3

    if moon_phase == 0:
        M_zoom = np.array([[zoom_factor, 0], [0, zoom_factor * y_zoom_factor]])
        ring_path = Wedge((0, 0),
                          1,
                          0,
                          360,
                          width=0.15,
                          color=light_color,
                          zorder=1000).get_path()
        ring_vertices = ring_path.vertices
        ring_code = ring_path.codes
        ring_vertices = np.dot(ring_vertices, M_zoom.T)
        ring_vertices[:,0] += hour 
        ring_vertices[:,1] += day 
        ring_path = mpath.Path(ring_vertices, ring_code)
        ring = mpatches.PathPatch(ring_path,
                                #   facecolor=[shadow_color]*3,
                                  facecolor=[174/255,174/255,159/255],
                                  linewidth=0,
                                  zorder=999)
        ax.add_patch(ring)
        return None
    #     # ax.add_patch(Wedge((hour, day), .2, 0, 360, width=0.05, color=light_color, zorder=1000))
    #     return None
    else:
        moon_phase += 0.0001
    if mares != True:
        if moon_phase < 25:
            moon_phase = 25
        if moon_phase > 335:
            moon_phase = 335  #蛾眉月太细看不见，这里直接加粗点

    moon_phase = degToRad(moon_phase)

    moon = Equator(Body.Moon, time, obs, True, False)
    sun = Equator(Body.Sun, time, obs, True, False)

    # light_side_angle = moon.position_angle(sun).rad
    light_side_angle = position_angle(degToRad(moon.ra * 15),
                                      degToRad(moon.dec),
                                      degToRad(sun.ra * 15),
                                      degToRad(sun.dec)).rad

    b = np.absolute(1 / (np.cos(moon_phase)))
    edge = moon_edge(resolution=resolution)
    termi = moon_terminator(resolution=resolution, k=k, b=b)
    # k, terminator radius scale. Making the radius of terminator slighly larger than moon angular size to make simulate the true moon phase
    cross_y = np.sqrt((k**2 - 1) / (b**2 - 1))

    if np.cos(moon_phase) > 0:
        if cross_y >= 1:
            light = None
            shadow = edge
        else:
            cross_x = np.sqrt((b**2 - k**2) / (b**2 - 1))
            moon_edge_light = edge[edge[:, 1] > cross_y, :]
            moon_edge_light = np.vstack(([cross_x, cross_y], moon_edge_light))
            moon_terminator_line = termi[termi[:, 1] > cross_y, :]
            moon_terminator_line = np.vstack(
                ([-cross_x, cross_y], moon_terminator_line[::-1]))
            light = np.vstack((moon_edge_light, moon_terminator_line))
            indices = np.where(edge[:, 1] >= cross_y)[0]
            lower = 1 - (len(edge[:, 0]) - indices[-1])
            upper = indices[0]
            moon_edge_shadow = np.vstack((edge[lower:, :], edge[:upper, :]))
            moon_edge_shadow = np.vstack(([cross_x,
                                           cross_y], moon_edge_shadow[::-1]))
            shadow = np.vstack((moon_terminator_line, moon_edge_shadow))
    elif np.cos(moon_phase) < 0:
        if cross_y >= 1:
            light = edge
            shadow = None
        else:
            cross_x = np.sqrt((b**2 - k**2) / (b**2 - 1))
            cross_y = -cross_y
            indices = np.where(edge[:, 1] <= cross_y)[0]
            lower = 1 - (len(edge[:, 0]) - indices[-1])
            upper = indices[0]
            moon_edge_light = np.vstack((edge[lower:, :], edge[:upper, :]))
            moon_edge_light = np.vstack(([cross_x, cross_y], moon_edge_light))
            moon_terminator_line = termi[termi[:, 1] < cross_y, :]
            moon_terminator_line = np.vstack(([-cross_x,
                                               cross_y], moon_terminator_line))
            light = np.vstack((moon_edge_light, moon_terminator_line))
            moon_edge_shadow = edge[edge[:, 1] < cross_y, :]
            moon_edge_shadow = np.vstack(([cross_x,
                                           cross_y], moon_edge_shadow[::-1]))
            shadow = np.vstack((moon_terminator_line, moon_edge_shadow))
    else:
        light = edge[edge[:, 1] >= 0, :]
        shadow = edge[edge[:, 1] <= 0, :]

    # global affine transformation


    M_zoom = np.array([[zoom_factor, 0], [0, zoom_factor * y_zoom_factor]])

    # moon phase affine transformation
    M_rotation = np.array(
        [[np.cos(light_side_angle), -np.sin(light_side_angle)],
         [np.sin(light_side_angle),
          np.cos(light_side_angle)]])

    def affine_t(vectors):
        vectors = vectors.T
        vectors = np.dot(M_rotation, vectors)
        vectors = np.dot(M_zoom, vectors)
        vectors[0] += hour
        vectors[1] += day
        return vectors.T

    if light is not None:
        light = affine_t(light)
        light_codes = np.ones(len(light),
                              dtype=mpath.Path.code_type) * mpath.Path.LINETO
        light_codes[0] = mpath.Path.MOVETO
        light_path = mpath.Path(light, light_codes)
        light_patch = mpatches.PathPatch(light_path,
                                         facecolor=light_color,
                                         linewidth=0,
                                         zorder=999)
        ax.add_patch(light_patch)

    if mares == True:
        if shadow is not None:
            shadow = affine_t(shadow)
            shadow_codes = np.ones(
                len(shadow), dtype=mpath.Path.code_type) * mpath.Path.LINETO
            shadow_codes[0] = mpath.Path.MOVETO
            shadow_path = mpath.Path(shadow, shadow_codes)
            shadow_patch = mpatches.PathPatch(shadow_path,
                                              facecolor=[shadow_color] * 3,
                                              linewidth=0,
                                              zorder=999)
            ax.add_patch(shadow_patch)
        # Lunar mare patch

        pole_angle = -moon_psa(time)  # 天球中位置角与此处仿射变换矩阵的旋转角方向相反

        M_rotation_mare = np.array([[np.cos(pole_angle), -np.sin(pole_angle)],
                                    [np.sin(pole_angle),
                                     np.cos(pole_angle)]])

        def affine_mare(vectors):
            vectors = vectors.T
            vectors = np.dot(M_rotation_mare, vectors)
            vectors = np.dot(M_zoom, vectors)
            vectors[0] += hour
            vectors[1] += day
            return vectors.T

        for lm in lms:
            lm = affine_mare(lm)
            lm_codes = np.ones(len(lm),
                               dtype=mpath.Path.code_type) * mpath.Path.LINETO
            lm_codes[0] = mpath.Path.MOVETO
            lm_path = mpath.Path(lm, lm_codes)
            lm_patch = mpatches.PathPatch(lm_path,
                                          facecolor=[0.3, 0.3, 0.3, 0.4],
                                          linewidth=0,
                                          zorder=1000)
            ax.add_patch(lm_patch)


if __name__ == '__main__':
    time = Time.Make(2023, 1, 18, 12, 0, 00.0)
    plt.figure()
    plt.subplot(111)
    ax = plt.gca()
    draw_moon_symbol(ax,
                     time,
                     lms,
                     day=0,
                     hour=0,
                     mares=True,
                     zoom_factor=0.7 / 1.3,
                     y_zoom_factor=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_aspect(1)
    plt.show()