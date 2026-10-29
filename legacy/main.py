from astronomy import Time, Body, Observer, MoonPhase, SearchMoonQuarter, NextMoonQuarter, HourAngle, DefineStar
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from backgound import getbackground
from moon_symbol import draw_moon_symbol
from astropy.table import Table
from numpy.lib import recfunctions as rfn
'''
Background part: use sun altitude to color the background
draw a image with shape (365, 24*60), 60min *24h for 365 days
'''
# font
font_path = './fonts/Libertinus-7.040/static/OTF/'
font_files = fm.findSystemFonts(fontpaths=font_path)
for font_file in font_files:
    fm.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Libertinus Sans'
plt.rcParams['font.size'] = 13

# observer
north40 = Observer(40, 116, 40)  # Beijing
north24 = Observer(24, 116, 5)  # Amoy
north52 = Observer(52, 4, 20)  # Leiden

obs = north40
time_zone = (obs.longitude) / 15

# year number in Common Era (CE)
year = datetime.date.today().year+1

# days count in this year
if calendar.isleap(year):
    days_in_year = 366
else:
    days_in_year = 365

# time orgin: noon (12 am) of the first day
time_origin = Time.Make(year, 1, 1, 12 - time_zone, 00, 00.0)

# y scale for all
y_scale_factor = 1 / 14

month_background_width =0.8

# font size
fontsize = 10

# load lunar mare shape file
lms = []
for f in ['lm1.txt', 'lm2.txt', 'lm3.txt']:
    lm = np.loadtxt('./data/' + f)
    lm[:, 1] = -lm[:, 1]  # counteracting the final y axis flipping
    lms.append(lm)

# read rise and set times, sun alts and EOT
data_path = './data/' + str(year) + '/'
EOT_path = './data/' + str(year) + '/'

rise_table = Table.read(data_path + str(obs.latitude) + '/' + 'rise.fits')
set_table = Table.read(data_path + str(obs.latitude) + '/' + 'set.fits')
transit_table = Table.read(data_path + str(obs.latitude) + '/' +
                           'transit.fits')
twilight_table = Table.read(data_path + str(obs.latitude) + '/' +
                            'twilight.fits')
EOT = np.load(EOT_path + 'EOT.npy')


def draw_moons(ax, moon_rise_or_set, quarter_day_list):
    moon_rise_or_set_night = moon_rise_or_set[night_mask(moon_rise_or_set)[:,
                                                                           0]]
    for day, hour in moon_rise_or_set_night:
        if (day in quarter_day_list):
            pass
        else:
            time_ut = Time.Make(year, 1, 1, 12 - time_zone, 00,
                                00.0).ut + (day - 1) + (hour + 12) / 24
            time = Time(ut=time_ut)
            moon_phase = MoonPhase(time)
            draw_moon_symbol(ax=ax,
                             time=time,
                             moon_phase=moon_phase,
                             lms=lms,
                             day=day,
                             hour=hour,
                             y_zoom_factor=1 / y_scale_factor,
                             zoom_factor=0.075,
                             mares=False)


def draw_patch(ax, vertex, color=[1] * 3, zorder=1):
    codes = np.ones(len(vertex),
                    dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO
    path = mpath.Path(vertex, codes)
    patch = mpatches.PathPatch(path,
                               facecolor=color,
                               linewidth=0,
                               zorder=zorder)
    ax.add_patch(patch)


def month_background(ax, sun_rise_or_set_day_hour, left=True):
    if left:
        shift = -month_background_width
    else:
        shift = month_background_width
    blue_vertex_1 = np.copy(sun_rise_or_set_day_hour[:, ::-1])
    # day is y coordinate, hour is x coord so here we need to filp it.
    blue_vertex_2 = np.copy(sun_rise_or_set_day_hour[::-1, ::-1])
    blue_vertex_2[:, 0] += shift
    blue_vertex = np.vstack([blue_vertex_1, blue_vertex_2])
    draw_patch(
        ax,
        blue_vertex,
        color=[232 / 255, 246 / 255, 252 / 255],
        #    color=[200 / 255, 230 / 255, 232 / 255],
        zorder=10)


# get data in astropy table
def rise_set(body: str):
    return [
        rfn.structured_to_unstructured(rise_table['Day', body].as_array()),
        rfn.structured_to_unstructured(set_table['Day', body].as_array())
    ]


def dusk_dawn():
    return [
        rfn.structured_to_unstructured(twilight_table['Day',
                                                      'Dusk'].as_array()),
        rfn.structured_to_unstructured(twilight_table['Day',
                                                      'Dawn'].as_array())
    ]


def transit(body: str):
    return rfn.structured_to_unstructured(transit_table['Day',
                                                        body].as_array())


def night_mask(day_hour, bleed=0.1):

    mask = np.tile(
        np.logical_and(day_hour[:, 1] >= sun_set[:, 1] - bleed,
                       day_hour[:, 1] <= sun_rise[:, 1] + bleed), (2, 1)).T
    return mask


def plot_object_label(ax,
                      body,
                      event,
                      x,
                      y,
                      verticalalignment='center',
                      horizontalalignment='center',
                      fontsize=fontsize - 4,
                      zorder=100,
                      c='Red'):
    ax.text(x,
            y,
            s=body[0] + body[-1] + '_' + event,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            fontsize=fontsize,
            zorder=zorder,
            c=c)


def plot_set(ax, bodies: list, zorder, lw, colors, object_lable=False):
    for body, c in zip(bodies, colors):
        _, set = rise_set(body)
        mask = night_mask(set, bleed=0.2)
        set = np.ma.array(set, mask=np.logical_not(mask)).filled(np.nan)
        ax.plot(set[:, 1], set[:, 0], zorder=zorder, lw=lw, c=c)
        if object_lable is True:
            try:
                plot_object_label(ax, body, 'S', x=set[mask][1], y=set[mask][0])
                plot_object_label(ax, body, 'S', x=set[mask][-1], y=set[mask][-2])
            except:
                print(body+' setting does not appear in night ')


def plot_rise(ax, bodies: list, zorder, lw, colors, object_lable=False):
    for body, c in zip(bodies, colors):
        rise, _ = rise_set(body)
        mask = night_mask(rise, bleed=0.2)
        rise = np.ma.array(rise, mask=np.logical_not(mask)).filled(np.nan)
        ax.plot(rise[:, 1], rise[:, 0], zorder=zorder, lw=lw, c=c)
        if object_lable is True:
            try:
                plot_object_label(ax, body, 'R', x=rise[mask][1], y=rise[mask][0])
                plot_object_label(ax,
                                body,
                                'R',
                                x=rise[mask][-1],
                                y=rise[mask][-2])
            except:
                print(body+' rising does not appear in night ')



def plot_transit(ax, bodies: list, zorder, lw, colors, object_lable=False):
    for body, c in zip(bodies, colors):
        transit_list = transit(body)
        mask = night_mask(transit_list, bleed=0.2)
        transit_list = np.ma.array(transit_list,
                                   mask=np.logical_not(mask)).filled(np.nan)
        ax.plot(transit_list[:, 1],
                transit_list[:, 0],
                zorder=zorder,
                lw=lw,
                c=c)

        if object_lable is True:
            try:
                plot_object_label(ax,
                                body,
                                'T',
                                x=transit_list[mask][1],
                                y=transit_list[mask][0])
                plot_object_label(ax,
                                body,
                                'T',
                                x=transit_list[mask][-1],
                                y=transit_list[mask][-2])
            except:
                print(body+' transiting does not appear in night ')

# sun
sun_rise, sun_set = rise_set('Sun')

# for conviniance
sun_rise_days = sun_rise[:, 0]
sun_rise_hours = sun_rise[:, 1]
sun_set_days = sun_set[:, 0]
sun_set_hours = sun_set[:, 1]

# moon
moon_rise, moon_set = rise_set('Moon')
'''
 plotting codes
'''


def main_plot(filename,
              frame=False,
              lines=False,
              moon=False,
              background=False,
              grid=False,
              object_lable=False,
              text=False,
              meridian_ra=False,
              eot=False):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)

    date_zorder = 30
    days_7 = np.arange(1, days_in_year, 7)  # days with step of 7

    DefineStar(body=Body.Star8, ra=0, dec=0,
               distanceLightYears=1000)  # J2000 origin
    RA_offset = HourAngle(Body.Star8, Time(time_origin.ut - 0.5),
                          obs)  # the start of this year
    sidereal_year = 365.25636042

    min_set_hour = -int(-np.min(sun_set_hours)) - 1
    max_rise_hour = int(np.max(sun_rise_hours)) + 1

    if frame:
        # left_white
        left_white_vertex = sun_set[:, ::-1]
        # day is y coordinate, hour is x coord so here we need to filp it.
        left_edge = np.array([[-12, days_in_year], [-12, 1]])
        left_white_vertex = np.vstack([left_white_vertex, left_edge])
        draw_patch(ax, left_white_vertex, zorder=10)

        # right_white
        right_white_vertex = sun_rise[:, ::-1]
        # day is y coordinate, hour is x coord so here we need to filp it.
        right_edge = np.array([[12, days_in_year], [12, 1]])
        right_white_vertex = np.vstack([right_white_vertex, right_edge])
        draw_patch(ax, right_white_vertex, zorder=10)

        # month and date background blue
        month_background(ax, sun_set, left=True)
        month_background(ax, sun_rise, left=False)

        # left and right black lines
        black_line_zorder = 20
        ax.plot(sun_rise_hours,
                sun_rise_days,
                c='k',
                lw=1,
                zorder=black_line_zorder)
        ax.plot(sun_set_hours,
                sun_set_days,
                c='k',
                lw=1,
                zorder=black_line_zorder)

        ax.plot(sun_rise_hours + month_background_width,
                sun_rise_days,
                c='k',
                lw=0.5,
                zorder=black_line_zorder)
        ax.plot(sun_set_hours - month_background_width,
                sun_set_days,
                c='k',
                lw=0.5,
                zorder=black_line_zorder)

        # top and butomm line
        ax.plot([sun_set_hours[1] - month_background_width, sun_rise_hours[1] + month_background_width-0.015], [1, 1],
                c='k',
                lw=1,
                zorder=black_line_zorder)
        ax.plot([sun_set_hours[-1] - month_background_width + 0.015, sun_rise_hours[-1] + + month_background_width-0.015],
                [days_in_year, days_in_year],
                c='k',
                lw=1,
                zorder=black_line_zorder)
        # month line
        for month in np.arange(2, 13):
            set_day = int(
                (Time.Make(year, month, 1, 12 - time_zone, 1, 00.0).ut -
                 time_origin.ut) // 1 + 1)
            set_hour = sun_set[set_day - 1, 1]
            offset = 0.01
            monthline_color = 'Black'
            ax.plot([set_hour - month_background_width + offset, set_hour - offset],
                    [set_day, set_day],
                    lw=0.5,
                    c=monthline_color,
                    zorder=black_line_zorder - 1)
            rise_day = set_day - 1
            rise_hour = sun_rise[rise_day - 1, 1]
            ax.plot([rise_hour + offset, rise_hour + month_background_width - offset],
                    [rise_day, rise_day],
                    lw=0.5,
                    c=monthline_color,
                    zorder=black_line_zorder - 1)

    if meridian_ra:
        # RA at meridian
        for RA_J2000 in np.arange(
                0, 24,
                0.25):  # a imaginary epoch which 0 point pass through sky
            RA_imaginary = RA_J2000 - RA_offset  # the true RA at meridian
            if RA_imaginary < 0:
                RA_imaginary += 24
            RA_y = RA_imaginary / 24 * sidereal_year  # the day number of this RA
            if RA_J2000 % 1 > 0.1:  #small ticks
                ax.plot([-0.075, 0.075], [RA_y, RA_y],
                        zorder=0,
                        c='White',
                        lw=0.5)
            else:  #large ticks
                ax.plot([-0.175, 0.175], [RA_y, RA_y],
                        zorder=0,
                        c='White',
                        lw=0.5)

        # RA texts
        for RA_J2000 in np.arange(
                0, 24,
                0.25):  # a imaginary epoch which 0 point pass through sky
            RA_imaginary = RA_J2000 - RA_offset  # the true RA at meridian
            if RA_imaginary < 0:
                RA_imaginary += 24
            RA_y = RA_imaginary / 24 * sidereal_year  # the day number of this RA
            if RA_J2000 % 1 > 0.1:  #small ticks
                pass
            else:  #large ticks
                ax.text(0.225,
                        RA_y,
                        str(int(RA_J2000)),
                        fontsize=fontsize - 1,
                        c='White',
                        verticalalignment='center',
                        horizontalalignment='left',
                        zorder=0)

    if eot:
        # 均时差 equation of time
        EOT = np.load(data_path + 'EOT.npy')
        days_number = np.arange(days_in_year) + 1
        ax.plot(EOT, days_number, color='white', lw=0.5, zorder=1)
        # EOT axis
        ax.plot([0, 0], [1, days_in_year], color='white', lw=0.5, zorder=1)

    if text:
        # plot days of date
        for day in days_7:
            set_hour = sun_set[day - 1, 1]
            _, _, set_day, _, _, _ = Time.Make(year, 1, day, 12, 00,
                                               00.0).Calendar()
            rise_hour = sun_rise[day - 1, 1]
            _, _, rise_day, _, _, _ = Time.Make(year, 1, day + 1, 12, 00,
                                                00.0).Calendar()

            x_shift = 0.2
            y_shift = 0.5
            ax.text(x=set_hour - x_shift,
                    y=day + y_shift,
                    s=str(set_day),
                                        verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=fontsize,
                    zorder=date_zorder)
            ax.text(x=rise_hour + x_shift,
                    y=day + y_shift,
                    s=str(rise_day),
                    verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=fontsize,
                    zorder=date_zorder)

        # hour label
        hour_label_offset = 0.75
        hour_labels = np.arange(min_set_hour+1, max_rise_hour)
        hour_labels_texts = np.copy(hour_labels)
        hour_labels_texts[hour_labels_texts<=0] += 12
        for x_hour, label_text in zip(hour_labels,hour_labels_texts):
            ax.text(
                x_hour,
                1-hour_label_offset,
                str(label_text),
                verticalalignment='bottom',
                horizontalalignment='center',
                zorder=date_zorder,
                fontsize=fontsize
            )
            ax.text(
                x_hour,
                days_in_year+hour_label_offset,
                str(label_text),
                verticalalignment='top',
                horizontalalignment='center',
                zorder=date_zorder,
                fontsize=fontsize
            )


    if background:
        # sky background
        ax.imshow(getbackground(obs, withmoon=False),
                  zorder=-2,
                  extent=[-12, 12, days_in_year + 0.5, 0 + 0.5])

    if grid:
        # grid
        grid_c = [0.75] * 3
        grid_s = 0.5

        xs = np.arange(min_set_hour, max_rise_hour, y_scale_factor)
        xs = xs[np.logical_or(xs < -0.01, xs > 0.01)]
        ys = np.arange(1, days_in_year + 1)

        for y in days_7:
            ax.scatter(xs, [y] * xs.shape[0],
                       c=[grid_c],
                       zorder=-1,
                       s=grid_s,
                       edgecolors='none',
                       lw=0)
        vertical_line_x = np.arange(min_set_hour, max_rise_hour, 0.5)
        vertical_line_x = vertical_line_x[vertical_line_x != 0.0]
        for x in vertical_line_x:
            ax.scatter([x] * ys.shape[0],
                       ys,
                       c=[grid_c],
                       zorder=-1,
                       s=grid_s,
                       edgecolors='none',
                       lw=0)

        # twilight line
        for twilght in dusk_dawn():
            ax.plot(twilght[:, 1],
                    twilght[:, 0],
                    zorder=1,
                    lw=0.5,
                    c=[0.75] * 3,
                    ls=(0, (8, 5)))
    if lines:
        # planet lines
        planets_colors_3 = ['firebrick', 'mediumorchid', 'gold']
        planets_colors_5 = ['aqua', 'darkorange'] + planets_colors_3

        plot_rise(ax, ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn'],
                  zorder=5,
                  lw=0.8,
                  colors=planets_colors_5,
                  object_lable=object_lable)
        plot_transit(ax, ['Mars', 'Jupiter', 'Saturn'],
                     zorder=6,
                     lw=0.8,
                     colors=planets_colors_3,
                     object_lable=object_lable)
        plot_set(ax, ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn'],
                 zorder=7,
                 lw=0.8,
                 colors=planets_colors_5,
                 object_lable=object_lable)

        # stars lines
        plot_rise(ax, ['Star1', 'Star2', 'Star3', 'Star4', 'Star5'],
                  zorder=5 - 1,
                  lw=0.5,
                  colors=['White'] * 5,
                  object_lable=object_lable)
        plot_transit(ax, ['Star1', 'Star2', 'Star3', 'Star4', 'Star5'],
                     zorder=5 - 1,
                     lw=0.5,
                     colors=['White'] * 5,
                     object_lable=object_lable)
    if moon:
        # 朔 望 上下弦
        mq = SearchMoonQuarter(time_origin)

        quarter_day = (mq.time.ut - time_origin.ut) // 1 + 1.0
        quarter_day = int(quarter_day)

        moon_rise_masked = moon_rise[np.logical_and(
            moon_rise[:, 1] >= sun_set[:, 1] - 2,
            moon_rise[:, 1] <= sun_rise[:, 1] + 2)]
        moon_set_masked = moon_set[np.logical_and(
            moon_set[:, 1] >= sun_set[:, 1] - 2,
            moon_set[:, 1] <= sun_rise[:, 1] + 2)]
        quarter_day_list = []
        while quarter_day <= days_in_year:
            quarter_day_list.append(quarter_day)
            moon_quarter_rise = moon_rise_masked[moon_rise_masked[:, 0] ==
                                                 quarter_day]
            moon_quarter_set = moon_set_masked[moon_set_masked[:, 0] ==
                                               quarter_day]
            if moon_quarter_rise.size > 0:  # rise
                draw_moon_symbol(ax=ax,
                                 time=mq.time,
                                 moon_phase=90 * mq.quarter,
                                 lms=lms,
                                 day=quarter_day,
                                 hour=moon_quarter_rise[0, 1],
                                 y_zoom_factor=1 / y_scale_factor,
                                 zoom_factor=0.1,
                                 mares=True)
            if moon_quarter_set.size > 0:  # set
                draw_moon_symbol(
                    ax=ax,
                    time=mq.time,
                    moon_phase=90 * mq.quarter,
                    lms=lms,
                    day=quarter_day,
                    hour=moon_quarter_set[0, 1],  # set
                    y_zoom_factor=1 / y_scale_factor,
                    zoom_factor=0.1,
                    mares=True)

            mq = NextMoonQuarter(mq)
            quarter_day = (mq.time.ut - time_origin.ut) // 1 + 1

        # moon phase
        draw_moons(ax, moon_rise, quarter_day_list)
        draw_moons(ax, moon_set, quarter_day_list)

    # fig basic adjustment
    ax.set_xlim(-12, 12)
    ax.set_ylim(days_in_year, 1)

    ax.axis('off')

    fig_space = 0.05
    fig.subplots_adjust(left=fig_space,
                        bottom=fig_space,
                        right=1 - fig_space,
                        top=1 - fig_space)
    # output scale
    ax.set_aspect(y_scale_factor)

    if not background:
        fig.set_facecolor('grey')

    # plt.show()
    fig.savefig('./output/'+ str(year) + '/' + str(obs.latitude)+'/' + str(obs.latitude) + '_' + filename + '.pdf')


if __name__ == '__main__':
    main_plot(filename='background', background=True)
    main_plot(filename='lines', lines=True)
    main_plot(filename='grid', grid=True)
    main_plot(filename='moon', moon=True)
    main_plot(filename='frame', frame=True)
    main_plot(filename='text', text=True)
    main_plot(filename='meridian_ra', meridian_ra=True)
    main_plot(filename='eot', eot=True)

    main_plot(filename='line_label', lines=True, object_lable=True)

    main_plot(filename='all',
              frame=True,
              lines=True,
              moon=True,
              background=True,
              grid=True,
              eot=True,
              meridian_ra=True,
              text=True)
