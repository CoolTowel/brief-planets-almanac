# from turtle import position
# from unittest import skip
from turtle import color
from astronomy import Equator, Horizon, Time, Body, Observer, Refraction, MoonPhase, SearchMoonQuarter, NextMoonQuarter
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from backgound import getbackground
from moon_symbol import draw_moon_logo
"""
Background part: use sun altitude to color the background
draw a image with shape (365, 24*60), 60min *24h for 365 days
"""

y_scale_factor = 1 / 14


def draw_moons(ax, times, days_hours, sun_r_days_hours, sun_s_days_hours):
    for time, day_hour in zip(times, days_hours.T):
        day = int(day_hour[0])
        hour = day_hour[1]
        sun_r_days = sun_r_days_hours[0]
        sun_r_hours = sun_r_days_hours[1]
        sun_r_hour = sun_r_hours[sun_r_days == day][0]

        sun_s_days = sun_s_days_hours[0]
        sun_s_hours = sun_s_days_hours[1]
        sun_s_hour = sun_s_hours[sun_s_days == day][0]

        if hour > sun_s_hour and hour < sun_r_hour:  #（白天的跳过）
            moon_phase = MoonPhase(time)
            # if (moon_phase > 354 or moon_phase < 6) or (
            #         moon_phase > 83.6 and moon_phase < 96.4) or (
            #             moon_phase > 174 and moon_phase < 186) or (
            #                 moon_phase > 263.4 and moon_phase < 276.5):  #朔，上弦，望，下弦

            #     draw_moon_logo(ax=ax,
            #                    time=time,
            #                    moon_phase=moon_phase,
            #                    lms=lms,
            #                    day=day,
            #                    hour=hour,
            #                    y_zoom_factor = 1/y_scale_factor,
            #                    zoom_factor=0.1,
            #                    mares=True)
            # else:
            if hour > sun_s_hour and hour < sun_r_hour:  #非朔望，上下弦，只画在日落后日出前的
                draw_moon_logo(ax=ax,
                               time=time,
                               moon_phase=moon_phase,
                               lms=lms,
                               day=day,
                               hour=hour,
                               y_zoom_factor=1 / y_scale_factor,
                               zoom_factor=0.075,
                               mares=False)


def draw_patch(ax, vertex, color=[1] * 3):
    codes = np.ones(len(vertex),
                    dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO
    path = mpath.Path(vertex, codes)
    patch = mpatches.PathPatch(path, facecolor=color, linewidth=0, zorder=1)
    ax.add_patch(patch)


# 月海文件载入
lms = []
for f in ['lm1.txt', 'lm2.txt', 'lm3.txt']:
    lm = np.loadtxt("./data/" + f)
    lm[:, 1] = -lm[:, 1]  # 最终输出图像y轴反转，故此处反转y轴坐标
    lms.append(lm)

#定义年份
year = datetime.date.today().year

if calendar.isleap(year):
    days_in_year = 366
else:
    days_in_year = 365
"""
先计算太阳和月亮出没时间

"""
data_path = "./data/" + str(year) + '/'

try:
    sun_r_days_hours = np.load(data_path + "sun_r_days_hours.npy")
    sun_s_days_hours = np.load(data_path + "sun_s_days_hours.npy")

    moon_rise_times = np.load(data_path + "moon_rise_times.npy",
                              allow_pickle=True)
    moon_set_times = np.load(data_path + "moon_set_times.npy",
                             allow_pickle=True)
    moon_rise_days_hours = np.load(data_path + "moon_rise_days_hours.npy")
    moon_set_days_hours = np.load(data_path + "moon_set_days_hours.npy")

except IOError:

    print("计算日月出没")
    sun_r_days_hours, sun_s_days_hours = rise_and_set(Body.Sun)

    np.save(data_path + "sun_r_days_hours.npy", sun_r_days_hours)
    np.save(data_path + "sun_s_days_hours.npy", sun_s_days_hours)

    moon_rise_times, moon_set_times, moon_rise_days_hours, moon_set_days_hours = rise_and_set(
        Body.Moon, x=True)

    np.save(data_path + "moon_rise_times.npy", moon_rise_times)
    np.save(data_path + "moon_set_times.npy", moon_set_times)
    np.save(data_path + "moon_rise_days_hours.npy", moon_rise_days_hours)
    np.save(data_path + "moon_set_days_hours.npy", moon_set_days_hours)

sun_r_days = sun_r_days_hours[0]
sun_r_hours = sun_r_days_hours[1]

sun_s_days = sun_s_days_hours[0]
sun_s_hours = sun_s_days_hours[1]

# 绘制

fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)

ax.set_xlim(-12, 12)
ax.set_ylim(days_in_year, 1)

font_path = './fonts/Libertinus-7.040/static/OTF/'
font_files = fm.findSystemFonts(fontpaths=font_path)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Libertinus Sans'
plt.rcParams['font.size'] = 13

# ax.yaxis.set_ticks_position('both')
# ax.yaxis.set_tick_params(colors='black', direction='inout')
# # Move left to centre, passing through (0,0)
# ax.spines['left'].set_position('center')
# ax.spines['left'].set_color('white')
# ax.spines['left'].set_linewidth(0.5)
# # turn off right axis
# ax.spines['right'].set_color('none')
# ax.set_yticklabels([])

# ax.spines['bottom'].set_color('none')
# ax.spines['top'].set_color('none')

ax.axis('off')

# left_white
left_white_vertex = sun_s_days_hours[::-1].T
left_edge = np.array([[-12, days_in_year], [-12, 1]])
left_white_vertex = np.vstack([left_white_vertex, left_edge])
draw_patch(ax, left_white_vertex)

# right_white
right_white_vertex = sun_r_days_hours[::-1].T
right_edge = np.array([[12, days_in_year], [12, 1]])
right_white_vertex = np.vstack([right_white_vertex, right_edge])
draw_patch(ax, right_white_vertex)


# month and date background blue
def month_background(sun_rise_set_days_hours, left=True):
    if left:
        shift = -1
    else:
        shift = 1
    blue_vertex_1 = sun_rise_set_days_hours[::-1].T
    blue_vertex_2 = np.vstack([
        sun_rise_set_days_hours[0][::-1],
        sun_rise_set_days_hours[1][::-1] + shift
    ])[::-1].T
    blue_vertex = np.vstack([blue_vertex_1, blue_vertex_2])
    draw_patch(ax, blue_vertex, color=[226 / 255, 246 / 255, 248 / 255])


month_background(sun_s_days_hours, left=True)
month_background(sun_r_days_hours, left=False)

# left and right black lines
ax.plot(sun_r_hours, sun_r_days, c='k', lw=1)
ax.plot(sun_s_hours, sun_s_days, c='k', lw=1)

ax.plot(sun_r_hours + 1, sun_r_days, c='k', lw=0.5)
ax.plot(sun_s_hours - 1, sun_s_days, c='k', lw=0.5)

# top and butomm line
ax.plot([sun_s_hours[1] - 1, sun_r_hours[1] + 0.985], [1, 1], c='k', lw=1)
ax.plot([sun_s_hours[-1] - 0.985, sun_r_hours[-1] + 0.985],
        [days_in_year, days_in_year],
        c='k',
        lw=1)

# plot days of date
days_7 = np.arange(1, days_in_year, 7)  # days with step of 7

for day in days_7:
    set_hour = sun_s_days_hours[1, day - 1]
    set_day = Time.Make(year, 1, day, 12, 00, 00.0).day
    rise_hour = sun_r_days_hours[1, day - 1]
    rise_day = Time.Make(year, 1, day + 1, 12, 00, 00.0).day

    fontsize = 8.5
    x_shift = 0.3
    y_shift = 0.5
    ax.text(x=set_hour - x_shift,
            y=day + y_shift,
            s=str(set_day),
            verticalalignment='center',
            horizontalalignment='center',
            fontsize=fontsize)
    ax.text(x=rise_hour + x_shift,
            y=day + y_shift,
            s=str(rise_day),
            verticalalignment='center',
            horizontalalignment='center',
            fontsize=fontsize)

# sky background
ax.imshow(getbackground(withmoon=False),
          zorder=-1,
          extent=[-12, 12, days_in_year + 0.5, 0 + 0.5])

# 均时差 equation of time
EOT = np.load(data_path + 'EOT.npy')
days_number = np.arange(days_in_year) + 1
ax.plot(EOT, days_number, color='white', lw=0.5, zorder=0)
# EOT axis
ax.plot([0, 0], [1, days_in_year], color='white', lw=0.5, zorder=0)

# moon phase
draw_moons(ax, moon_rise_times, moon_rise_days_hours, sun_r_days_hours,
           sun_s_days_hours)
draw_moons(ax, moon_set_times, moon_set_days_hours, sun_r_days_hours,
           sun_s_days_hours)

mq = SearchMoonQuarter(Time.Make(year, 1, 1, 12, 00, 00.0))
quarter_day = (mq.time.ut - Time.Make(year, 1, 1, 12, 00, 00.0).ut)//1 + 1

while quarter_day <= days_in_year:
    # rise quarters
    for moon_rs_days_hours in [moon_set_days_hours, moon_rise_days_hours]:

        quarter_day_indice = np.flatnonzero(
            np.asarray(moon_rs_days_hours[0] == quarter_day))
        if quarter_day_indice.size > 0:
            for quarter_day_index in quarter_day_indice:
                hour = moon_rs_days_hours[1, quarter_day_index]
                sun_r_hour = sun_r_hours[sun_r_days == quarter_day][0]
                sun_s_hour = sun_s_hours[sun_s_days == quarter_day][
                    0]  # 太阳出没时间
                if hour > sun_s_hour - 2 and hour < sun_r_hour + 2:
                    draw_moon_logo(ax=ax,
                                   time=mq.time,
                                   moon_phase=90 * mq.quarter + 0.0001,
                                   lms=lms,
                                   day=quarter_day,
                                   hour=hour,
                                   y_zoom_factor=1 / y_scale_factor,
                                   zoom_factor=0.1,
                                   mares=True)

    mq = NextMoonQuarter(mq)
    quarter_day = (mq.time.ut - Time.Make(year, 1, 1, 12, 00, 00.0).ut)//1 + 1

# output scale
ax.set_aspect(y_scale_factor)  # 最好放在最后 但是会影响各种patch的比例，所以patch的比例也需自行调整

plt.show()
# fig.savefig("test.pdf")