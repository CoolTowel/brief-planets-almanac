from astronomy import Time, Body, Observer, SearchRiseSet, Direction
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np

def rise_and_set(body,
                 obs=Observer(40, 0, 100),
                 year=datetime.date.today().year,
                 return_time = False):

    if calendar.isleap(year):
        days_in_year = 366
    else:
        days_in_year = 365

    
    def search(event):
        results_list = []
        days_list = []
        time_list = []
        
        days = np.arange(days_in_year + 1) + 1
        # 加1因为一年最后一天晚上包含第二年的第一天 再+1 因为arange从0开始，
        
        for d in days:
            time = SearchRiseSet(body, obs, event,
                                 Time.Make(year, 1, d, 0, 00, 00.0), 1)
            if time is not None:
                results_list.append(time.hour + time.minute / 60 +
                                    time.second / 3600)
                days_list.append(d)
                time_list.append(time)
        time_list = np.array(time_list)
        results_list = np.array(results_list)
        days_list = np.array(days_list)

        shifted_results_list = np.copy(results_list)
        events_after_noon = results_list >= 12
        shifted_results_list[events_after_noon] -= 24  # 午夜为0时，前晚时间为负数
        days_list[~events_after_noon] -= 1  # 发生在前半夜的事件算入前一天

        event_in_this_year = np.logical_and(days_list>0.5, days_list < days_in_year+0.5) # 排除 day0 和day 366（或367）

        return time_list[event_in_this_year], np.vstack([days_list[event_in_this_year], shifted_results_list[event_in_this_year]])

    rise_time_list, rise = search(Direction.Rise)
    set_time_list, set = search(Direction.Set)
    if not return_time:

        return (rise, set)
    else:
        return (rise_time_list, set_time_list, rise, set)


if __name__ == '__main__':
    moon_rise, moon_set = rise_and_set(Body.Moon)
    sun_r, sun_s = rise_and_set(Body.Sun)
    plt.plot(sun_r[1], sun_r[0])
    plt.plot(sun_s[1], sun_s[0])
    plt.scatter(moon_rise[1], moon_rise[0])
    plt.scatter(moon_set[1], moon_set[0])
    plt.gca().invert_yaxis()
    plt.show()
