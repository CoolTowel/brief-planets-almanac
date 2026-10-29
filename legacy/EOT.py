import numpy as np
from astronomy import Time, Body, Observer, SearchRiseSet, Direction, HourAngle
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np

east8 = Observer(40, 120, 40)

time_zone = 8
obs = east8

year = datetime.date.today().year+1
if calendar.isleap(year):
    days_in_year = 366
else:
    days_in_year = 365

days = np.arange(days_in_year) + 1

EOT = []
for day in days:
    ha = HourAngle(Body.Sun,
                   Time.Make(year, 1, day, 24 - time_zone, 00, 00.0),
                   observer=obs) #hour angle
    EOT.append(ha)

EOT = np.array(EOT)

EOT = 12 - EOT

data_path = "./data/" + str(year) + '/'
np.save(data_path + "EOT.npy", EOT)

