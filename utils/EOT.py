import numpy as np
from astronomy import Time, Body, Observer, SearchRiseSet, Direction, HourAngle
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def EOT(year, obs: Observer):
    # east8 = Observer(40, 120, 40)
    EPH_DIR = os.path.join(DATA_DIR, str(year), str(obs.latitude))
    time_zone = (obs.longitude)/15
    # obs = east8

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
    np.save(os.path.join(EPH_DIR,"EOT.npy"), EOT)
