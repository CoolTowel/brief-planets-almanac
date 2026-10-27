from utils import ephemeris
from astronomy import Time, Body, Observer, MoonPhase, SearchMoonQuarter, NextMoonQuarter, HourAngle, DefineStar

north40 = Observer(40, 116, 40) # Beijing

obs = north40
ephemeris.ephemeris(obs, plot_immdiately=True)