"""The shared five-star catalogue used by calculation and plotting."""
from dataclasses import dataclass

from ._vendor import astronomy


@dataclass(frozen=True)
class StarDefinition:
    """J2000 coordinates: right ascension in hours, declination in degrees."""

    body: astronomy.Body
    name: str
    ra: float
    dec: float
    distance_light_years: float = 1000.0


# Keep the original catalogue, precision and identifiers. Negative declinations
# apply the sign to both degrees and arcminutes (the user-authorized correction).
STARS = (
    StarDefinition(astronomy.Body.Star1, "轩辕十四", 10 + 8 / 60, 11 + 57 / 60),
    StarDefinition(astronomy.Body.Star2, "大角", 14 + 15 / 60, 19 + 9 / 60),
    StarDefinition(astronomy.Body.Star3, "织女", 18 + 36 / 60, 38 + 46 / 60),
    StarDefinition(astronomy.Body.Star4, "北落师门", 22 + 57 / 60, -(29 + 37 / 60)),
    StarDefinition(astronomy.Body.Star5, "天狼", 6 + 45 / 60, -(16 + 43 / 60)),
)
STAR_NAMES = tuple(star.body.name for star in STARS)


def register_stars():
    """Reset the vendor library's star slots to the shared catalogue."""
    for star in STARS:
        astronomy.DefineStar(
            star.body, star.ra, star.dec, star.distance_light_years
        )


def star_manifest():
    """Return a JSON-compatible record of the exact catalogue."""
    return [
        {
            "body": star.body.name,
            "name": star.name,
            "ra_hours_j2000": star.ra,
            "dec_degrees_j2000": star.dec,
            "distance_light_years": star.distance_light_years,
        }
        for star in STARS
    ]
