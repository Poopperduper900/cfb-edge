"""
Weather ingestion. Open-Meteo: no API key, no signup, free for non-commercial
use up to 10k calls/day, CC BY 4.0 (attribution required if you publish).

Gemini was right that the model priced weather it never ingested. This closes
that gap — but with one correction it missed, which matters more than the
ingestion itself.

THE LOOKAHEAD TRAP
------------------
The obvious build is: pull ERA5 reanalysis for historical games, fit the wind
coefficient, deploy against forecasts. That silently overstates every weather
adjustment you will ever make, because the coefficient was estimated on perfect
information and is applied to a noisy forecast.

At kickoff-minus-48h a wind forecast has an RMSE of roughly 3-4 mph. If wind is
worth -0.4 points per mph over 12, and your forecast is off by 3.5 mph, your
adjustment carries ~1.4 points of pure noise — comparable to the entire edge you
are hunting.

The fix: fit on the *Historical Forecast API* (archived operational model runs,
coverage from ~2021-22), not on ERA5 reanalysis. Same variables, same format,
but it is what you would actually have known at bet time. `mode="forecast"`
below does this. ERA5 is kept for descriptive work only.
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from .config import CACHE

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"        # ERA5 reanalysis
HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST = "https://api.open-meteo.com/v1/forecast"

HOURLY = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,relative_humidity_2m"

# Indoor / retractable-roof FBS venues. Weather adjustments MUST be zeroed here;
# forgetting this is the single most common weather-model bug and it produces
# confident garbage on a handful of games every season.
DOME_VENUES = {
    "Ford Field", "Alamodome", "Caesars Superdome", "Mercedes-Benz Stadium",
    "Lucas Oil Stadium", "AT&T Stadium", "NRG Stadium", "Carrier Dome",
    "JMA Wireless Dome", "UNI-Dome", "Fargodome", "DakotaDome",
    "Idaho Central Credit Union Arena", "Kibbie Dome", "State Farm Stadium",
    "Allegiant Stadium", "U.S. Bank Stadium", "Tropicana Field",
    "Georgia State Stadium",  # retractable-adjacent; verify per season
}


def _cache(url: str, params: dict) -> dict:
    key = f"wx_{abs(hash(url + json.dumps(params, sort_keys=True)))}.json"
    p = CACHE / key
    if p.exists():
        return json.loads(p.read_text())
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()
    p.write_text(json.dumps(data))
    return data


def game_weather(
    lat: float,
    lon: float,
    kickoff_utc: pd.Timestamp,
    mode: str = "forecast",
) -> dict:
    """
    Weather at kickoff for one venue.

    mode="forecast"  -> live forecast (upcoming games) or archived operational
                        forecast (backtests). This is the honest one.
    mode="reanalysis"-> ERA5 actuals. Descriptive only. Do NOT fit betting
                        coefficients on this and then deploy them.
    """
    kickoff_utc = pd.Timestamp(kickoff_utc).tz_convert("UTC")
    day = kickoff_utc.date().isoformat()

    if mode == "reanalysis":
        url, extra = ARCHIVE, {"start_date": day, "end_date": day}
    elif kickoff_utc < pd.Timestamp.utcnow():
        url, extra = HIST_FORECAST, {"start_date": day, "end_date": day}
    else:
        url, extra = FORECAST, {"forecast_days": 16}

    data = _cache(
        url,
        {
            "latitude": round(lat, 3),
            "longitude": round(lon, 3),
            "hourly": HOURLY,
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            **extra,
        },
    )

    h = data.get("hourly", {})
    times = pd.to_datetime(h.get("time", []), utc=True)
    if len(times) == 0:
        return {}
    i = int(np.argmin(np.abs(times - kickoff_utc)))

    # Games run ~3.5 hours. Use the window, not the kickoff instant — a total is
    # exposed to the whole game, and wind that arrives in the third quarter
    # counts exactly as much as wind at kickoff.
    j = min(i + 4, len(times) - 1)
    sl = slice(i, j + 1)

    def avg(k, default=np.nan):
        v = h.get(k)
        return float(np.nanmean(v[sl])) if v else default

    return {
        "kickoff_utc": kickoff_utc,
        "temp_f": avg("temperature_2m"),
        "wind_mph": avg("wind_speed_10m"),
        "gust_mph": avg("wind_gusts_10m"),
        "precip_in": float(np.nansum(h.get("precipitation", [0])[sl])),
        "humidity": avg("relative_humidity_2m"),
        "source": url,
        "hours_ahead": float((kickoff_utc - pd.Timestamp.utcnow()).total_seconds() / 3600),
    }


def attach_weather(
    games_df: pd.DataFrame, venues_df: pd.DataFrame, mode: str = "forecast"
) -> pd.DataFrame:
    """Join venue coordinates onto games and pull weather for each."""
    v = venues_df.rename(columns={"name": "venue"})
    coord = v.set_index("venue")[["location.x", "location.y"]] if "location.x" in v else None
    if coord is None:
        coord = v.set_index("venue")[["longitude", "latitude"]]
        coord.columns = ["location.x", "location.y"]

    rows = []
    for _, g in games_df.iterrows():
        venue = g.get("venue")
        if venue in DOME_VENUES:
            rows.append({"gameId": g.get("id"), "dome": True, "wind_mph": 0.0,
                         "temp_f": 70.0, "precip_in": 0.0, "wx_confidence": 1.0})
            continue
        if venue not in coord.index:
            rows.append({"gameId": g.get("id"), "dome": False})
            continue
        lon, lat = coord.loc[venue, "location.x"], coord.loc[venue, "location.y"]
        try:
            wx = game_weather(float(lat), float(lon), g["start_date"], mode=mode)
        except Exception:
            wx = {}
        wx.update(gameId=g.get("id"), dome=False,
                  wx_confidence=forecast_confidence(wx.get("hours_ahead", 0.0)))
        rows.append(wx)
    return games_df.merge(pd.DataFrame(rows), left_on="id", right_on="gameId", how="left")


def forecast_confidence(hours_ahead: float) -> float:
    """
    Shrink the weather adjustment by how far out the forecast is.

    Wind RMSE roughly doubles from 12h to 96h out. Rather than pretend a
    Tuesday forecast for Saturday is as good as a Saturday-morning one, scale
    the whole adjustment. This is why the weather angle is a Friday/Saturday
    play, not a Monday one — and why the market is slow to price it, since most
    of the money is already down by the time the forecast is trustworthy.
    """
    h = max(float(hours_ahead), 0.0)
    return float(np.clip(1.0 - 0.006 * h, 0.35, 1.0))


def total_adjustment(wx: dict, base_sd: float = 12.6) -> tuple[float, float]:
    """
    Returns (points off the total, extra SD).

    Wind is the only weather variable with a large, well-replicated effect on
    scoring. Temperature and precipitation are mostly folklore below the
    extremes — cold games are lower-scoring largely because they are late-season
    games between run-heavy teams, which the ratings already capture. Do not
    double-count that.

    Gusts matter more than sustained wind for passing, which is why both are
    pulled and the gust term carries its own coefficient.
    """
    if wx.get("dome") or not wx:
        return 0.0, 0.0
    conf = wx.get("wx_confidence", 1.0)
    wind = wx.get("wind_mph") or 0.0
    gust = wx.get("gust_mph") or wind
    precip = wx.get("precip_in") or 0.0
    temp = wx.get("temp_f")

    pts = 0.0
    if wind > 10:
        pts -= 0.34 * (wind - 10)
    if gust > 22:
        pts -= 0.18 * (gust - 22)
    if precip > 0.10:
        pts -= 1.2 * min(precip / 0.25, 2.5)
    if temp is not None and temp < 25:
        pts -= 0.04 * (25 - temp)

    extra_sd = 0.9 if wind > 18 else 0.0
    return float(pts * conf), float(extra_sd)
