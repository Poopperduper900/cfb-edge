"""cfbmodel — college football game + player prop modelling."""
from .config import C, Constants  # noqa: F401

__all__ = [
    "C", "Constants",
    "ingest",      # CFBD v2 API, disk-cached
    "weather",     # Open-Meteo, forecast-horizon shrunk, dome-guarded
    "ratings",     # ridge-adjusted EPA + market-implied power ratings
    "priors",      # preseason: recruiting, returning production, portal
    "qb",          # hierarchical QB value + injury dropoff
    "game_model",  # projection + key-number-aware margin pmf
    "script",      # joint game-script simulation (margin -> volume)
    "props",       # usage priors, efficiency shrinkage, pricing
    "edge",        # devig, EV, Kelly, CLV
    "backtest",    # walk-forward + market-efficiency test
]
__version__ = "0.2.0"
