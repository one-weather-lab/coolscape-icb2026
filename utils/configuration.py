"""
Script Name: configuration.py
Purpose: Configuration dictionaries and shared helpers for the CoolScape
    ICB2026 workshop notebook. Single source of truth for scenario, region,
    population-group, and mPET category definitions.
Author(s): Christos Giannaros, One Weather Lab, UoI <chris.giannaros@uoi.gr>
Last updated: 2026-06-23
Version: 0.1.0
License: MIT
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

from math import inf
from typing import Any

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration [Scenarios]
# -----------------------------------------------------------------------------
# Canonical key -> record.

SCENARIOS: dict[str, dict[str, str]] = {
    "white_paint": {
        "csv": "white_or_cool-colored_paint",
        "nc": "white_or_cool_colored_paint",
        "label": "White/Cool-colored paint",
        "color": "#a6cee3",
    },
    "reflective": {
        "csv": "reflective_coating",
        "nc": "reflective_coating",
        "label": "Reflective coating",
        "color": "#1f78b4",
    },
    "supercool": {
        "csv": "super-cool_material",
        "nc": "super_cool_material",
        "label": "Super-cool material",
        "color": "#08519c",
    },
}


# -----------------------------------------------------------------------------
# Configuration [Regions]
# -----------------------------------------------------------------------------
# ``pixels`` is the regional-unit cooling-grid-point denominator for
# benefited-area %.

REGIONS: dict[str, dict[str, Any]] = {
    "Central_Athens": {
        "label": "Central Athens",
        "ru_str": "RU_45",
        "ru_int": 45,
        "pixels": 551,
    },
    "West_Athens": {
        "label": "West Athens",
        "ru_str": "RU_47",
        "ru_int": 47,
        "pixels": 427,
    },
    "Piraeus": {
        "label": "Piraeus",
        "ru_str": "RU_51",
        "ru_int": 51,
        "pixels": 328,
    },
}


# -----------------------------------------------------------------------------
# Configuration [Population Groups]
# -----------------------------------------------------------------------------

POPULATION_GROUPS: dict[str, dict[str, str]] = {
    "Female_Adults": {
        "label": "Female Adults",
        "gpkg": "WRF_400m_PopWeights_Female_Adults_2021.gpkg",
        "pop_col": "Female Adults",
    },
    "Male_Adults": {
        "label": "Male Adults",
        "gpkg": "WRF_400m_PopWeights_Male_Adults_2021.gpkg",
        "pop_col": "Male Adults",
    },
    "Female_Seniors": {
        "label": "Female Seniors",
        "gpkg": "WRF_400m_PopWeights_Female_Seniors_2021.gpkg",
        "pop_col": "Female Seniors",
    },
    "Male_Seniors": {
        "label": "Male Seniors",
        "gpkg": "WRF_400m_PopWeights_Male_Seniors_2021.gpkg",
        "pop_col": "Male Seniors",
    },
}


# -----------------------------------------------------------------------------
# Configuration [mPET Categories]
# -----------------------------------------------------------------------------
# Chen & Matzarakis (2018) standard thermal stress classification.

MPET_BINS: list[float] = [-inf, 4, 8, 13, 18, 23, 29, 35, 41, inf]

MPET_CLASSES: list[str] = [
    "Extreme cold",
    "Strong cold",
    "Moderate cold",
    "Slight cold",
    "No thermal stress",
    "Slight heat",
    "Moderate heat",
    "Strong heat",
    "Extreme heat stress",
]

MPET_COLORS: list[str] = [
    "#3366FF",   # Extreme cold
    "#3385DF",   # Strong cold
    "#33A5FF",   # Moderate cold
    "#33FFED",   # Slight cold
    "#75FF33",   # No thermal stress
    "#FFF033",   # Slight heat
    "#FFA233",   # Moderate heat
    "#FF5D33",   # Strong heat
    "#FF3333",   # Extreme heat stress
]


# -----------------------------------------------------------------------------
# Configuration [Heat-Wave Phases]
# -----------------------------------------------------------------------------
# Keys are (month, day) tuples; values are "HW0" (Etesians) or "HW1" (sea
# breeze).  Covers the nine-day heat wave 28 Jul – 5 Aug 2021.

HW_PHASES: dict[tuple[int, int], str] = {
    (7, 28): "HW0",
    (7, 29): "HW0",
    (7, 30): "HW1",
    (7, 31): "HW1",
    (8, 1): "HW1",
    (8, 2): "HW1",
    (8, 3): "HW0",
    (8, 4): "HW0",
    (8, 5): "HW1",
}

HW_DAYS: dict[str, int] = {"HW0": 4, "HW1": 5}

# Heatmap column ordering (region x HW phase)
HEATMAP_COLUMNS: list[str] = [
    "Central_Athens_HW0", "Central_Athens_HW1",
    "West_Athens_HW0", "West_Athens_HW1",
    "Piraeus_HW0", "Piraeus_HW1",
]

# Time windows (UTC)
DAYTIME_WINDOW: tuple[int, int] = (7, 20)   # 07:00–20:00 UTC
NIGHT_WINDOW: tuple[int, int] = (21, 6)     # 21:00–06:00 UTC


# -----------------------------------------------------------------------------
# Configuration [Analysis Constants]
# -----------------------------------------------------------------------------

# Coordinate rounding precision for spatial joins
ROUND_LL: int = 5

# Prolonged-exposure threshold (strict >6 hours per day)
PROLONGED_HOURS: int = 6


# -----------------------------------------------------------------------------
# Configuration [Display]
# -----------------------------------------------------------------------------

# Per-region colors for time-series and overview plots
REGION_COLORS: dict[str, str] = {
    "Central_Athens": "#d62728",
    "West_Athens": "#e67e22",
    "Piraeus": "#2ca02c",
}

# Map extent for panel maps
MAP_EXTENT: list[float] = [23.52, 23.98, 37.8, 38.16]


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _add_coord_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add rounded coordinate keys and time decomposition columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``lon``, ``lat``, ``date``, ``time`` columns.

    Returns
    -------
    pd.DataFrame
        Input with added ``lon5``, ``lat5``, ``hour``, ``day``, ``month``.
    """
    df = df.copy()
    df["lon5"] = df["lon"].round(ROUND_LL)
    df["lat5"] = df["lat"].round(ROUND_LL)
    # Parse time: format is "H:MM"
    df["hour"] = df["time"].str.strip().str.split(":").str[0].astype(int)
    # Parse date: format is "DD.M.YYYY " (trailing space)
    date_parts = df["date"].str.strip().str.split(".")
    df["day"] = date_parts.str[0].astype(int)
    df["month"] = date_parts.str[1].astype(int)
    return df


def _days_in_hw(hw: str) -> list[tuple[int, int]]:
    """Return the (month, day) tuples belonging to a given HW phase.

    Parameters
    ----------
    hw : str
        ``"HW0"`` or ``"HW1"``.

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of (month, day) tuples.
    """
    return sorted(k for k, v in HW_PHASES.items() if v == hw)
