"""
Script Name: mpet_data.py
Purpose: mPET data readers, threshold lookup, and heat-stress analysis
    functions for the CoolScape ICB2026 workshop notebook. Reads acclimatized
    thresholds and published RayMan mPET CSVs, and computes hourly category
    frequencies and prolonged-exposure summaries.
Author(s): Christos Giannaros, One Weather Lab, UoI <chris.giannaros@uoi.gr>
Last updated: 2026-07-07
Version: 0.2.1
License: MIT
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .configuration import (
    HW_PHASES,
    MPET_BINS,
    MPET_CLASSES,
    POPULATION_GROUPS,
    PROLONGED_HOURS,
    SCENARIOS,
    _add_coord_keys,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("owl.coolscape.mpet_data")


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _parse_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day, month columns from date/time strings (top-level CSV format)."""
    df = df.copy()
    # Time format is "H:MM", date format is "DD.M.YYYY " (trailing space)
    df["hour"] = df["time"].str.strip().str.split(":").str[0].astype(int)
    date_parts = df["date"].str.strip().str.split(".")
    df["day"] = date_parts.str[0].astype(int)
    df["month"] = date_parts.str[1].astype(int)
    return df


def _assign_hw(df: pd.DataFrame) -> pd.DataFrame:
    """Assign HW phase from (month, day) using HW_PHASES lookup."""
    df = df.copy()
    # Map each row's (month, day) to its HW phase via the lookup table
    df["hw"] = df.apply(
        lambda r: HW_PHASES.get((r["month"], r["day"]), "unknown"), axis=1,
    )
    return df


# [Threshold readers]

def load_strong_heat_stress_thresholds(data_dir: Path) -> pd.DataFrame:
    """Load the acclimatized strong-heat-stress thresholds CSV.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``acclimatized_strong_heat_stress_thresholds.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``region``, ``group``, ``month``, ``day``, ``strong_heat_stress_threshold_35``.
    """
    path = Path(data_dir) / "acclimatized_strong_heat_stress_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {path}")
    df = pd.read_csv(path)
    return df


def lookup_daily_strong_heat_stress_thresholds(
    thr_df: pd.DataFrame,
    region_label: str,
    group_label: str,
) -> pd.DataFrame:
    """Look up per-day acclimatized strong-heat-stress thresholds.

    Parameters
    ----------
    thr_df : pd.DataFrame
        Output of ``load_strong_heat_stress_thresholds``.
    region_label : str
        Region label (e.g. ``"West Athens"``).
    group_label : str
        Group label (e.g. ``"Female Adults"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``month``, ``day``, ``strong_heat_stress_threshold_35``.
        One row per heat-wave day (9 rows expected).
    """
    mask = (thr_df["region"] == region_label) & (thr_df["group"] == group_label)
    # Defensive copy to avoid modifying the source DataFrame
    subset = thr_df.loc[mask, ["month", "day", "strong_heat_stress_threshold_35"]].copy()
    if subset.empty:
        LOG.warning(
            "No daily thresholds for %s / %s", region_label, group_label,
        )
    return subset.reset_index(drop=True)


# [mPET readers]

def load_mpet_group(
    rayman_dir: Path,
    region: str,
    group: str,
    material: str,
    *,
    base: bool,
) -> pd.DataFrame:
    """Load mPET CSVs for a region/group/material, both HW phases.

    Parameters
    ----------
    rayman_dir : Path
        Directory containing the mPET CSVs.
    region : str
        Key in ``REGIONS``.
    group : str
        Key in ``POPULATION_GROUPS``.
    material : str
        Key in ``SCENARIOS``.
    base : bool
        If True, load the base-case CSV; otherwise the scenario CSV.

    Returns
    -------
    pd.DataFrame
        Concatenated CSV with ``hour``, ``day``, ``month``, ``hw`` added.
    """
    # Build the canonical filename from scenario and group keys
    csv_mat = SCENARIOS[material]["csv"]
    group_label = POPULATION_GROUPS[group]["label"].replace(" ", "_")

    frames = []
    for hw in ("HW0", "HW1"):
        if base:
            fname = f"{region}_{hw}_{group_label}_base_case_for_{csv_mat}.csv"
        else:
            fname = f"{region}_{hw}_{group_label}_{csv_mat}.csv"
        csv_path = Path(rayman_dir) / fname

        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["date", "time", "mPET"])
        else:
            LOG.warning("Missing mPET CSV: %s", fname)
            continue

        # Add hour/day/month from date/time strings, then assign HW phase
        df = _parse_date_time(df)
        df = _assign_hw(df)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No mPET files found for {region}/{group}/{material}"
        )
    return pd.concat(frames, ignore_index=True)


def load_rayman_output(
    rayman_dir: Path,
    region: str,
    hw: str,
    group: str,
    material: str,
    *,
    base: bool,
) -> pd.DataFrame:
    """Load a RayMan mPET CSV with coordinates.

    Parameters
    ----------
    rayman_dir : Path
        Root RayMan output directory. CSVs live directly in this directory.
    region : str
        Key in ``REGIONS``.
    hw : str
        ``"HW0"`` or ``"HW1"``.
    group : str
        Key in ``POPULATION_GROUPS``.
    material : str
        Key in ``SCENARIOS``.
    base : bool
        If True, load the base-case CSV; otherwise the scenario CSV.

    Returns
    -------
    pd.DataFrame
        Schema with ``date``, ``time``, ``lon``, ``lat``, ``mPET``, plus
        ``lon5``, ``lat5``, ``hour``, ``day``, ``month``.
    """
    csv_mat = SCENARIOS[material]["csv"]
    group_label = POPULATION_GROUPS[group]["label"].replace(" ", "_")
    if base:
        fname = f"{region}_{hw}_{group_label}_base_case_for_{csv_mat}.csv"
    else:
        fname = f"{region}_{hw}_{group_label}_{csv_mat}.csv"

    path = Path(rayman_dir) / fname
    if not path.exists():
        raise FileNotFoundError(f"RayMan output CSV not found: {path}")
    df = pd.read_csv(
        path,
        usecols=["date", "time", "lon", "lat", "mPET"],
    )
    # Add rounded coordinate keys and time decomposition
    df = _add_coord_keys(df)
    return df


# [mPET analysis]

def hourly_category_frequencies(
    df: pd.DataFrame,
    *,
    strong_thr: float = 35.0,
) -> pd.DataFrame:
    """Compute hourly frequency (%) of mPET stress categories.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``hour`` (int) and ``mPET`` (float) columns.
    strong_thr : float
        Strong-heat-stress boundary (default 35.0). The extreme-heat
        boundary shifts proportionally: extreme = 41 + (strong_thr - 35).

    Returns
    -------
    pd.DataFrame
        24 rows (hours) x 9 columns (MPET_CLASSES), values in percent.
    """
    # Shift strong and extreme edges with the threshold
    offset = strong_thr - 35.0
    edges = list(MPET_BINS)
    edges[7] = strong_thr
    edges[8] = 41.0 + offset

    labels = MPET_CLASSES
    result = pd.DataFrame(0.0, index=range(24), columns=labels)

    for h in range(24):
        subset = df.loc[df["hour"] == h, "mPET"]
        if subset.empty:
            continue
        # Left-closed intervals: [4, 8) = "Strong cold", [35, 41) = "Strong heat", etc.
        cats = pd.cut(subset, bins=edges, labels=labels, right=False)
        counts = cats.value_counts()
        total = counts.sum()
        if total > 0:
            result.loc[h] = (counts / total * 100).reindex(labels, fill_value=0)

    return result


def count_prolonged_exposure_days(
    df: pd.DataFrame,
    threshold: float | pd.DataFrame,
    *,
    prolonged_hours: int = PROLONGED_HOURS,
) -> pd.DataFrame:
    """Count days with prolonged strong heat-stress exposure per grid point.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched mPET data with ``lon5``, ``lat5``, ``hour``, ``day``,
        ``month``, ``mPET`` columns.
    threshold : float or pd.DataFrame
        If float, constant threshold for all days.
        If DataFrame, must have ``month``, ``day``, ``strong_heat_stress_threshold_35`` columns
        for per-day thresholds.
    prolonged_hours : int
        Hours that must be exceeded (strict ``>``) for a day to qualify.

    Returns
    -------
    pd.DataFrame
        Columns: ``lon5``, ``lat5``, ``n_days``, ``max_exceed_h``.
    """
    df = df.copy()

    # Merge per-day thresholds or apply constant
    if isinstance(threshold, pd.DataFrame):
        df = df.merge(
            threshold[["month", "day", "strong_heat_stress_threshold_35"]],
            on=["month", "day"], how="left",
        )
        df["exceeds"] = df["mPET"] >= df["strong_heat_stress_threshold_35"]
    else:
        df["exceeds"] = df["mPET"] >= threshold

    # Count exceeding hours per grid point per day
    point_day = df.groupby(["lon5", "lat5", "month", "day"]).agg(
        exceed_hours=("exceeds", "sum"),
    ).reset_index()

    # Strict >6h criterion
    point_day["prolonged"] = point_day["exceed_hours"] > prolonged_hours

    # Aggregate across days per grid point
    result = point_day.groupby(["lon5", "lat5"]).agg(
        n_days=("prolonged", "sum"),
        max_exceed_h=("exceed_hours", "max"),
    ).reset_index()

    return result
