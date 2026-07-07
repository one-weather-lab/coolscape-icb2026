"""
Script Name: population.py
Purpose: Population data readers, benefited-point identification, and
    weight-to-count conversion for the CoolScape ICB2026 workshop notebook.
    Reads WorldPop GeoPackage weights and regional-unit population totals.
Author(s): Christos Giannaros, One Weather Lab, UoI <chris.giannaros@uoi.gr>
Last updated: 2026-06-23
Version: 0.1.0
License: MIT
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .configuration import PROLONGED_HOURS, ROUND_LL

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("owl.coolscape.population")


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _scale_marker_radius(
    pop: float,
    max_pop: float,
    *,
    rmin: float = 3.0,
    rmax: float = 15.0,
) -> float:
    """Scale a marker radius linearly between rmin and rmax.

    Parameters
    ----------
    pop : float
        Population value for this marker.
    max_pop : float
        Maximum population across all markers.
    rmin : float
        Minimum marker radius in pixels.
    rmax : float
        Maximum marker radius in pixels.

    Returns
    -------
    float
    """
    if max_pop <= 0:
        return rmin
    return rmin + (rmax - rmin) * (pop / max_pop)


# [Population readers]

def read_gpkg_pop_weights(
    gpkg_path: Path,
    ru_keep: list[int],
    *,
    round_ll: int = ROUND_LL,
) -> pd.DataFrame:
    """Read population weights from a GeoPackage via sqlite3.

    Parameters
    ----------
    gpkg_path : Path
        Path to the ``.gpkg`` file.
    ru_keep : list[int]
        Regional-unit integer codes to retain (e.g. ``[45, 47, 51]``).
    round_ll : int
        Decimal places for coordinate rounding.

    Returns
    -------
    pd.DataFrame
        Columns: ``lon5``, ``lat5``, ``RU_CODE`` (int), ``POP_WEIGHT_SUM``.
    """
    conn = sqlite3.connect(str(gpkg_path))

    # Discover the WRF-prefixed data table in the GeoPackage
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    data_tables = [t for t in tables if t.lower().startswith("wrf")]
    if not data_tables:
        conn.close()
        raise FileNotFoundError(
            f"No WRF table found in {gpkg_path.name}. Available: {tables}"
        )
    tbl_name = data_tables[0]

    # Read centroid coordinates and fractional weight
    df = pd.read_sql_query(
        f'SELECT RU_CODE, POP_WEIGHT_SUM, lon_center, lat_center '
        f'FROM "{tbl_name}"',
        conn,
    )
    conn.close()

    # Coerce types defensively
    df["RU_CODE"] = pd.to_numeric(df["RU_CODE"], errors="coerce").astype(int)
    df["POP_WEIGHT_SUM"] = pd.to_numeric(
        df["POP_WEIGHT_SUM"], errors="coerce",
    ).fillna(0.0)

    # Filter to target regional units and round coordinates for spatial joins
    df = df[df["RU_CODE"].isin(ru_keep)].copy()
    df["lon5"] = df["lon_center"].round(round_ll)
    df["lat5"] = df["lat_center"].round(round_ll)

    # Collapse duplicate centroids within the same regional unit
    pop_map = (
        df.groupby(["lon5", "lat5", "RU_CODE"], as_index=False)
        .agg(POP_WEIGHT_SUM=("POP_WEIGHT_SUM", "sum"))
    )
    return pop_map


def regional_pop_total(
    totals_df: pd.DataFrame,
    ru_str: str,
    pop_col: str,
) -> float:
    """Look up the total population for a regional unit and group.

    Parameters
    ----------
    totals_df : pd.DataFrame
        Population totals with ``RU_CODE`` and group columns.
    ru_str : str
        Regional-unit code (e.g. ``"RU_45"``).
    pop_col : str
        Column name for the population group (e.g. ``"Female Adults"``).

    Returns
    -------
    float
    """
    row = totals_df.loc[totals_df["RU_CODE"] == ru_str, pop_col]
    if row.empty:
        raise ValueError(f"No population total for {ru_str}")
    return float(row.iloc[0])


# [Benefited-point analysis]

def identify_benefited_points(
    base_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    threshold: float | pd.DataFrame,
    *,
    prolonged_hours: int = PROLONGED_HOURS,
) -> pd.DataFrame:
    """Identify grid points with health-related benefit.

    Parameters
    ----------
    base_df : pd.DataFrame
        Enriched base-case mPET data.
    scen_df : pd.DataFrame
        Enriched scenario mPET data.
    threshold : float or pd.DataFrame
        Acclimatized threshold (scalar or per-day DataFrame with
        ``month``, ``day``, ``strong_heat_stress_threshold_35``).
    prolonged_hours : int
        Hours that must be exceeded (strict ``>``) for prolonged exposure.

    Returns
    -------
    pd.DataFrame
        Columns: ``month``, ``day``, ``lon5``, ``lat5``, ``hours_base``,
        ``mpet_base_max``, ``mpet_sce_max``, ``base_exceed``, ``dmpet``,
        ``benefited``.
    """
    # Positional alignment guard (F10)
    assert len(base_df) == len(scen_df), (
        f"Base/scenario length mismatch: {len(base_df)} vs {len(scen_df)}. "
        "Check that both CSVs cover the same grid points and time steps."
    )

    base = base_df.copy()
    scen = scen_df.copy()

    # Apply per-day or constant threshold to determine exceeding hours
    if isinstance(threshold, pd.DataFrame):
        base = base.merge(
            threshold[["month", "day", "strong_heat_stress_threshold_35"]],
            on=["month", "day"], how="left",
        )
        base["exceeds"] = base["mPET"] >= base["strong_heat_stress_threshold_35"]
    else:
        base["exceeds"] = base["mPET"] >= threshold

    # Count exceeding hours per grid point per day (base case)
    base_hours = base.groupby(["lon5", "lat5", "month", "day"]).agg(
        hours_base=("exceeds", "sum"),
    ).reset_index()

    # Strict >6h criterion for prolonged exposure
    base_hours["base_exceed"] = base_hours["hours_base"] > prolonged_hours

    # Independent-max comparison: max mPET over exceeding hours per point-day
    base_exceed_mask = base["exceeds"]
    base_exc = base.loc[base_exceed_mask].groupby(
        ["lon5", "lat5", "month", "day"],
    ).agg(mpet_base_max=("mPET", "max")).reset_index()

    # Scenario max over the same hours (positional alignment, guarded above)
    scen["exceeds_base"] = base["exceeds"].values
    scen_exc = scen.loc[scen["exceeds_base"]].groupby(
        ["lon5", "lat5", "month", "day"],
    ).agg(mpet_sce_max=("mPET", "max")).reset_index()

    # Merge base hours + base max + scenario max
    result = base_hours.merge(
        base_exc, on=["lon5", "lat5", "month", "day"], how="left",
    ).merge(
        scen_exc, on=["lon5", "lat5", "month", "day"], how="left",
    )

    # Only prolonged-exposure points qualify
    result = result[result["base_exceed"]].copy()

    # Delta mPET: negative = cooling benefit
    result["dmpet"] = result["mpet_sce_max"] - result["mpet_base_max"]
    result["benefited"] = result["dmpet"] < 0

    return result[
        ["month", "day", "lon5", "lat5", "hours_base",
         "mpet_base_max", "mpet_sce_max", "base_exceed",
         "dmpet", "benefited"]
    ].reset_index(drop=True)


def weight_to_population(
    benefited_df: pd.DataFrame,
    pop_map: pd.DataFrame,
    total_pop: float,
) -> pd.DataFrame:
    """Convert benefited grid points to absolute population counts.

    Parameters
    ----------
    benefited_df : pd.DataFrame
        Output of ``identify_benefited_points`` (benefited rows only).
    pop_map : pd.DataFrame
        Output of ``read_gpkg_pop_weights``.
    total_pop : float
        Total population for the regional unit and group.

    Returns
    -------
    pd.DataFrame
        Input with ``POP_WEIGHT_SUM`` and ``pop_count`` columns added.
    """
    merged = benefited_df.merge(
        pop_map[["lon5", "lat5", "POP_WEIGHT_SUM"]],
        on=["lon5", "lat5"],
        how="left",
    )

    # Unmatched edge cells filled with zero weight (small fraction, typically <7%)
    merged["POP_WEIGHT_SUM"] = merged["POP_WEIGHT_SUM"].fillna(0)
    # Absolute count = fractional weight x regional total
    merged["pop_count"] = merged["POP_WEIGHT_SUM"] * total_pop

    return merged
