"""
Script Name: wrf_fields.py
Purpose: WRF data readers and heat-wave phase identification for the CoolScape
    ICB2026 workshop notebook. Reads precomputed WRF NetCDF aggregates staged
    in data/wrf_400m_output/.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .configuration import HW_PHASES, REGIONS

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("owl.coolscape.wrf_fields")


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [WRF readers]

def load_base_case_area_means(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the precomputed base-case area-mean T2 time series.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``wrf_400m_output/``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are region names from ``REGIONS``; each DataFrame has
        ``time`` (UTC Timestamp) and ``t2`` (degC) columns.
    """
    nc_path = Path(data_dir) / "wrf_400m_output" / "t2_base_case_area_means.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Staged file not found: {nc_path}")

    ds = xr.open_dataset(nc_path)

    # Extract per-region hourly T2 series from the staged NetCDF
    result: dict[str, pd.DataFrame] = {}
    for region_key in REGIONS:
        da = ds["t2"].sel(region=region_key)
        t2_vals = da.values
        # T2 physical range check (degC)
        if np.nanmin(t2_vals) < 0 or np.nanmax(t2_vals) > 55:
            LOG.warning(
                "T2 out of expected range for %s: [%.1f, %.1f]",
                region_key, np.nanmin(t2_vals), np.nanmax(t2_vals),
            )
        result[region_key] = pd.DataFrame({
            "time": pd.to_datetime(ds["time"].values),
            "t2": t2_vals,
        })

    ds.close()
    return result


# [Heat-wave phase spans]

def phase_spans(
    times: pd.DatetimeIndex | np.ndarray,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Identify contiguous HW0 and HW1 spans from a time series.

    Parameters
    ----------
    times : array-like of datetime
        UTC timestamps (hourly).

    Returns
    -------
    tuple[list, list]
        ``(hw0_spans, hw1_spans)``, each a list of ``(start, end)`` pairs
        suitable for ``ax.axvspan``.
    """
    # Extract unique calendar days from hourly timestamps
    times = pd.DatetimeIndex(times)
    days = times.normalize().unique().sort_values()

    hw0_spans: list[tuple[Any, Any]] = []
    hw1_spans: list[tuple[Any, Any]] = []
    current_phase = None
    span_start = None

    # Walk calendar days; on each phase transition, close the previous span
    for d in days:
        phase = HW_PHASES.get((d.month, d.day), None)
        if phase != current_phase:
            # Close previous span
            if current_phase is not None and span_start is not None:
                if current_phase == "HW0":
                    hw0_spans.append((span_start, d))
                else:
                    hw1_spans.append((span_start, d))
            span_start = d
            current_phase = phase

    # Close final span
    if current_phase is not None and span_start is not None:
        span_end = days[-1] + pd.Timedelta(days=1)
        if current_phase == "HW0":
            hw0_spans.append((span_start, span_end))
        else:
            hw1_spans.append((span_start, span_end))

    return hw0_spans, hw1_spans
