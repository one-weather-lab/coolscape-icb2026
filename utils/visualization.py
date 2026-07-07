"""
Script Name: visualization.py
Purpose: Visualization helpers and plot builders for the CoolScape ICB2026
    workshop notebook. Colormaps, shapefile overlays, and section-level
    figure builders.
Author(s): Christos Giannaros, One Weather Lab, UoI <chris.giannaros@uoi.gr>
Last updated: 2026-07-06
Version: 0.2.1
License: MIT
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import shapefile
import xarray as xr
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from IPython.display import Image, display

from .configuration import (
    HW_PHASES,
    MAP_EXTENT,
    MPET_CLASSES,
    MPET_COLORS,
    POPULATION_GROUPS,
    REGION_COLORS,
    REGIONS,
    SCENARIOS,
    _days_in_hw,
)
from .mpet_data import (
    count_prolonged_exposure_days,
    hourly_category_frequencies,
    load_enriched,
    load_mpet_group,
    lookup_daily_strong_heat_stress_thresholds,
)
from .population import (
    _scale_marker_radius,
    identify_benefited_points,
    read_gpkg_pop_weights,
    regional_pop_total,
    weight_to_population,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("owl.coolscape.visualization")


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Shapefile overlays]

def draw_shapefile_overlay(
    ax: plt.Axes,
    shp_path: Path,
    *,
    color: str = "black",
    linewidth: float = 1.0,
) -> None:
    """Draw shapefile polygon boundaries on a matplotlib Axes.

    Parameters
    ----------
    ax : plt.Axes
        Target axes (cartopy GeoAxes or plain matplotlib Axes).
    shp_path : Path
        Path to the shapefile.
    color : str
        Line color.
    linewidth : float
        Line width.
    """
    sf = shapefile.Reader(str(shp_path))
    for shp_rec in sf.shapes():
        pts = np.array(shp_rec.points)
        parts = list(shp_rec.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            # Use cartopy projection transform if available
            transform = (
                ax.projection if hasattr(ax, "projection") else ax.transData
            )
            ax.plot(
                ring[:, 0], ring[:, 1],
                color=color, linewidth=linewidth, transform=transform,
            )


def add_shapefile_to_map(
    folium_map,
    shp_path: Path,
    *,
    color: str = "black",
    weight: float = 2.0,
    tooltip: str | None = None,
) -> None:
    """Add shapefile polygon boundaries to a folium Map.

    Parameters
    ----------
    folium_map : folium.Map
        Target map.
    shp_path : Path
        Path to the shapefile.
    color : str
        Line color.
    weight : float
        Line weight.
    tooltip : str or None
        Optional tooltip text.
    """
    import folium

    sf = shapefile.Reader(str(shp_path))
    for shp_rec in sf.shapes():
        pts = shp_rec.points
        parts = list(shp_rec.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            # folium expects (lat, lon)
            coords = [(lat, lon) for lon, lat in ring]
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                tooltip=tooltip,
            ).add_to(folium_map)


# [Colormaps]

def make_diverging_white_cmap(
    clevs: np.ndarray,
    *,
    base: str = "RdBu_r",
    white_band: float = 0.05,
) -> tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    """Create a diverging colormap with a white band centered at zero.

    Parameters
    ----------
    clevs : np.ndarray
        Contour levels (must span zero).
    base : str
        Base matplotlib colormap name.
    white_band : float
        Half-width of the zero-centered white band.

    Returns
    -------
    tuple[Colormap, BoundaryNorm]
    """
    base_cmap = plt.get_cmap(base)
    n = len(clevs) - 1
    colors = []
    for i in range(n):
        mid = (clevs[i] + clevs[i + 1]) / 2
        if -white_band <= mid <= white_band:
            colors.append("white")
        else:
            frac = (mid - clevs[0]) / (clevs[-1] - clevs[0])
            colors.append(base_cmap(frac))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm


def make_sequential_cooling_cmap(
    values: np.ndarray,
    *,
    base: str = "Blues",
    white_thresh: float = -0.05,
    n_levels: int = 41,
) -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, np.ndarray]:
    """Create a sequential colormap for cooling values, white above threshold.

    Parameters
    ----------
    values : np.ndarray
        Data values (negative = cooling).
    base : str
        Base matplotlib colormap name.
    white_thresh : float
        Values above this are mapped to white (no cooling).
    n_levels : int
        Number of contour levels.

    Returns
    -------
    tuple[Colormap, BoundaryNorm, ndarray]
        ``(cmap, norm, clevs)``.
    """
    vmin = float(np.nanmin(values))
    vmax = 0.0
    clevs = np.linspace(vmin, vmax, n_levels)

    base_cmap = plt.get_cmap(base)
    n = len(clevs) - 1
    colors = []
    for i in range(n):
        mid = (clevs[i] + clevs[i + 1]) / 2
        if mid >= white_thresh:
            colors.append("white")
        else:
            # Deeper blue for stronger cooling
            frac = (0.0 - mid) / (0.0 - vmin) if vmin != 0 else 0
            colors.append(base_cmap(frac))

    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad("lightgrey", alpha=0.4)
    norm = mcolors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm, clevs


# [Atmospheric cooling plots]

def plot_base_t2_timeseries(
    region_series: dict,
    hw0_spans: list,
    hw1_spans: list,
    out_dir: Path,
) -> None:
    """Single-panel base-case T2 time series with HW phase shading.

    Parameters
    ----------
    region_series : dict[str, pd.DataFrame]
        Output of ``load_base_case_area_means``.
    hw0_spans : list[tuple]
        HW0 (Etesians) span pairs from ``phase_spans``.
    hw1_spans : list[tuple]
        HW1 (sea breeze) span pairs from ``phase_spans``.
    out_dir : Path
        Output directory for saved figure.
    """
    times_utc = region_series["Central_Athens"]["time"]

    fig, ax = plt.subplots(figsize=(14, 4.5))

    for region_key, rinfo in REGIONS.items():
        df = region_series[region_key]
        ax.plot(
            df["time"], df["t2"],
            color=REGION_COLORS[region_key], linewidth=0.9,
            marker="o", markersize=3, markevery=3,
            label=rinfo["label"],
        )

    # Hatched background shading: /// for Etesians, \\ for sea breeze
    for start, end in hw0_spans:
        ax.axvspan(start, end, facecolor="white", edgecolor="0.6",
                   hatch="///", alpha=0.35, zorder=0)
    for start, end in hw1_spans:
        ax.axvspan(start, end, facecolor="white", edgecolor="0.6",
                   hatch="\\\\", alpha=0.35, zorder=0)

    ax.set_ylabel("T2 (°C)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_xlim(times_utc.iloc[0], times_utc.iloc[-1])

    # Combined legend: regions + HW phases
    region_handles, _ = ax.get_legend_handles_labels()
    phase_handles = [
        Patch(facecolor="white", edgecolor="0.4", hatch="///",
              label="HW0 (Etesians)"),
        Patch(facecolor="white", edgecolor="0.4", hatch="\\\\",
              label="HW1 (sea breeze)"),
    ]
    ax.legend(
        handles=region_handles + phase_handles,
        loc="upper left", fontsize=9, framealpha=0.8, ncol=2,
    )

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_1.1_base_t2_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_diff_panel_maps(
    wrf_dir: Path,
    shp_aua: Path,
    shp_attica: Path,
    shp_elaionas: Path,
    out_dir: Path,
) -> None:
    """2x3 daytime delta-T2 panel maps with wind vectors and shapefile overlays.

    Parameters
    ----------
    wrf_dir : Path
        Directory containing ``t2_diff_daytime_means.nc``.
    shp_aua : Path
        Regional-units shapefile.
    shp_attica : Path
        Attica boundary shapefile.
    shp_elaionas : Path
        Elaionas intervention zone shapefile.
    out_dir : Path
        Output directory for saved figure.
    """
    ds = xr.open_dataset(wrf_dir / "t2_diff_daytime_means.nc")
    lat_2d = ds["XLAT"].values
    lon_2d = ds["XLONG"].values

    # Symmetric range centered on zero for the diverging colormap
    clevs = np.linspace(-0.8, 0.8, 41)
    cmap, norm = make_diverging_white_cmap(clevs)

    hw_codes = ["HW0", "HW1"]
    sc_keys = list(SCENARIOS.keys())

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(16.5, 7.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    plt.subplots_adjust(
        left=0.06, right=0.90, top=0.94, bottom=0.06,
        wspace=0.02, hspace=0.04,
    )

    # Subsample wind vectors for readability on the 135x135 grid
    quiver_skip = 8
    cf_last = None

    for row, hw in enumerate(hw_codes):
        hw_lc = hw.lower()
        u10 = ds[f"u10_{hw_lc}"].values
        v10 = ds[f"v10_{hw_lc}"].values

        for col, sc_key in enumerate(sc_keys):
            ax = axes[row, col]
            nc_key = SCENARIOS[sc_key]["nc"]
            diff_2d = ds[f"dt2_{nc_key}_{hw_lc}"].values

            cf = ax.contourf(
                lon_2d, lat_2d, diff_2d,
                levels=clevs, cmap=cmap, norm=norm,
                extend="both", transform=ccrs.PlateCarree(),
            )
            cf_last = cf

            s = quiver_skip
            q = ax.quiver(
                lon_2d[::s, ::s], lat_2d[::s, ::s],
                u10[::s, ::s], v10[::s, ::s],
                scale_units="xy", scale=100,
                width=0.0018, headwidth=3.5, headlength=5,
                headaxislength=4.5, color="k", alpha=0.9,
                transform=ccrs.PlateCarree(),
            )

            draw_shapefile_overlay(ax, shp_aua, color="0.3", linewidth=0.4)
            draw_shapefile_overlay(ax, shp_attica, color="0.3", linewidth=0.4)
            draw_shapefile_overlay(ax, shp_elaionas, color="red", linewidth=0.7)

            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

            # Gridlines: labels on left column and bottom row only
            gl = ax.gridlines(
                draw_labels=True, linewidth=0, color="grey",
                x_inline=False, y_inline=False,
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (col == 0)
            gl.bottom_labels = (row == 1)
            gl.xlocator = mticker.FixedLocator(np.arange(23.6, 24.0, 0.1))
            gl.ylocator = mticker.FixedLocator(np.arange(37.9, 38.2, 0.1))
            gl.xlabel_style = {"size": 10}
            gl.ylabel_style = {"size": 10}

            # Scenario title on top row, HW phase label on each panel
            if row == 0:
                ax.set_title(SCENARIOS[sc_key]["label"], fontsize=14)
            ax.text(
                0.01, 0.98, hw, transform=ax.transAxes,
                ha="left", va="top", fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=2.0),
            )

    # Reference wind vector key
    pos = axes[1, 2].get_position()
    ax_key = fig.add_axes([
        pos.x0 + 0.58 * pos.width, pos.y0 - 0.045,
        0.32 * pos.width, 0.05,
    ])
    ax_key.axis("off")
    ax_key.quiverkey(
        q, X=0.8, Y=-0.1, U=5,
        label=r"5 m s$^{-1}$", labelpos="E",
        coordinates="axes", fontproperties=FontProperties(size=11),
    )

    # Shared colorbar
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.80])
    cb = fig.colorbar(cf_last, cax=cax)
    cb.set_label(r"$\Delta$T ($\degree$C)", fontsize=12)
    cb.ax.tick_params(labelsize=12)

    fig.savefig(str(out_dir / "fig_1.2_diff_panel_maps.png"), dpi=300)
    # display the figure in Jupyter without saving to file
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=plt.rcParams["figure.dpi"])
    plt.close(fig)
    display(Image(data=buf.getvalue()))

    ds.close()


def plot_cooling_heatmap(
    wrf_dir: Path,
    out_dir: Path,
) -> None:
    """3-panel hourly cooling heatmap by scenario.

    Parameters
    ----------
    wrf_dir : Path
        Directory containing ``t2_hourly_cooling_heatmaps.nc``.
    out_dir : Path
        Output directory for saved figure.
    """
    ds = xr.open_dataset(wrf_dir / "t2_hourly_cooling_heatmaps.nc")

    # Collect data and determine global color range
    heatmaps = {}
    all_values = []
    for sc_key in SCENARIOS:
        nc_key = SCENARIOS[sc_key]["nc"]
        data = ds[f"heatmap_{nc_key}"].values
        heatmaps[sc_key] = data
        all_values.append(data[np.isfinite(data)])

    valid = np.concatenate(all_values)
    cmap_hm, norm_hm, _ = make_sequential_cooling_cmap(valid)

    # X-axis layout: paired HW0/HW1 columns per region (3 pairs = 6 columns)
    n_cols = 6
    hw_tick_labels = ["HW0", "HW1", "HW0", "HW1", "HW0", "HW1"]
    region_names = ["Central\nAthens", "West\nAthens", "Piraeus"]
    pair_centers = [0.5, 2.5, 4.5]

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(
        nrows=1, ncols=4, width_ratios=[1, 1, 1, 0.045], wspace=0.28,
    )
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for idx, sc_key in enumerate(SCENARIOS):
        ax = axs[idx]
        z = np.ma.masked_invalid(heatmaps[sc_key])
        im = ax.imshow(
            z, origin="lower", aspect="auto", cmap=cmap_hm, norm=norm_hm,
            extent=[-0.5, n_cols - 0.5, -0.5, 23.5],
        )

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(hw_tick_labels, rotation=0, fontsize=9)
        for xc, reg in zip(pair_centers, region_names):
            ax.text(xc, -0.07, reg, ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=9)

        # Vertical separators between region pairs
        for xsep in [1.5, 3.5]:
            ax.axvline(xsep, color="black", lw=0.8, linestyle="--")

        ax.set_yticks(range(24))
        ax.set_yticklabels([str(h) for h in range(24)], fontsize=9)
        ax.set_title(SCENARIOS[sc_key]["label"], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Hour of Day (UTC)", fontsize=11)

    cbar = fig.colorbar(im, cax=cax, extend="min")
    cbar.set_label(r"$\Delta$T ($\degree$C)", fontsize=11)
    cbar.ax.tick_params(labelsize=11)
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()

    fig.savefig(str(out_dir / "fig_1.3_cooling_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.show()
    ds.close()


# [Heat-stress plots]

def plot_mpet_single_panel(
    df: "pd.DataFrame",
    strong_thr: float,
    group_label: str,
) -> None:
    """Single-panel stacked bar chart of mPET categories at a given threshold.

    Parameters
    ----------
    df : pd.DataFrame
        mPET data with ``hour`` and ``mPET`` columns.
    strong_thr : float
        Strong-heat-stress boundary for category binning.
    group_label : str
        Population group label for the title.
    """
    freq = hourly_category_frequencies(df, strong_thr=strong_thr)
    hours = np.arange(24)
    bottom = np.zeros(24)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Stacked bars: each mPET category stacks on the previous
    for i, cls in enumerate(MPET_CLASSES):
        vals = freq[cls].values
        ax.bar(hours, vals, bottom=bottom, width=0.85,
               color=MPET_COLORS[i], alpha=0.85,
               edgecolor="none", label=cls)
        bottom += vals

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(list(range(0, 24, 3)) + [23])
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in list(range(0, 24, 3)) + [23]],
        fontsize=10,
    )
    ax.set_ylabel("Frequency of Occurrence (%)", fontsize=11)
    ax.set_xlabel("Hours of Day (UTC)", fontsize=11)

    # Live exceedance statistics in the title
    n_total = len(df)
    offset = strong_thr - 35.0
    strong_pct = (df["mPET"] >= strong_thr).sum() / n_total * 100
    extreme_thr = 41.0 + offset
    extreme_pct = (df["mPET"] >= extreme_thr).sum() / n_total * 100
    ax.set_title(
        f"{group_label} | Threshold: {strong_thr:.1f} "
        r"$\degree$C" + f"  |  Strong: {strong_pct:.1f}%"
        f"  |  Extreme: {extreme_pct:.1f}%",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.8)
    plt.tight_layout()
    plt.show()


def interactive_threshold_toggle(
    rayman_dir: Path,
    region: str,
    group_key: str,
    material: str,
    thr_df: "pd.DataFrame",
    focus_day: tuple[int, int],
) -> None:
    """Create and display a binary toggle comparing fixed vs. acclimatized thresholds.

    Parameters
    ----------
    rayman_dir : Path
        Directory containing the mPET CSVs.
    region : str
        Key in ``REGIONS``.
    group_key : str
        Key in ``POPULATION_GROUPS``.
    material : str
        Key in ``SCENARIOS``.
    thr_df : pd.DataFrame
        Output of ``load_strong_heat_stress_thresholds``.
    focus_day : tuple[int, int]
        (month, day) of the analysis day (e.g. ``(8, 3)`` for 3 August).
    """
    import ipywidgets as widgets
    from IPython.display import display

    group_label = POPULATION_GROUPS[group_key]["label"]
    region_label = REGIONS[region]["label"]
    month, day = focus_day

    # Look up the exact acclimatized threshold for the focus day
    daily = lookup_daily_strong_heat_stress_thresholds(thr_df, region_label, group_label)
    day_row = daily[(daily["month"] == month) & (daily["day"] == day)]
    if day_row.empty:
        LOG.warning("No threshold for %s/%s on %d/%d", region_label, group_label, day, month)
        accli_val = 35.0
    else:
        accli_val = float(day_row["strong_heat_stress_threshold_35"].iloc[0])

    # Load base-case mPET and filter to the focus day only
    df_all = load_mpet_group(rayman_dir, region, group_key, material, base=True)
    df = df_all[(df_all["month"] == month) & (df_all["day"] == day)].copy()

    output = widgets.Output()

    def _update(thr):
        output.clear_output(wait=True)
        with output:
            plot_mpet_single_panel(
                df=df, strong_thr=thr,
                group_label=f"{group_label} — {day:02d}/{month:02d}",
            )

    # Binary toggle: Fixed (35 degC) vs Acclimatized for the focus day
    toggle = widgets.ToggleButtons(
        options=[
            (f"Fixed (35.0 °C)", 35.0),
            (f"Acclimatized {day:02d}/{month:02d} ({accli_val:.1f} °C)", accli_val),
        ],
        description="Threshold:",
        style={"description_width": "80px", "button_width": "280px"},
    )

    toggle.observe(lambda change: _update(change["new"]), names="value")

    display(toggle)
    display(output)
    _update(35.0)


def plot_mpet_bioclimate_diagram(
    rayman_dir: Path,
    region: str,
    material: str,
    out_dir: Path,
) -> None:
    """2x2 mPET thermal bioclimate diagram for all four groups.

    Parameters
    ----------
    rayman_dir : Path
        Directory containing the mPET CSVs.
    region : str
        Key in ``REGIONS`` (e.g. ``"West_Athens"``).
    material : str
        Key in ``SCENARIOS``.
    out_dir : Path
        Output directory for saved figure.
    """
    region_label = REGIONS[region]["label"]

    # Panel layout: rows = Female/Male, cols = Adults/Seniors
    panel_layout = [
        [("Female_Adults", "Female Adults"), ("Female_Seniors", "Female Seniors")],
        [("Male_Adults", "Male Adults"), ("Male_Seniors", "Male Seniors")],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)

    for r, row_groups in enumerate(panel_layout):
        for c, (gk, gl) in enumerate(row_groups):
            ax = axes[r, c]
            # Load base-case mPET for the analysis region, both HW phases
            df = load_mpet_group(
                rayman_dir, region, gk, material, base=True,
            )
            # Bin mPET into 9 stress categories per UTC hour
            freq = hourly_category_frequencies(df, strong_thr=35.0)
            hours = np.arange(24)
            bottom = np.zeros(24)

            # Stacked bars: each category stacks on the previous
            for i, cls in enumerate(MPET_CLASSES):
                vals = freq[cls].values
                ax.bar(
                    hours, vals, bottom=bottom, width=0.85,
                    color=MPET_COLORS[i],
                    # Legend entries from the first panel only
                    label=cls if (r == 0 and c == 0) else None,
                    alpha=0.85, edgecolor="none",
                )
                bottom += vals

            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(0, 100)
            ax.set_xticks(list(range(0, 24, 3)) + [23])
            ax.set_xticklabels(
                [f"{h:02d}:00" for h in list(range(0, 24, 3)) + [23]],
                fontsize=10,
            )
            ax.set_title(gl, fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", labelsize=10)
            if c == 0:
                ax.set_ylabel("Frequency of Occurrence (%)", fontsize=11)
            if r == 1:
                ax.set_xlabel("Hours of Day (UTC)", fontsize=11)

    # Shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=11,
        bbox_to_anchor=(0.5, -0.01), frameon=False,
    )

    fig.suptitle(
        f"mPET Thermal Bioclimate Diagram at Cooling Grid Points: {region_label} | Base Case",
        fontsize=14, fontweight="bold", y=1.01,
    )
    # Reserve bottom margin for the shared legend
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    fig.savefig(str(out_dir / "fig_2.1_mpet_bioclimate.png"),
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_acclimatization_summary(
    rayman_dir: Path,
    region: str,
    thr_df: "pd.DataFrame",
    material: str,
    out_dir: Path,
) -> None:
    """Grouped bar chart comparing fixed vs. per-day acclimatized exceedance.

    Parameters
    ----------
    rayman_dir : Path
        Directory containing the mPET CSVs.
    region : str
        Key in ``REGIONS`` (e.g. ``"West_Athens"``).
    thr_df : pd.DataFrame
        Output of ``load_thresholds``.
    material : str
        Key in ``SCENARIOS``.
    out_dir : Path
        Output directory for saved figure.
    """
    region_label = REGIONS[region]["label"]

    all_groups = [
        ("Female_Adults", "Female Adults"),
        ("Male_Adults", "Male Adults"),
        ("Female_Seniors", "Female Seniors"),
        ("Male_Seniors", "Male Seniors"),
    ]

    group_labels = [gl for _, gl in all_groups]
    # Accumulate exceedance percentages per group for the four bar categories
    fixed_strong, accli_strong = [], []
    fixed_extreme, accli_extreme = [], []

    for gk, gl in all_groups:
        df = load_mpet_group(rayman_dir, region, gk, material, base=True)
        n = len(df)

        # Join per-day acclimatized thresholds to each mPET hour
        daily_thr = lookup_daily_strong_heat_stress_thresholds(thr_df, region_label, gl)
        df_with_thr = df.merge(
            daily_thr[["month", "day", "strong_heat_stress_threshold_35"]],
            on=["month", "day"], how="left",
        )
        # Offset shifts the extreme boundary proportionally with the strong boundary
        offset_col = df_with_thr["strong_heat_stress_threshold_35"] - 35.0

        # Fixed thresholds: constant 35 / 41 degC for all hours
        fixed_strong.append((df["mPET"] >= 35.0).sum() / n * 100)
        fixed_extreme.append((df["mPET"] >= 41.0).sum() / n * 100)

        # Per-day acclimatized: each hour evaluated against its day's threshold
        accli_strong.append(
            (df_with_thr["mPET"] >= df_with_thr["strong_heat_stress_threshold_35"]).sum() / n * 100
        )
        accli_extreme.append(
            (df_with_thr["mPET"] >= (41.0 + offset_col)).sum() / n * 100
        )

    x = np.arange(len(group_labels))
    width = 0.18  # bar width for four side-by-side bars per group

    fig, ax = plt.subplots(figsize=(12, 5.5))
    # Four bars per group: alpha and edge distinguish fixed (faded) vs acclimatized (solid+border)
    bars = [
        ax.bar(x - 1.5 * width, fixed_strong, width, color="#FF5D33",
               alpha=0.6, edgecolor="none",
               label=r"Strong heat stress (fixed 35 $\degree$C)"),
        ax.bar(x - 0.5 * width, accli_strong, width, color="#FF5D33",
               alpha=1.0, edgecolor="black", linewidth=0.8,
               label="Strong heat stress (acclimatized)"),
        ax.bar(x + 0.5 * width, fixed_extreme, width, color="#FF3333",
               alpha=0.4, edgecolor="none",
               label=r"Extreme heat stress (fixed 41 $\degree$C)"),
        ax.bar(x + 1.5 * width, accli_extreme, width, color="#FF3333",
               alpha=0.8, edgecolor="black", linewidth=0.8,
               label="Extreme heat stress (acclimatized)"),
    ]

    # Value annotations above each bar
    for bar_set in bars:
        for bar in bar_set:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Exceedance (%)", fontsize=12)
    ax.set_ylim(0, max(fixed_strong) + 8)
    ax.set_title(
        f"Fixed vs. Acclimatized Heat Stress Thresholds: {region_label} | Base Case",
        fontsize=13, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(
        loc="lower center", ncol=2, fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.25),
    )

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_2.2_summary_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_prolonged_exposure_maps(
    rayman_dir: Path,
    thr_df: "pd.DataFrame",
    group_key: str,
    material: str,
    mask_dir: Path,
    out_dir: Path,
) -> None:
    """Side-by-side folium maps of prolonged exposure (>6h) by HW phase.

    Parameters
    ----------
    rayman_dir : Path
        Root RayMan output directory.
    thr_df : pd.DataFrame
        Output of ``load_thresholds``.
    group_key : str
        Key in ``POPULATION_GROUPS``.
    material : str
        Key in ``SCENARIOS``.
    mask_dir : Path
        Directory containing shapefiles.
    out_dir : Path
        Output directory for saved HTML.
    """
    import branca.colormap as bcm
    import folium
    from IPython.display import HTML, display

    from .configuration import HW_DAYS, PROLONGED_HOURS

    group_label = POPULATION_GROUPS[group_key]["label"]
    shp_aua = mask_dir / "aua_regional_units.shp"
    shp_elaionas = mask_dir / "elaionas_intervention_zone.shp"

    # Yellow-to-red progression: one color per day of prolonged exposure
    step_colors = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#800026"]
    maps_html = {}

    for hw in ["HW0", "HW1"]:
        n_days_total = HW_DAYS[hw]
        all_points = []

        for region_key, rinfo in REGIONS.items():
            region_label = rinfo["label"]
            # Per-day thresholds for this region/group
            daily_thr = lookup_daily_strong_heat_stress_thresholds(thr_df, region_label, group_label)
            hw_days_list = _days_in_hw(hw)
            daily_thr_hw = daily_thr[
                daily_thr.apply(
                    lambda r: (r["month"], r["day"]) in hw_days_list, axis=1,
                )
            ]

            # Load enriched base-case mPET
            base_df = load_enriched(
                rayman_dir, region_key, hw, group_key, material, base=True,
            )

            # Count prolonged-exposure days per grid point
            exposure = count_prolonged_exposure_days(
                base_df, daily_thr_hw, prolonged_hours=PROLONGED_HOURS,
            )
            exposure["region"] = region_label
            all_points.append(exposure)

        # Merge all three regions into one dataset for the map
        per_point = pd.concat(all_points, ignore_index=True)

        # Build folium map centered on the grid-point cloud
        center_lat = per_point["lat5"].mean()
        center_lon = per_point["lon5"].mean()
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=12,
            tiles="CartoDB positron",
        )

        # Boundary overlays
        add_shapefile_to_map(
            m, shp_aua, color="grey", weight=1.5,
            tooltip="Regional unit boundary",
        )
        add_shapefile_to_map(
            m, shp_elaionas, color="red", weight=2.5,
            tooltip="Elaionas intervention zone",
        )

        # Discrete colormap: 0 to total HW days
        n = n_days_total
        cmap_colors = step_colors[: n + 1]
        colormap = bcm.StepColormap(
            cmap_colors, index=list(range(n + 1)),
            vmin=0, vmax=n,
            caption=f"Days with > {PROLONGED_HOURS}h strong heat stress",
        )
        colormap.add_to(m)

        # Grid-point markers: color encodes days, grey = no prolonged exposure
        for _, row in per_point.iterrows():
            d = int(row["n_days"])
            folium.CircleMarker(
                location=[row["lat5"], row["lon5"]],
                radius=5,
                color="black" if d > 0 else "grey",
                weight=2 if d > 0 else 0.5,
                fill=True,
                fill_color=colormap(min(d, n)),
                fill_opacity=0.8,
                tooltip=f"{d}/{n} days",
            ).add_to(m)

        # Save each phase map as standalone HTML file
        hw_path = out_dir / f"fig_2.3_prolonged_exposure_{hw.lower()}.html"
        m.save(str(hw_path))

    # Display side-by-side via relative-path iframes (avoids JupyterLab trust issue)
    # Relative path from the notebook (notebooks/) to outputs (../outputs/)
    rel = "../outputs"
    from IPython.display import HTML, display
    display(HTML(f"""
<div style="text-align: center; font-weight: bold; font-size: 15px;
            margin-bottom: 6px;">
  Prolonged Heat-Stress Exposure (>6 h/day) at <u>Cooling Grid Points</u>: {group_label} | Base Case
</div>
<div style="display: flex; gap: 4px;">
  <div style="flex: 1; position: relative;">
    <iframe src="{rel}/fig_2.3_prolonged_exposure_hw0.html"
            style="width: 100%; height: 500px; border: none;"></iframe>
    <div style="position: absolute; bottom: 8px; left: 50%;
                transform: translateX(-50%); background: rgba(255,255,255,0.92);
                padding: 4px 12px; border-radius: 4px; font-weight: bold;
                font-size: 14px;">
      HW0 (Etesians) : {HW_DAYS['HW0']} days
    </div>
  </div>
  <div style="width: 2px; background: #333;"></div>
  <div style="flex: 1; position: relative;">
    <iframe src="{rel}/fig_2.3_prolonged_exposure_hw1.html"
            style="width: 100%; height: 500px; border: none;"></iframe>
    <div style="position: absolute; bottom: 8px; left: 50%;
                transform: translateX(-50%); background: rgba(255,255,255,0.92);
                padding: 4px 12px; border-radius: 4px; font-weight: bold;
                font-size: 14px;">
      HW1 (Sea breeze) : {HW_DAYS['HW1']} days
    </div>
  </div>
</div>
"""))


# [Health-benefits plots]

def plot_benefited_area(
    rayman_dir: Path,
    thr_df: "pd.DataFrame",
    group_key: str,
    out_dir: Path,
) -> None:
    """2x3 panel of benefited-area % by scenario and day.

    Parameters
    ----------
    rayman_dir : Path
        Root RayMan output directory.
    thr_df : pd.DataFrame
        Output of ``load_strong_heat_stress_thresholds``.
    group_key : str
        Key in ``POPULATION_GROUPS``.
    out_dir : Path
        Output directory for saved figure.
    """
    from .configuration import HW_DAYS

    group_label = POPULATION_GROUPS[group_key]["label"]
    hw_phases = ["HW0", "HW1"]
    region_order = list(REGIONS.keys())

    bar_colors = {k: v["color"] for k, v in SCENARIOS.items()}
    ylabel = "Benefited area (%)"

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    legend_added = False

    for i, hw in enumerate(hw_phases):
        hw_days_list = _days_in_hw(hw)

        for j, region_key in enumerate(region_order):
            ax = axes[i, j]
            rinfo = REGIONS[region_key]
            region_label = rinfo["label"]
            # Full-mask pixel count as denominator (F7 resolved)
            region_pixels = rinfo["pixels"]

            # Per-day thresholds for this region/group
            daily_thr = lookup_daily_strong_heat_stress_thresholds(
                thr_df, region_label, group_label,
            )
            daily_thr_hw = daily_thr[
                daily_thr.apply(
                    lambda r: (r["month"], r["day"]) in hw_days_list, axis=1,
                )
            ]

            all_days = sorted(hw_days_list)
            day_labels = [f"{d:02d}/{m:02d}" for m, d in all_days]

            for k, sc_key in enumerate(SCENARIOS):
                sc_label = SCENARIOS[sc_key]["label"]
                n_scn = len(SCENARIOS)
                width = 0.25
                offset = (k - (n_scn - 1) / 2.0) * width

                # Load base and scenario enriched CSVs
                base_df = load_enriched(
                    rayman_dir, region_key, hw, group_key, sc_key, base=True,
                )
                scen_df = load_enriched(
                    rayman_dir, region_key, hw, group_key, sc_key, base=False,
                )

                # Identify benefited points using per-day thresholds
                result = identify_benefited_points(
                    base_df, scen_df, daily_thr_hw,
                )

                # Compute % per day against full regional mask
                pct_by_day = []
                for md_tuple in all_days:
                    m, d = md_tuple
                    day_result = result[
                        (result["month"] == m) & (result["day"] == d)
                    ]
                    n_qualifying = day_result["benefited"].sum()
                    pct_by_day.append(100.0 * n_qualifying / region_pixels)

                x_pos = np.arange(len(all_days)) + offset
                ax.bar(
                    x_pos, pct_by_day, width, color=bar_colors[sc_key],
                    alpha=0.9, label=sc_label if not legend_added else None,
                )

            legend_added = True

            ax.set_ylim(0, 20.0)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_xticks(np.arange(len(all_days)))
            ax.set_xticklabels(day_labels, fontsize=10)
            if i == 0:
                ax.set_title(region_label, fontsize=13)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if i == 1:
                ax.set_xlabel("Day", fontsize=11)
            # HW phase label on each panel
            ax.text(
                0.01, 0.98, hw, transform=ax.transAxes,
                ha="left", va="top", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=2.0),
            )

    fig.suptitle(
        f"{ylabel}: {group_label}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.legend(
        loc="upper center", ncol=len(SCENARIOS), frameon=False,
        bbox_to_anchor=(0.5, 0.995), fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_dir / "fig_3.1_benefited_area.png"),
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_benefited_population(
    rayman_dir: Path,
    thr_df: "pd.DataFrame",
    group_key: str,
    pop_dir: Path,
    out_dir: Path,
) -> None:
    """1x2 grouped bar chart of mean daily benefited population (Table 2 analog).

    Parameters
    ----------
    rayman_dir : Path
        Root RayMan output directory.
    thr_df : pd.DataFrame
        Output of ``load_strong_heat_stress_thresholds``.
    group_key : str
        Key in ``POPULATION_GROUPS``.
    pop_dir : Path
        Directory containing population GeoPackages and totals CSV.
    out_dir : Path
        Output directory for saved figure.
    """
    from .configuration import HW_DAYS

    group_label = POPULATION_GROUPS[group_key]["label"]
    pop_col = POPULATION_GROUPS[group_key]["pop_col"]
    gpkg_path = pop_dir / POPULATION_GROUPS[group_key]["gpkg"]
    totals_df = pd.read_csv(pop_dir / "AUA_Regional_Units_PopTotals_2021.csv")

    hw_phases = ["HW0", "HW1"]
    region_order = list(REGIONS.keys())
    ru_keep = [REGIONS[r]["ru_int"] for r in region_order]

    # Load population weights for this group
    pop_map = read_gpkg_pop_weights(gpkg_path, ru_keep)
    pop_totals = {
        rk: regional_pop_total(totals_df, REGIONS[rk]["ru_str"], pop_col)
        for rk in region_order
    }

    # Collect summary data across all HW/region/scenario combinations
    summary_rows = []
    for hw in hw_phases:
        hw_days_list = _days_in_hw(hw)
        n_hw_days = HW_DAYS[hw]

        for region_key in region_order:
            rinfo = REGIONS[region_key]
            region_label = rinfo["label"]
            daily_thr = lookup_daily_strong_heat_stress_thresholds(
                thr_df, region_label, group_label,
            )
            daily_thr_hw = daily_thr[
                daily_thr.apply(
                    lambda r: (r["month"], r["day"]) in hw_days_list, axis=1,
                )
            ]
            ru_pop = pop_map[pop_map["RU_CODE"] == rinfo["ru_int"]]
            total_pop = pop_totals[region_key]

            for sc_key in SCENARIOS:
                base_df = load_enriched(
                    rayman_dir, region_key, hw, group_key, sc_key, base=True,
                )
                scen_df = load_enriched(
                    rayman_dir, region_key, hw, group_key, sc_key, base=False,
                )
                benefited = identify_benefited_points(
                    base_df, scen_df, daily_thr_hw,
                )
                benefited_only = benefited[benefited["benefited"]].copy()

                # Convert to population counts
                pop_df = weight_to_population(benefited_only, ru_pop, total_pop)
                sum_pop = pop_df["pop_count"].sum()
                mean_pop = sum_pop / n_hw_days
                pct = mean_pop / total_pop * 100 if total_pop > 0 else 0

                summary_rows.append({
                    "hw": hw, "region": region_key,
                    "region_label": region_label,
                    "scenario": sc_key,
                    "scenario_label": SCENARIOS[sc_key]["label"],
                    "mean_pop": mean_pop, "pct": pct,
                })

    summary = pd.DataFrame(summary_rows)

    # 1x2 bar chart (HW0, HW1)
    fig, axes_pop = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    region_labels_list = [REGIONS[r]["label"] for r in region_order]

    for idx, hw in enumerate(hw_phases):
        ax = axes_pop[idx]
        hw_data = summary[summary["hw"] == hw]
        x = np.arange(len(region_order))
        width = 0.22
        n_scn = len(SCENARIOS)

        for k, sc_key in enumerate(SCENARIOS):
            sc_label = SCENARIOS[sc_key]["label"]
            offset = (k - (n_scn - 1) / 2.0) * width
            sc_data = hw_data[hw_data["scenario"] == sc_key]

            vals = [sc_data[sc_data["region"] == r]["mean_pop"].values[0]
                    for r in region_order]
            pcts = [sc_data[sc_data["region"] == r]["pct"].values[0]
                    for r in region_order]

            bar_set = ax.bar(
                x + offset, vals, width,
                color=SCENARIOS[sc_key]["color"], alpha=0.9,
                label=sc_label if idx == 0 else None,
            )

            # Value annotations: count and percentage
            for bar, v, p in zip(bar_set, vals, pcts):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, v + 200,
                        f"{v:,.0f}\n({p:.1f}%)",
                        ha="center", va="bottom", fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(region_labels_list, fontsize=11)
        hw_title = {"HW0": "HW0 (Etesians)", "HW1": "HW1 (Sea breeze)"}
        ax.set_title(hw_title[hw], fontsize=13, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Mean daily benefited population", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Fixed y-axis for consistent cross-group comparison
    for ax in axes_pop:
        ax.set_ylim(0, 35000)

    fig.suptitle(
        f"Mean Daily Benefited Population: {group_label}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.legend(
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    fig.savefig(str(out_dir / "fig_3.2_benefited_population.png"),
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_benefited_population_maps(
    rayman_dir: Path,
    thr_df: "pd.DataFrame",
    group_key: str,
    material: str,
    pop_dir: Path,
    mask_dir: Path,
    out_dir: Path,
) -> None:
    """Side-by-side folium maps with benefited cells sized by population.

    Parameters
    ----------
    rayman_dir : Path
        Root RayMan output directory.
    thr_df : pd.DataFrame
        Output of ``load_strong_heat_stress_thresholds``.
    group_key : str
        Key in ``POPULATION_GROUPS``.
    material : str
        Key in ``SCENARIOS``.
    pop_dir : Path
        Directory containing population GeoPackages and totals CSV.
    mask_dir : Path
        Directory containing shapefiles.
    out_dir : Path
        Output directory for saved HTML.
    """
    import folium
    from IPython.display import HTML, display

    from .configuration import HW_DAYS

    group_label = POPULATION_GROUPS[group_key]["label"]
    pop_col = POPULATION_GROUPS[group_key]["pop_col"]
    gpkg_path = pop_dir / POPULATION_GROUPS[group_key]["gpkg"]
    totals_df = pd.read_csv(pop_dir / "AUA_Regional_Units_PopTotals_2021.csv")

    shp_aua = mask_dir / "aua_regional_units.shp"
    shp_elaionas = mask_dir / "elaionas_intervention_zone.shp"

    region_order = list(REGIONS.keys())
    ru_keep = [REGIONS[r]["ru_int"] for r in region_order]
    pop_map = read_gpkg_pop_weights(gpkg_path, ru_keep)

    for hw in ["HW0", "HW1"]:
        hw_days_list = _days_in_hw(hw)
        all_benefited = []
        all_base_points = []

        for region_key, rinfo in REGIONS.items():
            region_label = rinfo["label"]
            daily_thr = lookup_daily_strong_heat_stress_thresholds(
                thr_df, region_label, group_label,
            )
            daily_thr_hw = daily_thr[
                daily_thr.apply(
                    lambda r: (r["month"], r["day"]) in hw_days_list, axis=1,
                )
            ]
            ru_pop = pop_map[pop_map["RU_CODE"] == rinfo["ru_int"]]
            total_pop = regional_pop_total(
                totals_df, rinfo["ru_str"], pop_col,
            )

            base_df = load_enriched(
                rayman_dir, region_key, hw, group_key, material, base=True,
            )
            scen_df = load_enriched(
                rayman_dir, region_key, hw, group_key, material, base=False,
            )

            # All base cooling grid points for grey background
            base_pts = base_df[["lon5", "lat5"]].drop_duplicates()
            base_pts["region"] = region_label
            all_base_points.append(base_pts)

            # Benefited points with population
            result = identify_benefited_points(
                base_df, scen_df, daily_thr_hw,
            )
            benefited_only = result[result["benefited"]].copy()
            pop_df = weight_to_population(benefited_only, ru_pop, total_pop)

            # Mean population per grid point across days
            pt_pop = pop_df.groupby(["lon5", "lat5"], as_index=False).agg(
                pop_count=("pop_count", "mean"),
            )
            pt_pop["region"] = region_label
            all_benefited.append(pt_pop)

        benefited_pts = pd.concat(all_benefited, ignore_index=True)
        base_all = pd.concat(all_base_points, ignore_index=True)
        max_pop = benefited_pts["pop_count"].max() if len(benefited_pts) > 0 else 1

        # Build folium map centered on the grid-point cloud
        center_lat = base_all["lat5"].mean()
        center_lon = base_all["lon5"].mean()
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=12,
            tiles="CartoDB positron",
        )

        add_shapefile_to_map(
            m, shp_aua, color="grey", weight=1.5,
            tooltip="Regional unit boundary",
        )
        add_shapefile_to_map(
            m, shp_elaionas, color="red", weight=2.5,
            tooltip="Elaionas intervention zone",
        )

        # Grey background: non-benefited cooling grid points
        benefited_coords = set(
            zip(benefited_pts["lon5"], benefited_pts["lat5"])
        )
        for _, row in base_all.iterrows():
            if (row["lon5"], row["lat5"]) not in benefited_coords:
                folium.CircleMarker(
                    [row["lat5"], row["lon5"]], radius=2,
                    color="grey", weight=0.3, fill=True,
                    fill_color="grey", fill_opacity=0.4,
                ).add_to(m)

        # Benefited points: circle radius scaled by population count
        for _, row in benefited_pts.iterrows():
            pop = row["pop_count"]
            radius = _scale_marker_radius(pop, max_pop, rmin=3, rmax=15)
            folium.CircleMarker(
                [row["lat5"], row["lon5"]],
                radius=radius,
                color="black", weight=1,
                fill=True, fill_color="#1f78b4", fill_opacity=0.7,
                tooltip=f"{pop:,.0f} people/day",
            ).add_to(m)

        # Save as standalone HTML
        hw_path = out_dir / f"fig_3.2_benefited_pop_{hw.lower()}.html"
        m.save(str(hw_path))

    # Side-by-side display via relative-path iframes
    rel = "../outputs"
    display(HTML(f"""
<div style="text-align: center; font-weight: bold; font-size: 15px;
            margin-bottom: 6px;">
  Mean Daily Benefited Population at <u>Cooling Grid Points</u>: {group_label} | {SCENARIOS[material]["label"]} | Circle size ~ population
</div>
<div style="display: flex; gap: 4px;">
  <div style="flex: 1; position: relative;">
    <iframe src="{rel}/fig_3.2_benefited_pop_hw0.html"
            style="width: 100%; height: 500px; border: none;"></iframe>
    <div style="position: absolute; bottom: 8px; left: 50%;
                transform: translateX(-50%); background: rgba(255,255,255,0.92);
                padding: 4px 12px; border-radius: 4px; font-weight: bold;
                font-size: 14px;">
      HW0 (Etesians) : {HW_DAYS['HW0']} days
    </div>
  </div>
  <div style="width: 2px; background: #333;"></div>
  <div style="flex: 1; position: relative;">
    <iframe src="{rel}/fig_3.2_benefited_pop_hw1.html"
            style="width: 100%; height: 500px; border: none;"></iframe>
    <div style="position: absolute; bottom: 8px; left: 50%;
                transform: translateX(-50%); background: rgba(255,255,255,0.92);
                padding: 4px 12px; border-radius: 4px; font-weight: bold;
                font-size: 14px;">
      HW1 (Sea breeze) : {HW_DAYS['HW1']} days
    </div>
  </div>
</div>
"""))
