"""CoolScape workshop utilities.

Re-exports the public API from context-grouped sub-modules:

    configuration.py  — config dicts, shared helpers
    mpet_data.py      — mPET readers, threshold lookup
    wrf_fields.py     — WRF data readers, masking, phase spans  (Section 1)
    population.py     — population weights, benefited-point logic (Section 3)
    visualization.py  — colormaps, shapefile overlays             (cross-cutting)

Modules are added incrementally as sections are built.
Import pattern:  ``import utils as cs``  →  ``cs.SCENARIOS``, ``cs.load_thresholds()``.
"""

# --- configuration.py — config dicts + shared helpers ---
from .configuration import (
    SCENARIOS,
    REGIONS,
    REGION_COLORS,
    POPULATION_GROUPS,
    MPET_BINS,
    MPET_CLASSES,
    MPET_COLORS,
    HW_PHASES,
    HW_DAYS,
    HEATMAP_COLUMNS,
    DAYTIME_WINDOW,
    NIGHT_WINDOW,
    ROUND_LL,
    PROLONGED_HOURS,
    MAP_EXTENT,
    _add_coord_keys,
    _days_in_hw,
)

# --- mpet_data.py — mPET readers, thresholds, analysis ---
from .mpet_data import (
    load_strong_heat_stress_thresholds,
    lookup_daily_strong_heat_stress_thresholds,
    load_mpet_group,
    load_enriched,
    hourly_category_frequencies,
    count_prolonged_exposure_days,
)

# --- population.py — population readers + benefited-point analysis ---
from .population import (
    read_gpkg_pop_weights,
    regional_pop_total,
    identify_benefited_points,
    weight_to_population,
)

# --- wrf_fields.py — WRF readers + phase spans ---
from .wrf_fields import (
    load_base_case_area_means,
    phase_spans,
)

# --- visualization.py — colormaps, shapefile overlays, plot builders ---
from .visualization import (
    draw_shapefile_overlay,
    add_shapefile_to_map,
    make_diverging_white_cmap,
    make_sequential_cooling_cmap,
    plot_base_t2_timeseries,
    plot_diff_panel_maps,
    plot_cooling_heatmap,
    plot_mpet_single_panel,
    interactive_threshold_toggle,
    plot_mpet_bioclimate_diagram,
    plot_acclimatization_summary,
    plot_prolonged_exposure_maps,
    plot_benefited_area,
    plot_benefited_population,
    plot_benefited_population_maps,
)

__all__ = [
    # configuration.py
    "SCENARIOS",
    "REGIONS",
    "REGION_COLORS",
    "POPULATION_GROUPS",
    "MPET_BINS",
    "MPET_CLASSES",
    "MPET_COLORS",
    "HW_PHASES",
    "HW_DAYS",
    "HEATMAP_COLUMNS",
    "DAYTIME_WINDOW",
    "NIGHT_WINDOW",
    "ROUND_LL",
    "PROLONGED_HOURS",
    "MAP_EXTENT",
    "_add_coord_keys",
    "_days_in_hw",
    # mpet_data.py
    "load_strong_heat_stress_thresholds",
    "lookup_daily_strong_heat_stress_thresholds",
    "load_mpet_group",
    "load_enriched",
    "hourly_category_frequencies",
    "count_prolonged_exposure_days",
    # population.py
    "read_gpkg_pop_weights",
    "regional_pop_total",
    "identify_benefited_points",
    "weight_to_population",
    # wrf_fields.py
    "load_base_case_area_means",
    "phase_spans",
    # visualization.py
    "draw_shapefile_overlay",
    "add_shapefile_to_map",
    "make_diverging_white_cmap",
    "make_sequential_cooling_cmap",
    "plot_base_t2_timeseries",
    "plot_diff_panel_maps",
    "plot_cooling_heatmap",
    "plot_mpet_single_panel",
    "interactive_threshold_toggle",
    "plot_mpet_bioclimate_diagram",
    "plot_acclimatization_summary",
    "plot_prolonged_exposure_maps",
    "plot_benefited_area",
    "plot_benefited_population",
    "plot_benefited_population_maps",
]
