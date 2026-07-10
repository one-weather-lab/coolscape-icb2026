# ICB2026 Workshop Notebook

This repository contains the notebook for the workshop "A health-relevant computational framework for just resilience: Linking targeted urban cooling scenarios to population-wide benefits," held at the 24th International Congress of Biometeorology in Novi Sad, Serbia, on 16 July 2026. It reproduces an assessment of targeted cool-roof interventions in the Athens Urban Area, Greece, during the heat wave of 28 July to 5 August 2021. 

The workflow links atmospheric cooling to population-specific heat-stress response and health-related benefits.

## Notebook

| # | Title | Core topics |
|---|---|---|
| 01 | `coolscape_icb2026` | Atmospheric cooling under contrasting wind patterns, modified Physiologically Equivalent Temperature (mPET), acclimatization-based heat-stress thresholds, prolonged exposure, benefited area and population, and just resilience interpretation |

## Running the Notebook

### Option A: Binder

**Run the workshop notebook via Binder:**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/one-weather-lab/coolscape-icb2026/HEAD?labpath=notebooks%2Fcoolscape_icb2026.ipynb)

Open the notebook and run the cells sequentially. The environment and required workshop data are included in the repository.

### Option B: Local Machine

Clone the repository:

```bash
git clone https://github.com/one-weather-lab/coolscape-icb2026.git
cd coolscape-icb2026
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate coolscape-icb2026
```

Launch JupyterLab:

```bash
jupyter lab notebooks/
```

For pip-based installation, use `pip install -r requirements.txt`.

## Data and Attribution

The repository uses a curated subset of the dataset supporting the reference study on which this workshop is based.

- **Meteorological data:** Near-surface WRF v4.3.2 + BEP/BEM output over the Athens Urban Area at 400 m spatial and 1 h temporal resolution.
- **Human-biometeorological data:** mPET time series derived via the RayMan Pro model for four population groups at grid points with air-temperature reductions relative to the base case.
- **Population data:** WorldPop sex- and age-disaggregated population estimates aggregated to the 400 m model grid, with corresponding regional-unit totals.
- **Spatial data:** Boundaries for the Athens Urban Area regional units, Attica, and the Elaionas intervention zone.

The original dataset is available under the Creative Commons Attribution 4.0 International license:

Giannaros C, Kotroni V and Lagouvardos K (2026), *Scenario-based modeling data for exploring urban cooling pathways: A targeted, health-relevant approach toward just resilience in Athens Urban Area, Greece*, Zenodo. https://doi.org/10.5281/zenodo.18014875

WorldPop data are available under the Creative Commons Attribution 4.0 International license:

Bondarenko M et al. and WorldPop (2025), *Estimates of 2015-2030 total number of people per grid square broken down by gender and age groupings, R2025A version v1*. https://doi.org/10.5258/SOTON/WP00847

## Study Reference

Giannaros C, Kotroni V and Lagouvardos K (2026), "Scenario-based modeling for exploring urban cooling pathways: A targeted, health-relevant approach toward just resilience," *Environmental Research: Health*, **4**, 021002. https://doi.org/10.1088/2752-5309/ae6095

## Acknowledgment

This work was conducted in collaboration with the National Observatory of Athens and funded by the European Union's Horizon Europe Climate-Adapt4EOSC project under Grant Agreement No. 101188248.
