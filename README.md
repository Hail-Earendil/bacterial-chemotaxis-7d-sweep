# bacterial-chemotaxis-7d-sweep

Code and reproducible results pipeline for the paper:

**“Apparent selection pressure for channel capacity and dynamic range in bacterial chemotactic sensors.”**

This repository contains a Python script that runs a **7-dimensional parameter sweep** of an Monod–Wyman–Changeux receptor model and exports the numerical outputs and visualizations used in the manuscript and supplementary information.

---

## Paper and data
- **arXiv:** [https://arxiv.org/abs/2601.02446](https://arxiv.org/abs/2601.02446)
- **Dryad dataset DOI (contains NPZ sweep output):** [https://doi.org/10.5061/dryad.wpzgmsc3j](https://doi.org/10.5061/dryad.wpzgmsc3j)
  
---

## What’s in this repository

- **`7D_Sweep_Code.py`** — main script that:
  - runs (or resumes) a 7D sweep
  - saves results to a single `.npz` file
  - exports tables, logs, and figures after the sweep completes

---

## What the script produces

Running the script creates an output directory:

`outputs/<RUN_TAG>/`

- `<RUN_TAG>` defaults to today’s date in **`YYYYMMDD`** format

A completed run typically contains:

- **`7D_Sweep_Results.npz`** — the main 7D sweep results (NumPy archive)
- **`logs/`** — warning summaries and event logs (helps diagnose edge cases)
- **`tables/`** — exported CSV tables and Blahut–Arimoto convergence summaries
- **`plots/`** — **2D** visualizations of the sweep (heatmaps)
- **`curves/`** — **1D** visualizations of the sweep (line plots)

Note: If a sweep is incomplete, the script saves a partial .npz file and exits cleanly. Post-processing runs only once the sweep is complete.

---

## Requirements

- **Python 3.10+** recommended
- **NumPy, SciPy, Matplotlib**

If you don’t already have the dependencies:

```bash
python3 -m pip install numpy scipy matplotlib
```
---

## Quick start

### Option A — Run a new sweep (or resume an existing one)

From the repository root:

`python3 7D_Sweep_Code.py`

Outputs will appear in:

`outputs/<YYYYMMDD>/`

The sweep is **resumable**: if `outputs/<RUN_TAG>/7D_Sweep_Results.npz` already exists, the script will continue from where it left off.

### Option B — Regenerate tables/figures from an existing npz file

If you already have a sweep file (e.g. downloaded from Dyrad), you can regenerate derived outputs without recomputing the sweep.

1) Download the NPZ and place it here:

`outputs/<RUN_TAG>/7D_Sweep_Results.npz`

2) Set the run tag so the script reads from that folder, then run the script:

```bash
export SWEEP_RUN_TAG=<RUN_TAG>
python3 7D_Sweep_Code.py
```

Important: the script internally constructs the sweep grids. If the downloaded NPZ was generated using a different grid configuration than the one currently hardcoded in `7D_Sweep_Code.py`, the script may raise a *grid mismatch* error. To avoid this, keep the grid settings consistent with the dataset version you downloaded.

---

## Funding / rights notice

This data was produced by University of Maryland, College Park under Army Research Office (ARO) Award Number W911NF-25-1-0260. ARO, as the Federal awarding agency, reserves a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this data for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
