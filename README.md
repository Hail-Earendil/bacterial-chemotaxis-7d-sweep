# bacterial-chemotaxis-7d-sweep

Code and reproducible-results pipeline for the paper:

**"An Investigation of the Channel Capacity of Bacterial Chemotactic Sensors for Low Chemoattractant Concentrations."**

This repository contains a single Python script that runs a **7-dimensional parameter sweep** of a Monod–Wyman–Changeux receptor model and regenerates every main-text figure and table used in the manuscript and supplementary information.

---

## Paper and data
- **arXiv:** [https://arxiv.org/abs/2601.02446](https://arxiv.org/abs/2601.02446)
- **Dryad dataset DOI (contains NPZ sweep output):** [https://doi.org/10.5061/dryad.wpzgmsc3j](https://doi.org/10.5061/dryad.wpzgmsc3j)
  - *Note: this Dryad entry is being updated to include both NPZ files (`7D_Sweep_Results.npz` and `7D_Sweep_Results_keymer.npz`) used in the revised paper. Updated dataset coming soon.*

---

## What's in this repository

- **`bacterial_chemotaxis.py`** — single-file library and CLI that:
  - runs (or resumes) the 7D sweep
  - saves results to `.npz` files
  - regenerates every main-text figure and stdout table from those NPZ files

---

## Requirements

- **Python 3.10+** recommended
- **NumPy, SciPy, Matplotlib**

```bash
python3 -m pip install numpy scipy matplotlib
```

---

## Quick start

### Option A — Regenerate figures and tables from an existing NPZ file

If you already have both sweep files (e.g. downloaded from Dryad) placed next to the script as `7D_Sweep_Results.npz` and `7D_Sweep_Results_keymer.npz`, you can regenerate every main-text figure and table in about a minute:

```bash
python3 bacterial_chemotaxis.py --all
```

Or run specific outputs:

```bash
python3 bacterial_chemotaxis.py --fig3-4        # Figs 3 and 4 (data figures)
python3 bacterial_chemotaxis.py --fig5          # Fig 5 (C vs p_0 with ceiling curve)
python3 bacterial_chemotaxis.py --strain-table  # per-strain C, DR, |n_eff|, gradient norms
python3 bacterial_chemotaxis.py --correlation   # Table II: correlation matrix + global maxima
python3 bacterial_chemotaxis.py --gradient-max  # Table IV: gradient-norm ratios
python3 bacterial_chemotaxis.py --si-heatmaps   # 30 SI heatmap PDFs (Figs S1–S60)
```

Outputs: `./sweep_figures/` (figures), `./si_heatmaps/` (SI heatmaps). Tables print to stdout. Use `--output-dir DIR` or `--overleaf-dest DIR` to redirect.

### Option B — Run a new sweep from scratch (or resume an interrupted one)

The sweep itself is slow (multi-hour on a single core), resumable (checkpoint every 30 minutes), and only needed if you're reproducing the NPZ data files rather than the figures.

```bash
python3 bacterial_chemotaxis.py --run-mello-sweep    # Mello/Tu K_d region (~12h)
python3 bacterial_chemotaxis.py --run-keymer-sweep   # Wingreen K_d region (~11h)
```

If `7D_Sweep_Results.npz` or `7D_Sweep_Results_keymer.npz` already exists in the working directory, the sweep resumes from where it left off. If a sweep is incomplete when the time budget expires, the script saves a partial NPZ and exits cleanly.

Run `python3 bacterial_chemotaxis.py --help` for full CLI options.

---

## Funding / rights notice

This data was produced by University of Maryland, College Park under Army Research Office (ARO) Award Number W911NF-25-1-0260. ARO, as the Federal awarding agency, reserves a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this data for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
