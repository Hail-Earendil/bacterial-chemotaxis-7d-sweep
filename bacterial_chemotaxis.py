"""
bacterial_chemotaxis.py

Consolidated reproduction script for Cui & Marzen (2026), submitted to
Physical Review E. Combines the monolithic 7D_Sweep_Code.py library with
the six paper-facing scripts (compute_pdf_metrics, compute_max_gradient,
recompute_table_R_and_maxima, regenerate_main_figures, regenerate_new_figs,
generate_new_strain_heatmaps, run_overnight_keymer_sweep) into a single
file with a --command CLI.

Usage
-----
    python bacterial_chemotaxis.py --all
        Regenerate every main-text figure (3, 4, 5) and print every
        stdout table (correlation matrix, per-strain metrics, gradient
        norms). Assumes both NPZ files are already on disk in the repo
        root; runs in ~1 minute.

    python bacterial_chemotaxis.py --fig3-4
    python bacterial_chemotaxis.py --fig5
    python bacterial_chemotaxis.py --strain-table
    python bacterial_chemotaxis.py --correlation
    python bacterial_chemotaxis.py --gradient-max
    python bacterial_chemotaxis.py --si-heatmaps
        Individual reproduction targets.

    python bacterial_chemotaxis.py --run-mello-sweep
    python bacterial_chemotaxis.py --run-keymer-sweep
        Reproduce the 7D sweep NPZ files from scratch. SLOW: each takes
        multiple hours on a single core. Both are resumable (checkpoint
        every 30 minutes) and can be interrupted with Ctrl-C. Only run
        these if you actually want to reproduce the sweep data.

Outputs land next to this script by default:
    - Figures:  ./sweep_figures/
    - Stdout tables: printed to the console
    - SI heatmaps: ./si_heatmaps/  (or --overleaf-dest DIR)

Pass --output-dir DIR to redirect figure output.

This file is a mechanical consolidation of code that produced the
submitted paper. No numerics were changed. See the section headers
below for the mapping from the original files.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from math import log
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from scipy.optimize import root_scalar


# ============================================================================
# PATHS
# ============================================================================
REPO_DIR = Path(__file__).resolve().parent
NPZ_MELLO_DEFAULT  = REPO_DIR / "7D_Sweep_Results.npz"
NPZ_KEYMER_DEFAULT = REPO_DIR / "7D_Sweep_Results_keymer.npz"


# ============================================================================
# WARNING SYSTEM  (verbatim from 7D_Sweep_Code.py lines 26-167)
# ----------------------------------------------------------------------------
# Records solver warnings (bracketing failures, flat-curve degeneracies)
# with parameter-set context. Used internally by the MWC solvers so they
# don't spam stdout during a 7D sweep.
# ============================================================================
FLAT_DELTA_P_THRESH = 1e-32

ERROR_PRINT_LIMIT = 50
_ERR_COUNT = {"flat": 0, "bracket": 0, "interp": 0, "ba": 0}

LOG_MAX_EVENTS_TOTAL = 200_000
LOG_MAX_UNIQUE_PARAMSETS_PER_KEY = 50_000
LOG_TOPK_PARAMSETS_PER_KEY = 50
LOG_STORE_EVENTS = True
_WARN_CONTEXT = None

_WARN_STATS = {}
_WARN_EVENTS = []
_WARN_DROPPED_EVENTS = 0


def _format_params_for_key(params: dict) -> str:
    if not params:
        return "NO_CONTEXT"
    def f(x: float) -> str:
        return f"{float(x):.6g}"
    return (
        f"L0={f(params['L0'])},"
        f"KdI1={f(params['KdI1'])},KdA1={f(params['KdA1'])},"
        f"KdI2={f(params['KdI2'])},KdA2={f(params['KdA2'])},"
        f"N_tar={f(params['N_tar'])},N_tsr={f(params['N_tsr'])}"
    )


def set_warn_context(params: dict | None) -> None:
    global _WARN_CONTEXT
    _WARN_CONTEXT = params


class WarnContext:
    def __init__(self, params: dict | None):
        self.params = params
        self._prev = None

    def __enter__(self):
        global _WARN_CONTEXT
        self._prev = _WARN_CONTEXT
        _WARN_CONTEXT = self.params

    def __exit__(self, exc_type, exc, tb):
        global _WARN_CONTEXT
        _WARN_CONTEXT = self._prev
        return False


def _ensure_warn_key_struct(key: str) -> None:
    if key not in _WARN_STATS:
        _WARN_STATS[key] = {
            "count_total": 0,
            "param_counts": {},
            "sample_messages": [],
            "dropped_paramsets": 0
        }


def _warn_once(key: str, msg: str) -> None:
    global _WARN_DROPPED_EVENTS
    cnt = _ERR_COUNT.get(key, 0)

    if cnt < ERROR_PRINT_LIMIT:
        print(f"[WARN:{key}] {msg}")
    _ERR_COUNT[key] = cnt + 1
    _ensure_warn_key_struct(key)
    _WARN_STATS[key]["count_total"] += 1

    if len(_WARN_STATS[key]["sample_messages"]) < 10:
        _WARN_STATS[key]["sample_messages"].append(str(msg))
    params = _WARN_CONTEXT
    pkey = _format_params_for_key(params) if params else "NO_CONTEXT"
    pc = _WARN_STATS[key]["param_counts"]

    if (pkey in pc) or (len(pc) < LOG_MAX_UNIQUE_PARAMSETS_PER_KEY):
        pc[pkey] = int(pc.get(pkey, 0)) + 1
    else:
        _WARN_STATS[key]["dropped_paramsets"] += 1

    if LOG_STORE_EVENTS:
        if len(_WARN_EVENTS) < LOG_MAX_EVENTS_TOTAL:
            _WARN_EVENTS.append({
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "key": key,
                "msg": str(msg),
                "params": params if params else None,
            })
        else:
            _WARN_DROPPED_EVENTS += 1


def write_warning_logs(outdir: str | Path, *, tag: str = "run") -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "tag": str(tag),
        "utc_written": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "console_print_limit_per_key": ERROR_PRINT_LIMIT,
        "stored_event_lines": int(len(_WARN_EVENTS)),
        "dropped_event_lines": int(_WARN_DROPPED_EVENTS),
        "per_key": {}
    }

    for key, info in _WARN_STATS.items():
        pc = info["param_counts"]
        top_items = sorted(pc.items(), key=lambda kv: kv[1], reverse=True)[:LOG_TOPK_PARAMSETS_PER_KEY]
        top_paramsets = [{"params_key": k, "count": int(v)} for k, v in top_items]

        summary["per_key"][key] = {
            "count_total": int(info["count_total"]),
            "unique_paramsets_tracked": int(len(pc)),
            "dropped_paramsets": int(info.get("dropped_paramsets", 0)),
            "sample_messages": list(info.get("sample_messages", [])),
            "top_paramsets": top_paramsets,
        }

    summary_path = outdir / "warnings_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if LOG_STORE_EVENTS:
        events_path = outdir / "warnings_events.ndjson"
        with open(events_path, "w") as f:
            for ev in _WARN_EVENTS:
                f.write(json.dumps(ev) + "\n")
    else:
        events_path = None

    print(f"[LOG] wrote {summary_path.resolve()}")
    if events_path:
        print(f"[LOG] wrote {events_path.resolve()}")

    return summary


# ============================================================================
# CORE MWC MODEL  (verbatim from 7D_Sweep_Code.py lines 169-286)
# ============================================================================
def p_active(c: np.ndarray, logL0: float, KdI1: float, KdA1: float, KdI2: float, KdA2: float, N_tar: float, N_tsr: float) -> np.ndarray:
    if KdI1 <= 0 or KdA1 <= 0 or KdI2 <= 0 or KdA2 <= 0:
        raise ValueError("Kd values must be > 0.")

    c = np.asarray(c, dtype=np.float64)

    d_tar = N_tar * (np.log1p(c / KdI1) - np.log1p(c / KdA1))
    d_tsr = N_tsr * (np.log1p(c / KdI2) - np.log1p(c / KdA2))
    Delta = logL0 + d_tar + d_tsr

    out = np.exp(-np.logaddexp(0.0, Delta))

    return np.clip(out, 1e-12, 1.0 - 1e-12)


def p_active_scalar(c: float, logL0: float, KdI1: float, KdA1: float, KdI2: float, KdA2: float, N_tar: float, N_tsr: float) -> float:
    return float(p_active(np.array([c], dtype=float), logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr)[0])


def endpoints_p0_pinf(logL0: float, KdI1: float, KdA1: float, KdI2: float, KdA2: float, N_tar: float, N_tsr: float) -> tuple[float, float]:
    p0 = 1.0 / (1.0 + np.exp(+logL0))

    log_factor = 0.0
    if N_tar > 0:
        log_factor += N_tar * np.log(KdA1 / KdI1)
    if N_tsr > 0:
        log_factor += N_tsr * np.log(KdA2 / KdI2)

    pinf = 1.0 / (1.0 + np.exp(logL0 + log_factor))

    return float(p0), float(pinf)


def _p_minus_target_logc(logc, target_p, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr):
    c = float(np.exp(logc))
    return p_active_scalar(c, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr) - target_p


def _p_minus_target_c(c, target_p, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr):
    return p_active_scalar(
        float(c), logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr) - target_p


def solve_c_at_p(target_p: float, logL0: float, KdI1: float, KdA1: float, KdI2: float, KdA2: float, N_tar: float, N_tsr: float) -> float | np.nan:
    t = float(np.clip(target_p, 1e-12, 1.0 - 1e-12))
    kds = [
        x for x in (
            KdI1, KdA1,
            (KdI2 if N_tsr > 0 else None),
            (KdA2 if N_tsr > 0 else None)
        )
        if x and x > 0
    ]
    if not kds:
        kds = [1.0]
    kmin, kmax = float(min(kds)), float(max(kds))
    cL = max(kmin * 1e-9, 1e-18)
    cR = min(kmax * 1e+9, 1e+18)

    pL = p_active_scalar(cL, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr)
    pR = p_active_scalar(cR, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr)

    lo, hi = (log(cL), log(cR))
    f_lo = pL - t
    f_hi = pR - t

    if not ((f_lo == 0.0) or (f_hi == 0.0) or
            (f_lo < 0 and f_hi > 0) or (f_lo > 0 and f_hi < 0)):
        lo2, hi2 = log(1e-300), log(1e300)
        f_lo2 = _p_minus_target_logc(lo2, t, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr)
        f_hi2 = _p_minus_target_logc(hi2, t, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr)
        if ((f_lo2 == 0.0) or (f_hi2 == 0.0) or (f_lo2 < 0 and f_hi2 > 0) or (f_lo2 > 0 and f_hi2 < 0)):
            lo, hi = lo2, hi2
        else:
            try:
                sol = root_scalar(_p_minus_target_c, bracket=(cL, cR), method="bisect", args=(t, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr), xtol=1e-12, rtol=1e-10, maxiter=200)
                if sol.converged:
                    _warn_once("bracket", f"Used linear-c fallback bracketing for target p={t:.3g}.")
                    return float(sol.root)
            except Exception:
                _warn_once("bracket", f"Unbracketable root at p={t:.3g}; returning NaN.")
                return float('nan')

    try:
        sol = root_scalar(_p_minus_target_logc, bracket=(lo, hi), method="bisect", args=(t, logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr), xtol=1e-12, rtol=1e-10, maxiter=200)
        if sol.converged:
            return float(np.exp(sol.root))
    except Exception:
        _warn_once("bracket", f"Root solve exception at p={t:.3g}; returning NaN.")

    return float('nan')


def heff_at_cstar(c_star: float, p_star: float,
                  p_min: float, p_max: float,
                  N_tar: float, N_tsr: float,
                  KdI1: float, KdA1: float,
                  KdI2: float, KdA2: float,
                  *, return_abs: bool = False) -> float:
    c = float(c_star)
    p = float(p_star)
    if not (np.isfinite(c) and c > 0 and 0.0 < p < 1.0):
        return float('nan')

    dDelta_dc = (
        N_tar * (1.0 / (KdI1 + c) - 1.0 / (KdA1 + c)) +
        N_tsr * (1.0 / (KdI2 + c) - 1.0 / (KdA2 + c))
    )
    dpdc = -p * (1.0 - p) * dDelta_dc

    denom = p - p_min
    if denom == 0.0 or not np.isfinite(denom):
        return float('nan')
    neff = 2.0 * (c / denom) * dpdc

    pmid = 0.5 * (p_min + p_max)
    if np.isfinite(pmid) and abs(p - pmid) <= 1e-12 * max(1.0, abs(pmid)):
        span = p_max - p_min
        if span != 0.0 and np.isfinite(span):
            neff = -4.0 * (c * p * (1.0 - p) / span) * dDelta_dc

    return float(abs(neff) if return_abs else neff)


# ============================================================================
# CONCENTRATION GRID AND CHANNEL CAPACITY  (7D_Sweep_Code.py lines 288-580)
# ============================================================================
def pick_c_grid_from_params(*,
    L0: float, KdI1: float, KdA1: float,
    KdI2: float, KdA2: float,
    N_tar: float, N_tsr: float,
    alpha_low: float = 1e-3, alpha_high: float = 1-1e-3,
    pad_decades: float = 0.25,
    N_transition: int = 25,
    M_min: int = 60, M_max: int = 400
) -> tuple[np.ndarray, np.ndarray, dict]:

    def _broad_grid(c_center: float,
                    p0: float, pinf: float,
                    delta_p: float):
        c_vals = np.logspace(
            np.log10(c_center) - 4,
            np.log10(c_center) + 4,
            max(M_min, 100)
        )
        pa = p_active(
            c_vals, float(np.log(L0)),
            KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr
        )
        return (c_vals,
                pa,
                dict(p0=p0, pinf=pinf, delta_p=delta_p,
                     c_lo=np.nan, c_hi=np.nan,
                     c10=np.nan, c50=np.nan, c90=np.nan))

    logL0 = float(np.log(L0))
    p0, pinf = endpoints_p0_pinf(
        logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr
    )
    delta_p = abs(pinf - p0)

    if (N_tar + N_tsr) == 0 or delta_p < FLAT_DELTA_P_THRESH:
        _warn_once(
            "flat",
            f"Flat curve: delta_p={delta_p:.2e} for "
            f"(L0={L0:.3g}, N_tar={N_tar}, N_tsr={N_tsr})."
        )
        kd_list = [KdI1, KdA1] + ([KdI2, KdA2] if N_tsr > 0 else [])
        c_ctr = float(np.exp(np.mean(np.log(np.array(kd_list, dtype=float)))))
        return _broad_grid(c_ctr, p0, pinf, delta_p)

    def relp(frac): return p0 + frac * (pinf - p0)
    p_lo = relp(alpha_low)
    p_hi = relp(alpha_high)
    p10, p50_mid, p90 = relp(0.10), relp(0.50), relp(0.90)

    kd_list = [KdI1, KdA1] + ([KdI2, KdA2] if N_tsr > 0 else [])
    c_ctr = float(np.exp(np.mean(np.log(np.array(kd_list, dtype=float)))))

    c_lo = solve_c_at_p(p_lo, logL0, KdI1, KdA1,
                        KdI2, KdA2, N_tar, N_tsr)
    c_hi = solve_c_at_p(p_hi, logL0, KdI1, KdA1,
                        KdI2, KdA2, N_tar, N_tsr)
    c10  = solve_c_at_p(p10,  logL0, KdI1, KdA1,
                        KdI2, KdA2, N_tar, N_tsr)
    c50  = solve_c_at_p(p50_mid, logL0, KdI1, KdA1,
                        KdI2, KdA2, N_tar, N_tsr)
    c90  = solve_c_at_p(p90,  logL0, KdI1, KdA1,
                        KdI2, KdA2, N_tar, N_tsr)

    targets = np.array([c_lo, c_hi, c10, c50, c90], float)
    if not np.all(np.isfinite(targets)) or np.any(targets <= 0):
        _warn_once(
            "interp",
            "Root solve failed for one or more targets; "
            "using interpolation fallback."
        )
        c_vals_tmp = np.logspace(
            np.log10(c_ctr) - 4,
            np.log10(c_ctr) + 4,
            max(M_min, 200)
        )
        pa_tmp = p_active(
            c_vals_tmp, logL0,
            KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr
        )
        inc = pa_tmp[0] < pa_tmp[-1]
        pgrid, cgrid = ((pa_tmp, c_vals_tmp) if inc
                        else (pa_tmp[::-1], c_vals_tmp[::-1]))
        try:
            c10 = float(np.interp(p10,      pgrid, cgrid))
            c50 = float(np.interp(p50_mid,  pgrid, cgrid))
            c90 = float(np.interp(p90,      pgrid, cgrid))
            c_lo = float(np.interp(p_lo,    pgrid, cgrid))
            c_hi = float(np.interp(p_hi,    pgrid, cgrid))
        except Exception:
            _warn_once(
                "interp",
                "Interpolation fallback failed; returning broad grid."
            )
            return _broad_grid(c_ctr, p0, pinf, delta_p)

    c_left  = min(c_lo, c_hi) / (10.0 ** pad_decades)
    c_right = max(c_lo, c_hi) * (10.0 ** pad_decades)
    trans_dec = max(abs(np.log10(c90) - np.log10(c10)), 1e-3)
    span_dec  = abs(np.log10(c_right) - np.log10(c_left))
    trans_dec = max(trans_dec, min(0.2, span_dec / 10.0))
    M_needed  = int(np.ceil(N_transition * (span_dec / trans_dec))) + 10
    M = int(np.clip(M_needed, M_min, M_max))

    c_vals = np.logspace(np.log10(c_left), np.log10(c_right), M)
    pa = p_active(
        c_vals, logL0,
        KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr
    )

    return c_vals, pa, dict(
        p0=p0, pinf=pinf, delta_p=delta_p,
        c_lo=c_lo, c_hi=c_hi, c10=c10, c50=c50, c90=c90
    )


def build_channel_matrix_binary(p_active_vals: np.ndarray,
                                *, clip_eps: float = 1e-12) -> np.ndarray:
    pa = np.asarray(p_active_vals, dtype=float)
    if pa.ndim != 1 or not np.all(np.isfinite(pa)):
        raise ValueError("p_active_vals must be finite 1D.")
    if np.any(pa < 0.0) or np.any(pa > 1.0):
        raise ValueError("p_active_vals must lie within [0,1].")
    pa = np.clip(pa, clip_eps, 1.0 - clip_eps)
    P = np.column_stack([1.0 - pa, pa])
    P /= P.sum(axis=1, keepdims=True)
    return P


CAP_GRID_M_MIN = 30
CAP_GRID_M_MAX = 80

BA_CAP_TOL_BITS = 3e-6
BA_REQUIRE_CONSEC = 2
BA_MAX_ITER = 5000
BA_TOL_R_L1 = 1e-6
BA_R_FLOOR = 1e-300
BA_WARM_R_FLOOR = 1e-12
BA_WARM_UNIFORM_MIX = 1e-3


def blahut_arimoto(
    P: np.ndarray,
    *,
    cap_tol_bits: float = 3e-6,
    check_every: int = 25,
    patience: int = 2,
    min_iter: int = 200,
    max_iter: int = 5000,
    r_init: np.ndarray | None = None,
    tol_r: float | None = None,
    tol_r_check_every: int | None = None,
    require_consecutive: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray, int]:

    EPS = 1e-300
    P = np.asarray(P, float)

    rowsum = P.sum(axis=1, keepdims=True)
    if np.any(rowsum <= 0):
        raise ValueError("Some rows of P sum to zero.")
    P = P / rowsum

    M, K = P.shape

    if r_init is None:
        r = np.full(M, 1.0 / M, dtype=float)
    else:
        r = np.asarray(r_init, float)
        r = np.maximum(r, BA_R_FLOOR)
        s = float(np.sum(r))
        r = (r / s) if (np.isfinite(s) and s > 0) else np.full(M, 1.0 / M, dtype=float)

    if require_consecutive is not None:
        patience = int(require_consecutive)

    check_every = int(max(1, check_every))
    patience = int(max(1, patience))
    min_iter = int(max(1, min_iter))
    max_iter = int(max(1, max_iter))

    if tol_r_check_every is None:
        tol_r_check_every = check_every
    tol_r_check_every = int(max(1, tol_r_check_every))

    prevC = None
    smallC = 0

    prev_r_check = None
    smallR = 0

    ln2 = np.log(2.0)

    for it in range(1, max_iter + 1):
        q = np.maximum(r @ P, EPS)  # (K,)
        logz = np.sum(P * (np.log(P + EPS) - np.log(q[None, :] + EPS)), axis=1)  # (M,)
        z = np.exp(logz)
        r = r * z
        r = np.maximum(r, BA_R_FLOOR)
        r = r / np.sum(r)

        if tol_r is not None and (it % tol_r_check_every == 0) and (it >= min_iter):
            if prev_r_check is not None:
                dr_l1 = float(np.sum(np.abs(r - prev_r_check)))
                if dr_l1 < float(tol_r):
                    smallR += 1
                else:
                    smallR = 0
            prev_r_check = r.copy()

            if smallR >= patience:
                C_now = float(np.sum(r * logz) / ln2)
                C_now = max(0.0, min(1.0, C_now))
                pX = r
                pY = np.maximum(pX @ P, EPS)
                return C_now, pX, pY, it

        if it % check_every == 0:
            C_now = float(np.sum(r * logz) / ln2)
            C_now = max(0.0, min(1.0, C_now))

            if prevC is not None and it >= min_iter:
                if abs(C_now - prevC) < cap_tol_bits:
                    smallC += 1
                else:
                    smallC = 0
                if smallC >= patience:
                    pX = r
                    pY = np.maximum(pX @ P, EPS)
                    return C_now, pX, pY, it

            prevC = C_now

    pX = r
    pY = np.maximum(pX @ P, EPS)
    q = np.maximum(pX @ P, EPS)
    logz = np.sum(P * (np.log(P + EPS) - np.log(q[None, :] + EPS)), axis=1)
    C_bits = float(np.sum(pX * logz) / ln2)
    C_bits = max(0.0, min(1.0, C_bits))
    return C_bits, pX, pY, max_iter


@dataclass
class BAWarmState:
    c_vals: np.ndarray
    pX: np.ndarray


def warm_start_r_from_prev(
    c_new: np.ndarray,
    prev: BAWarmState | None,
) -> np.ndarray | None:

    if prev is None:
        return None

    c_prev = np.asarray(prev.c_vals, float)
    r_prev = np.asarray(prev.pX, float)
    c_new = np.asarray(c_new, float)

    if c_prev.ndim != 1 or r_prev.ndim != 1 or c_new.ndim != 1:
        return None
    if c_prev.size != r_prev.size or c_prev.size < 2 or c_new.size < 2:
        return None
    if np.any(~np.isfinite(c_prev)) or np.any(~np.isfinite(r_prev)) or np.any(~np.isfinite(c_new)):
        return None
    if np.any(c_prev <= 0) or np.any(c_new <= 0):
        return None

    r_prev = np.maximum(r_prev, 0.0)
    s = float(np.sum(r_prev))
    if not (np.isfinite(s) and s > 0):
        return None
    r_prev = r_prev / s

    x_prev = np.log(c_prev)
    x_new = np.log(c_new)

    if not np.all(np.diff(x_prev) > 0):
        order = np.argsort(x_prev)
        x_prev = x_prev[order]
        r_prev = r_prev[order]

    r_new = np.interp(x_new, x_prev, r_prev, left=float(r_prev[0]), right=float(r_prev[-1]))
    r_new = np.maximum(r_new, BA_WARM_R_FLOOR)

    s2 = float(np.sum(r_new))
    if not (np.isfinite(s2) and s2 > 0):
        return None
    r_new = r_new / s2

    alpha = float(np.clip(BA_WARM_UNIFORM_MIX, 0.0, 1.0))
    if alpha > 0.0:
        M = int(r_new.size)
        r_new = (1.0 - alpha) * r_new + alpha * (1.0 / M)
        r_new = np.maximum(r_new, BA_WARM_R_FLOOR)
        r_new = r_new / np.sum(r_new)
    return r_new


# ============================================================================
# ALL-METRICS-AT-A-POINT WRAPPER  (7D_Sweep_Code.py lines 593-665)
# ============================================================================
def metrics_at_params_auto_c(
    *,
    L0: float,
    KdI1: float,
    KdA1: float,
    KdI2: float,
    KdA2: float,
    N_tar: float,
    N_tsr: float,
    mode: str = "binary",
    warm_state: BAWarmState | None = None,
) -> dict:
    c_vals, pa, info = pick_c_grid_from_params(
        L0=L0, KdI1=KdI1, KdA1=KdA1,
        KdI2=KdI2, KdA2=KdA2,
        N_tar=N_tar, N_tsr=N_tsr,
        M_min=CAP_GRID_M_MIN,
        M_max=CAP_GRID_M_MAX
    )

    logL0 = float(np.log(L0))
    p0, pinf = endpoints_p0_pinf(
        logL0, KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr
    )

    DR_p_signed = float(pinf - p0)
    DR_out_mag = abs(DR_p_signed)

    heff = np.nan
    c50 = np.nan

    if np.isfinite(p0) and np.isfinite(pinf) and abs(pinf - p0) > 0:
        p_star = 0.5 * (p0 + pinf)
        c50 = solve_c_at_p(
            p_star, logL0,
            KdI1, KdA1, KdI2, KdA2, N_tar, N_tsr
        )

        if np.isfinite(c50) and c50 > 0:
            heff = heff_at_cstar(
                c50, p_star,
                p0, pinf,
                N_tar, N_tsr,
                KdI1, KdA1, KdI2, KdA2,
                return_abs=True
            )

    if mode != "binary":
        print("[WARN] metrics_at_params_auto_c: only 'binary' channel implemented; using binary.")

    P = build_channel_matrix_binary(pa)

    r_init = warm_start_r_from_prev(c_vals, warm_state)

    C_bits, pX_opt, pY, iters = blahut_arimoto(
        P,
        r_init=r_init,
        tol_r=BA_TOL_R_L1,
        cap_tol_bits=BA_CAP_TOL_BITS,
        require_consecutive=BA_REQUIRE_CONSEC,
        max_iter=BA_MAX_ITER,
    )

    return {
        "C_bits": float(C_bits),
        "nH": float(heff) if np.isfinite(heff) else np.nan,
        "DR_p": float(DR_p_signed),
        "DR_out": float(DR_out_mag),
        "c50": float(c50) if np.isfinite(c50) else np.nan,
        "iters": int(iters),
        "ba_c_vals": c_vals,
        "ba_pX": pX_opt,
    }


# ============================================================================
# SWEEP AXIS CONSTANTS AND PROGRESS PRINTER  (7D_Sweep_Code.py lines 667-691)
# ============================================================================
INDEP_VARS = ["L0", "KdI1", "KdA1", "KdI2", "KdA2", "N_tar", "N_tsr"]
LOG_VARS   = {"L0", "KdI1", "KdA1", "KdI2", "KdA2"}
DEP_VARS   = ["C_bits", "nH", "DR_out", "DR_p", "c50"]
DEP_DEFAULT = ("C_bits", "nH", "DR_out")


class ProgressPrinter:
    def __init__(self, total: int,
                 every: float = 0.001,
                 min_seconds: float = 0.25):
        self.total = max(int(total), 1)
        self.every = float(every)
        self.min_seconds = float(min_seconds)
        self.start = time.time()
        self.last_mark = 0.0
        self.last_print = self.start

    @staticmethod
    def _hm_text(seconds: float) -> str:
        if not np.isfinite(seconds):
            return ""
        total_minutes = int(round(seconds / 60.0))
        h, m = divmod(total_minutes, 60)
        return f" ({h}h {m}m)"


# ============================================================================
# SLICING AND NAMING FOR HEATMAPS  (7D_Sweep_Code.py lines 709-858)
# ============================================================================
def _grid_of(results: dict, name: str) -> np.ndarray:
    return np.asarray(results["grids"][name])


def _nearest_index(arr: np.ndarray, value: float) -> int:
    arr = np.asarray(arr, float)
    return int(np.argmin(np.abs(arr - float(value))))


def slice2_nd(results: dict, dep: str,
              xvar: str, yvar: str,
              *, agg: str = "median",
              fixed: dict | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dep not in DEP_VARS:
        raise ValueError(f"Unknown dep '{dep}'. Options: {DEP_VARS}")
    if xvar not in INDEP_VARS or yvar not in INDEP_VARS:
        raise ValueError(f"xvar/yvar must be in {INDEP_VARS}.")

    Z = np.asarray(results[dep], float)
    ax_map = {name: i for i, name in enumerate(INDEP_VARS)}
    ax_x, ax_y = ax_map[xvar], ax_map[yvar]

    indexer: list[object] = [slice(None)] * Z.ndim
    if agg == "fixed":
        if not fixed:
            raise ValueError("agg='fixed' requires fixed={var: value}.")
        for name, ax in ax_map.items():
            if name in (xvar, yvar):
                continue
            grid = _grid_of(results, name)
            val  = fixed.get(name, grid[len(grid)//2])
            idx  = _nearest_index(grid, float(val))
            indexer[ax] = idx

    sub = Z[tuple(indexer)]

    surviving_axes = [ax for ax, sel in enumerate(indexer)
                      if isinstance(sel, slice)]
    pos_x = surviving_axes.index(ax_x)
    pos_y = surviving_axes.index(ax_y)
    if (pos_y, pos_x) != (0, 1):
        sub = np.moveaxis(sub, (pos_y, pos_x), (0, 1))

    reduce_axes = tuple(ax for ax in range(sub.ndim)
                        if ax not in (0, 1))
    if reduce_axes:
        if agg in ("median", "fixed"):
            sub = np.nanmedian(sub, axis=reduce_axes)
        elif agg == "mean":
            sub = np.nanmean(sub, axis=reduce_axes)
        else:
            raise ValueError("agg must be 'median', 'mean', or 'fixed'.")

    X = _grid_of(results, xvar)
    Y = _grid_of(results, yvar)
    if sub.shape != (len(Y), len(X)):
        sub = sub.reshape(len(Y), len(X))
    return X, Y, sub


def _log_edges(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, float)
    if g.ndim != 1 or np.any(g <= 0):
        raise ValueError("Grid must be 1D and strictly positive for log edges.")
    if g.size == 1:
        r = 10**0.1
        return np.array([g[0]/r, g[0]*r], float)
    mid = np.sqrt(g[1:] * g[:-1])
    left0  = g[0]**2  / mid[0]
    rightN = g[-1]**2 / mid[-1]
    return np.concatenate([[left0], mid, [rightN]])


def _meta_of(results: dict) -> dict | None:
    m = results.get("meta", None)
    if isinstance(m, np.ndarray):
        try:
            m = m.item()
        except Exception:
            pass
    return m


USE_N1N2 = True

_VAR_LATEX = {
    "L0":   r"$L_0$",
    "KdI1": r"$K_{d}^{(I),1}$",
    "KdA1": r"$K_{d}^{(A),1}$",
    "KdI2": r"$K_{d}^{(I),2}$",
    "KdA2": r"$K_{d}^{(A),2}$",
}

if USE_N1N2:
    _VAR_LATEX.update({
        "N_tar": r"$N_1$",
        "N_tsr": r"$N_2$",
    })
else:
    _VAR_LATEX.update({
        "N_tar": r"$n$",
        "N_tsr": r"$m$",
    })

_DEP_FULL = {
    "C_bits": r"Channel capacity (bits)",
    "nH":     r"Effective Hill coefficient $n_{\mathrm{eff}}$",
    "DR_out": r"Dynamic range",
    "DR_p":   r"$p(\infty)-p(0)$",
    "c50":    r"$c_{50}$ (mM)",
}


def _vname(name: str) -> str:
    return _VAR_LATEX.get(name, name)


def _dname(dep: str) -> str:
    return _DEP_FULL.get(dep, dep)


_VAR_FILE = {
    "L0":   "L0",
    "KdI1": "KI1",
    "KdA1": "KA1",
    "KdI2": "KI2",
    "KdA2": "KA2",
    "N_tar": "N1" if USE_N1N2 else "n",
    "N_tsr": "N2" if USE_N1N2 else "m",
}

_DEP_FILE = {
    "C_bits": "C",
    "nH":     "neff",
    "DR_out": "DR",
    "DR_p":   "DRp",
    "c50":    "c50",
}


def _slug(s: str) -> str:
    s = str(s)
    s = s.replace("$", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\.]+", "", s)
    return s.strip("_")


def _vfile(name: str) -> str:
    return _VAR_FILE.get(name, _slug(name))


def _dfile(dep: str) -> str:
    return _DEP_FILE.get(dep, _slug(dep))


# ============================================================================
# FULL-PANEL 21-HEATMAP EXPORTER  (7D_Sweep_Code.py lines 1025-1136 partial,
# 2796-3032)
# Used only by the SI heatmap generator.
# ============================================================================
def heatmap_color_limits(
    arr: np.ndarray,
    *,
    norm_mode: str = "fixed",
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:

    if (vmin is not None) and (vmax is not None):
        vmin = float(vmin)
        vmax = float(vmax)
        if vmin == vmax:
            eps = 1e-12 if vmin == 0.0 else abs(vmin) * 1e-12
            return vmin - eps, vmax + eps
        return vmin, vmax

    v = np.asarray(arr, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0

    norm_mode = str(norm_mode).strip().lower()
    if norm_mode == "fixed":
        vmin2 = float(np.min(v))
        vmax2 = float(np.max(v))
    elif norm_mode == "percentile":
        vmin2 = float(np.nanpercentile(v, float(low_pct)))
        vmax2 = float(np.nanpercentile(v, float(high_pct)))
        if (not np.isfinite(vmin2)) or (not np.isfinite(vmax2)) or (vmin2 == vmax2):
            vmin2 = float(np.min(v))
            vmax2 = float(np.max(v))
    else:
        raise ValueError("norm_mode must be 'fixed' or 'percentile'.")

    if vmin2 == vmax2:
        eps = 1e-12 if vmin2 == 0.0 else abs(vmin2) * 1e-12
        return vmin2 - eps, vmax2 + eps

    return vmin2, vmax2


def _global_range(
    arr: np.ndarray,
    *,
    norm_mode: str = "fixed",
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:

    return heatmap_color_limits(
        arr,
        norm_mode=norm_mode,
        low_pct=low_pct,
        high_pct=high_pct,
        vmin=vmin,
        vmax=vmax,
    )


def load_results_from_npz(npz_path: str | Path) -> dict:

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    grids = {
        "L0":   np.asarray(data["L0_grid"]),
        "KdI1": np.asarray(data["KdI1_grid"]),
        "KdA1": np.asarray(data["KdA1_grid"]),
        "KdI2": np.asarray(data["KdI2_grid"]),
        "KdA2": np.asarray(data["KdA2_grid"]),
        "N_tar": np.asarray(data["N_tar_grid"]),
        "N_tsr": np.asarray(data["N_tsr_grid"]),
    }

    results = {
        "C_bits": np.asarray(data["C_bits"], dtype=float),
        "nH":     np.asarray(data["nH"], dtype=float),
        "DR_out": np.asarray(data["DR_out"], dtype=float),
        "DR_p":   np.asarray(data["DR_p"], dtype=float),
        "c50":    np.asarray(data["c50"], dtype=float),
        "grids": grids,
    }

    if "done_mask" in data:
        results["done_mask"] = np.asarray(data["done_mask"]).astype(bool)
    if "cursor" in data:
        results["cursor"] = int(data["cursor"])
    if "complete" in data:
        results["complete"] = bool(data["complete"])

    return results


def plot_full_panel_21(
    results: dict,
    dep: str,
    *,
    fixed: dict,
    dot: dict | None = None,
    savepath: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    style: dict | None = None,
    dpi: int = 600,
):

    st = {} if style is None else dict(style)

    st.setdefault("figsize", (22.0, 9.5))
    st.setdefault("nrows", 7)
    st.setdefault("ncols", 3)

    st.setdefault("wspace", 0.40)
    st.setdefault("hspace", 0.40)
    st.setdefault("left", 0.06)
    st.setdefault("right", 0.93)
    st.setdefault("top", 0.97)
    st.setdefault("bottom", 0.08)

    st.setdefault("axis_label_fs", 11)
    st.setdefault("tick_fs", 9)
    st.setdefault("tick_len", 3.5)
    st.setdefault("tick_wid", 1.0)

    st.setdefault("panel_label_fs", 12)
    st.setdefault("panel_label_weight", "normal")
    st.setdefault("panel_label_xy", (-0.22, 1.04))

    st.setdefault("dot_size", 55)
    st.setdefault("dot_edge_wid", 1.2)

    st.setdefault("cbar_tick_fs", 9)
    st.setdefault("cbar_label_fs", 12)
    st.setdefault("cbar_labelpad", 14)

    st.setdefault("cmap_name", "viridis")
    st.setdefault("bad_alpha", 0.2)

    nrows = int(st["nrows"])
    ncols = int(st["ncols"])

    pairs = [
        (INDEP_VARS[i], INDEP_VARS[j])
        for i in range(len(INDEP_VARS))
        for j in range(i + 1, len(INDEP_VARS))
    ]
    if len(pairs) != 21:
        raise ValueError("Expected 21 parameter pairs.")
    if nrows * ncols != 21:
        raise ValueError(f"Grid must hold 21 panels; got nrows*ncols={nrows*ncols}.")

    fig = plt.figure(figsize=st["figsize"])
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1] * ncols + [0.045],
        wspace=st["wspace"],
        hspace=st["hspace"],
        figure=fig,
    )

    cmap = mpl.colormaps[st["cmap_name"]].copy()
    cmap.set_bad(alpha=float(st["bad_alpha"]))

    pcm_last = None

    for k, (xv, yv) in enumerate(pairs):
        r = k // ncols
        c = k % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.set_box_aspect(1)

        X, Y, Z = slice2_nd(results, dep, xv, yv, agg="fixed", fixed=fixed)

        use_logx = xv in LOG_VARS
        use_logy = yv in LOG_VARS
        Xe = _log_edges(X) if use_logx else np.linspace(X.min(), X.max(), len(X) + 1)
        Ye = _log_edges(Y) if use_logy else np.linspace(Y.min(), Y.max(), len(Y) + 1)

        Zm = np.ma.masked_invalid(Z)

        pcm = ax.pcolormesh(
            Xe, Ye, Zm,
            shading="flat",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        pcm_last = pcm

        if use_logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        if use_logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))

        ax.set_xlabel(_vname(xv), fontsize=st["axis_label_fs"])
        ax.set_ylabel(_vname(yv), fontsize=st["axis_label_fs"])
        ax.tick_params(
            axis="both", which="major",
            labelsize=st["tick_fs"],
            length=st["tick_len"],
            width=st["tick_wid"],
        )
        ax.tick_params(
            axis="both", which="minor",
            length=0.7 * st["tick_len"],
            width=st["tick_wid"],
        )

        panel = chr(ord("a") + k)
        px, py = st["panel_label_xy"]
        ax.text(
            px, py, f"({panel})",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=st["panel_label_fs"],
            fontweight=st["panel_label_weight"],
            clip_on=False,
        )

        if dot is not None and (xv in dot) and (yv in dot):
            ax.scatter(
                [float(dot[xv])], [float(dot[yv])],
                s=st["dot_size"],
                facecolors="white",
                edgecolors="black",
                linewidths=st["dot_edge_wid"],
                zorder=5,
            )

    if pcm_last is None:
        raise RuntimeError("No panels were rendered; pcm_last is None.")

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(pcm_last, cax=cax)
    cbar.ax.tick_params(labelsize=st["cbar_tick_fs"])
    cbar.set_label(_dname(dep), fontsize=st["cbar_label_fs"], labelpad=st["cbar_labelpad"])

    fig.subplots_adjust(
        left=st["left"], right=st["right"],
        top=st["top"], bottom=st["bottom"]
    )

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        plt.show()


FULL_PANEL_STYLE = dict(
    figsize=(9.5, 20.0),
    nrows=7,
    ncols=3,
    wspace=0.80,
    hspace=0.14,
    left=0.06,
    right=0.93,
    top=0.98,
    bottom=0.06,
    axis_label_fs=10,
    tick_fs=8,
    tick_len=1.0,
    tick_wid=0.5,
    panel_label_fs=18,
    panel_label_weight="normal",
    panel_label_xy=(-0.22, 1.04),
    dot_size=40,
    dot_edge_wid=0.5,
    cbar_tick_fs=8,
    cbar_label_fs=10,
    cbar_labelpad=14,
    cmap_name="viridis",
    bad_alpha=0.2,
)


def export_full_panel_pages_per_strain(
    results: dict,
    bio_list: list[dict],
    outdir: str | Path,
    *,
    fmt: str = "pdf",
    style: dict | None = None,
    norm_mode: str = "fixed",
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    vmin: float | None = None,
    vmax: float | None = None,
):

    outdir = Path(outdir) / "plots" / "full_panels"
    outdir.mkdir(parents=True, exist_ok=True)

    if style is None:
        style = FULL_PANEL_STYLE

    vC = _global_range(results["C_bits"], norm_mode=norm_mode, low_pct=low_pct, high_pct=high_pct, vmin=vmin, vmax=vmax)
    vH = _global_range(results["nH"],     norm_mode=norm_mode, low_pct=low_pct, high_pct=high_pct, vmin=vmin, vmax=vmax)
    vD = _global_range(results["DR_out"], norm_mode=norm_mode, low_pct=low_pct, high_pct=high_pct, vmin=vmin, vmax=vmax)

    for d in bio_list:
        label = str(d.get("label", d.get("name", "")))
        fixed = {k: d[k] for k in INDEP_VARS}

        for dep, (vmin_dep, vmax_dep) in (
            ("C_bits", vC),
            ("DR_out", vD),
            ("nH",     vH),
        ):
            savepath = outdir / f"strain{label}_{_dfile(dep)}_full.{fmt}"
            plot_full_panel_21(
                results, dep,
                fixed=fixed,
                dot=d,
                savepath=savepath,
                vmin=vmin_dep, vmax=vmax_dep,
                style=style,
                dpi=600,
            )

        print(f"[FULL PANEL] wrote strain {label} full panel pages -> {outdir.resolve()}")


# ============================================================================
# STRAIN DEFINITIONS  (7D_Sweep_Code.py lines 1474-1546, 1597-1637)
# ============================================================================
def bio_dots_all10() -> list[dict]:
    # Mello & Tu 2005, PNAS 102(48), 17354-17359.
    # Receptor-specific params (p. 17356, paragraph after Eq. 10):
    #   l_1 = 1.23, C_1 = 0.449, K_1 = 49.2 uM = 0.0492 mM  (Tar)
    #   l_2 = 1.54, C_2 = 0.314, K_2 = 34.5 uM = 0.0345 mM  (Tsr)
    #   l_0_bar = 0.826  (average equilibrium constant for cytoplasmic components)
    KdI1 = 0.0492
    C1 = 0.449
    KdA1 = KdI1 / C1
    KdI2 = 0.0345
    C2 = 0.314
    KdA2 = KdI2 / C2
    ell0_bar, ell1, ell2 = 0.826, 1.23, 1.54

    strains = [
        ("strain1",   0.6,  2.0,  4.95, 16.5),
        ("strain2",   1.0,  2.0,  4.00,  8.00),
        ("strain3",   2.0,  2.0,  4.39,  4.39),
        ("strain4",   6.0,  2.0, 18.70,  6.24),
        ("strain5",   1.0,  0.0, 14.00,  0.0),
        ("strain6",   2.0,  0.0, 29.80,  0.0),
        ("strain7",   6.0,  0.0, 73.50,  0.0),
        ("strain8",   0.0,  0.6,  0.0,   9.85),
        ("strain9",   0.0,  1.4,  0.0,  15.20),
        ("strain10",  0.0, 10.0,  0.0,  32.30),
    ]

    # Mello & Tu 2005, Eq. 10 with the strain-dependent cytoplasmic exponent:
    #   L_j = l0_bar^(N_{j,1} / f_{j,1}) * l_1^N_{j,1} * l_2^N_{j,2}
    # For Tar-only strains use N_{j,1}/f_{j,1}; for Tsr-only strains N_{j,2}/f_{j,2}.
    dots = []
    for name, f1, f2, N1, N2 in strains:
        cytoplasm_exp = (N1 / f1) if f1 > 0 else (N2 / f2)
        L_paper = float(
            (ell0_bar ** cytoplasm_exp) * (ell1 ** N1) * (ell2 ** N2)
        )
        L0 = 1.0 / L_paper
        dots.append({
            "name": name,
            "label": name.replace("strain", ""),
            "L0": L0,
            "KdI1": KdI1, "KdA1": KdA1,
            "KdI2": KdI2, "KdA2": KdA2,
            "N_tar": N1,  "N_tsr": N2,
            "f_tar": f1,  "f_tsr": f2,
        })
    return dots


def expand_grids_to_cover_biodots(grids: dict, bio_list: list[dict],
                                  *, L0_pad: float = 2.0,
                                  N_pad: int = 2) -> dict:
    g = np.asarray(grids["L0"], float)
    Lvals = np.array([d["L0"] for d in bio_list], float)
    Lmin, Lmax = float(Lvals.min()), float(Lvals.max())
    lo = min(g.min(), Lmin / L0_pad)
    hi = max(g.max(), Lmax * L0_pad)
    grids["L0"] = np.logspace(np.log10(lo), np.log10(hi), g.size)

    for key in ("N_tar", "N_tsr"):
        gN = np.asarray(grids[key], float)
        vals = np.array([d[key] for d in bio_list], float)
        nmin = max(0.0, float(vals.min()) - N_pad)
        nmax = float(vals.max()) + N_pad
        grids[key] = np.linspace(nmin, nmax, num=gN.size, dtype=float)

    return grids


def _log_grid_around(anchor: float,
                     span_decades: float = 3.0,
                     points: int = 7) -> np.ndarray:

    a = float(anchor)
    if a <= 0:
        raise ValueError("Anchor for log grid must be > 0.")
    half = span_decades / 2.0
    left  = a / (10.0 ** half)
    right = a * (10.0 ** half)
    g = np.logspace(np.log10(left), np.log10(right), max(int(points), 3))
    g = np.unique(np.append(g, a))
    return np.sort(g)


def build_grids_pilot(bio: dict,
                      *,
                      points_log: int = 7,
                      points_N: int = 9,
                      L0_span_dec: float = 3.0,
                      K_span_dec: float = 3.0) -> dict:

    L0_grid   = _log_grid_around(bio["L0"],   span_decades=L0_span_dec, points=points_log)
    KdI1_grid = _log_grid_around(bio["KdI1"], span_decades=K_span_dec,  points=points_log)
    KdA1_grid = _log_grid_around(bio["KdA1"], span_decades=K_span_dec,  points=points_log)
    KdI2_grid = _log_grid_around(bio["KdI2"], span_decades=K_span_dec,  points=points_log)
    KdA2_grid = _log_grid_around(bio["KdA2"], span_decades=K_span_dec,  points=points_log)

    def _lin_float_grid(anchor_val: float, low: float, high: float, points: int) -> np.ndarray:
        g = np.linspace(float(low), float(high), num=int(points), dtype=float)
        g = np.unique(np.append(g, float(anchor_val))).astype(float)
        return np.sort(g)

    N_tar_grid = _lin_float_grid(bio["N_tar"], low=0.0,  high=80.0, points=points_N)
    N_tsr_grid = _lin_float_grid(bio["N_tsr"], low=0.0,  high=40.0, points=points_N)

    return dict(
        L0=L0_grid, KdI1=KdI1_grid, KdA1=KdA1_grid,
        KdI2=KdI2_grid, KdA2=KdA2_grid,
        N_tar=N_tar_grid, N_tsr=N_tsr_grid
    )


# ============================================================================
# BA WRAPPER FOR FIGURE SCRIPTS  (7D_Sweep_Code.py lines 1640-1667)
# ============================================================================
def ba_at_params(L0: float, KdI1: float, KdA1: float,
                 KdI2: float, KdA2: float,
                 N_tar: float, N_tsr: float) -> dict:
    c_vals, pa, info = pick_c_grid_from_params(
        L0=L0, KdI1=KdI1, KdA1=KdA1,
        KdI2=KdI2, KdA2=KdA2,
        N_tar=N_tar, N_tsr=N_tsr,
        M_min=CAP_GRID_M_MIN,
        M_max=CAP_GRID_M_MAX
    )
    P = build_channel_matrix_binary(pa)
    C_bits, pX, pY, iters = blahut_arimoto(
        P,
        tol_r=BA_TOL_R_L1,
        cap_tol_bits=BA_CAP_TOL_BITS,
        require_consecutive=BA_REQUIRE_CONSEC,
        max_iter=BA_MAX_ITER,
    )
    return dict(
        c_vals=c_vals,
        pa=pa,
        P=P,
        pX=pX,
        pY=pY,
        C_bits=C_bits,
        info=info,
        iters=iters
    )


# ============================================================================
# GRADIENT-NORM UTILITIES  (7D_Sweep_Code.py lines 2311-2488)
# Used by strain-table generator for the reviewer concern 4 gradient column.
# ============================================================================
@dataclass(frozen=True)
class TableBStepConfig:
    dlog10: float = 0.01
    dN: float = 1.0


def _tableb_bounds_from_grids(grids: dict) -> Dict[str, Tuple[float, float]]:
    b: Dict[str, Tuple[float, float]] = {}
    for name in INDEP_VARS:
        g = np.asarray(grids[name], float)
        b[name] = (float(np.nanmin(g)), float(np.nanmax(g)))
    return b


def _tableb_clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _tableb_key_from_params(p: Dict[str, float]) -> tuple[float, ...]:
    return tuple(float(np.round(float(p[name]), 14)) for name in INDEP_VARS)


_TABLEB_METRICS_CACHE: Dict[tuple[float, ...], dict] = {}


def reset_tableb_metrics_cache() -> None:
    _TABLEB_METRICS_CACHE.clear()


def tableb_metrics_cached(p: Dict[str, float]) -> dict:
    k = _tableb_key_from_params(p)
    got = _TABLEB_METRICS_CACHE.get(k)
    if got is not None:
        return got

    m = metrics_at_params_auto_c(**{name: float(p[name]) for name in INDEP_VARS})
    _TABLEB_METRICS_CACHE[k] = m
    return m


def tableb_dep_value(p: Dict[str, float], dep: str) -> float:
    return float(tableb_metrics_cached(p).get(dep, np.nan))


def tableb_grad_norm_fixed_steps(
    p: Dict[str, float],
    dep: str,
    *,
    steps: TableBStepConfig,
    bounds: Dict[str, Tuple[float, float]],
) -> float:
    if dep not in ("C_bits", "nH", "DR_out"):
        raise ValueError("dep must be one of: C_bits, nH, DR_out")

    g2 = 0.0

    for name in INDEP_VARS:
        lo, hi = bounds[name]
        base = float(p[name])

        if name in LOG_VARS:
            if not (np.isfinite(base) and base > 0):
                comp = 0.0
            else:
                fac = 10.0 ** float(steps.dlog10)
                plus = base * fac
                minus = base / fac

                can_plus = (plus <= hi) and (plus > 0)
                can_minus = (minus >= lo) and (minus > 0)

                if can_plus and can_minus:
                    p_plus = dict(p); p_plus[name] = plus
                    p_minus = dict(p); p_minus[name] = minus
                    f_plus = tableb_dep_value(p_plus, dep)
                    f_minus = tableb_dep_value(p_minus, dep)
                    comp = (f_plus - f_minus) / (2.0 * float(steps.dlog10)) if (np.isfinite(f_plus) and np.isfinite(f_minus)) else 0.0
                elif can_plus:
                    p_plus = dict(p); p_plus[name] = plus
                    f_plus = tableb_dep_value(p_plus, dep)
                    f0 = tableb_dep_value(p, dep)
                    comp = (f_plus - f0) / float(steps.dlog10) if (np.isfinite(f_plus) and np.isfinite(f0)) else 0.0
                elif can_minus:
                    p_minus = dict(p); p_minus[name] = minus
                    f0 = tableb_dep_value(p, dep)
                    f_minus = tableb_dep_value(p_minus, dep)
                    comp = (f0 - f_minus) / float(steps.dlog10) if (np.isfinite(f0) and np.isfinite(f_minus)) else 0.0
                else:
                    comp = 0.0

        else:
            d = float(steps.dN)
            plus = base + d
            minus = base - d

            can_plus = plus <= hi
            can_minus = minus >= lo

            if can_plus and can_minus:
                p_plus = dict(p); p_plus[name] = plus
                p_minus = dict(p); p_minus[name] = minus
                f_plus = tableb_dep_value(p_plus, dep)
                f_minus = tableb_dep_value(p_minus, dep)
                comp = (f_plus - f_minus) / (2.0 * d) if (np.isfinite(f_plus) and np.isfinite(f_minus)) else 0.0
            elif can_plus:
                p_plus = dict(p); p_plus[name] = plus
                f_plus = tableb_dep_value(p_plus, dep)
                f0 = tableb_dep_value(p, dep)
                comp = (f_plus - f0) / d if (np.isfinite(f_plus) and np.isfinite(f0)) else 0.0
            elif can_minus:
                p_minus = dict(p); p_minus[name] = minus
                f0 = tableb_dep_value(p, dep)
                f_minus = tableb_dep_value(p_minus, dep)
                comp = (f0 - f_minus) / d if (np.isfinite(f0) and np.isfinite(f_minus)) else 0.0
            else:
                comp = 0.0

        g2 += float(comp) * float(comp)

    return float(np.sqrt(g2))


# ============================================================================
# NPZ I/O + RESUMABLE 7D SWEEP  (7D_Sweep_Code.py lines 2490-2794)
# ============================================================================
def _save_npz(npz_path: Path,
              grids: dict,
              arrays: dict,
              done_mask: np.ndarray,
              iters: np.ndarray,
              cursor: int,
              complete: bool,
              meta: dict) -> None:

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        L0_grid=np.asarray(grids["L0"]),
        KdI1_grid=np.asarray(grids["KdI1"]),
        KdA1_grid=np.asarray(grids["KdA1"]),
        KdI2_grid=np.asarray(grids["KdI2"]),
        KdA2_grid=np.asarray(grids["KdA2"]),
        N_tar_grid=np.asarray(grids["N_tar"]),
        N_tsr_grid=np.asarray(grids["N_tsr"]),
        C_bits=np.asarray(arrays["C_bits"]),
        nH=np.asarray(arrays["nH"]),
        DR_out=np.asarray(arrays["DR_out"]),
        DR_p=np.asarray(arrays["DR_p"]),
        c50=np.asarray(arrays["c50"]),
        done_mask=np.asarray(done_mask),
        iters=np.asarray(iters),
        cursor=int(cursor),
        complete=bool(complete),
        meta=np.array(meta, dtype=object),
    )


def _load_npz(npz_path: Path) -> dict:

    data = np.load(npz_path, allow_pickle=True)
    grids = dict(
        L0=data["L0_grid"],
        KdI1=data["KdI1_grid"],
        KdA1=data["KdA1_grid"],
        KdI2=data["KdI2_grid"],
        KdA2=data["KdA2_grid"],
        N_tar=data["N_tar_grid"],
        N_tsr=data["N_tsr_grid"],
    )
    arrays = {
        "C_bits": data["C_bits"],
        "nH": data["nH"],
        "DR_out": data["DR_out"],
        "DR_p": data["DR_p"],
        "c50": data["c50"],
    }
    done_mask = data["done_mask"].astype(bool)
    iters = data["iters"].astype(int)
    cursor = int(data["cursor"])
    complete = bool(data["complete"])
    meta = {}
    if "meta" in data:
        m = data["meta"]
        if isinstance(m, np.ndarray):
            try:
                meta = m.item()
            except Exception:
                meta = {}
        elif isinstance(m, dict):
            meta = m
    return dict(
        grids=grids,
        arrays=arrays,
        done_mask=done_mask,
        iters=iters,
        cursor=cursor,
        complete=complete,
        meta=meta,
    )


def _results_from_npz_arrays(grids: dict,
                             arrays: dict,
                             meta: dict,
                             done_mask: np.ndarray,
                             cursor: int,
                             complete: bool) -> dict:

    res = {k: np.asarray(v) for k, v in arrays.items()}
    res["grids"] = {k: np.asarray(v) for k, v in grids.items()}
    res["meta"] = meta
    res["done_mask"] = np.asarray(done_mask)
    res["cursor"] = int(cursor)
    res["complete"] = bool(complete)
    return res


def run_resumable_sweep_npz(npz_path: str | Path,
                            grids: dict,
                            *,
                            bio_list: list[dict] | None = None,
                            anchor: dict | None = None,
                            time_budget_hours: float = 10.0,
                            checkpoint_every_min: float = 30.0,
                            eta_every_min: float = 10.0,
                            progress: bool = True) -> dict:

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if npz_path.exists():
        loaded = _load_npz(npz_path)
        grids_saved = loaded["grids"]
        for k in INDEP_VARS:
            if grids_saved[k].shape != np.asarray(grids[k]).shape:
                raise ValueError(
                    f"Grid mismatch for {k} between NPZ and requested grids."
                )
        grids = grids_saved
        arrays = loaded["arrays"]
        done_mask = loaded["done_mask"]
        iters = loaded["iters"]
        cursor = loaded["cursor"]
        complete = loaded["complete"]
        meta = loaded["meta"]
        if bio_list is not None:
            meta.setdefault("bio_dots", bio_list)
        if anchor is not None:
            meta.setdefault("anchor", anchor)
        if complete:
            if progress:
                print(f"[NPZ] Found complete sweep at {npz_path.name}; skipping recompute.")
            return _results_from_npz_arrays(
                grids, arrays, meta, done_mask, cursor, complete
            )
        if progress:
            coverage = 100.0 * done_mask.sum() / done_mask.size
            print(
                f"[NPZ] Resuming sweep from NPZ: "
                f"cursor={cursor}, coverage={coverage:5.2f}%"
            )
    else:
        shape = (
            len(grids["L0"]), len(grids["KdI1"]), len(grids["KdA1"]),
            len(grids["KdI2"]), len(grids["KdA2"]),
            len(grids["N_tar"]), len(grids["N_tsr"])
        )
        arrays = {
            "C_bits": np.full(shape, np.nan, dtype=np.float32),
            "nH":     np.full(shape, np.nan, dtype=np.float32),
            "DR_out": np.full(shape, np.nan, dtype=np.float32),
            "DR_p":   np.full(shape, np.nan, dtype=np.float32),
            "c50":    np.full(shape, np.nan, dtype=np.float32),
        }
        done_mask = np.zeros(shape, dtype=bool)
        iters = np.zeros(shape, dtype=np.int32)
        cursor = 0
        complete = False
        meta = {}
        if bio_list is not None:
            meta["bio_dots"] = bio_list
        if anchor is not None:
            meta["anchor"] = anchor
        if progress:
            print(f"[NPZ] Starting new sweep; saving to {npz_path.name}")

        _save_npz(npz_path, grids, arrays, done_mask, iters, cursor, complete, meta)

    shape = done_mask.shape
    total = int(np.prod(shape))
    done0 = int(np.count_nonzero(done_mask))
    cov0 = 100.0 * (done0 / total)
    if progress:
        print(f"[NPZ] Initial coverage: {cov0:5.2f}%  ({done0}/{total})")

    L0_grid    = np.asarray(grids["L0"],   float)
    KdI1_grid  = np.asarray(grids["KdI1"], float)
    KdA1_grid  = np.asarray(grids["KdA1"], float)
    KdI2_grid  = np.asarray(grids["KdI2"], float)
    KdA2_grid  = np.asarray(grids["KdA2"], float)
    N_tar_grid = np.asarray(grids["N_tar"], float)
    N_tsr_grid = np.asarray(grids["N_tsr"], float)

    start_time = time.time()
    deadline   = float('inf')
    next_ckpt  = start_time + checkpoint_every_min * 60.0
    next_eta   = start_time + eta_every_min * 60.0

    i = int(cursor)
    warm_state: BAWarmState | None = None
    while i < total:
        now = time.time()
        if now >= deadline:
            if progress:
                print("[NPZ] Time budget hit; stopping at a point boundary.")
            break

        if now >= next_ckpt:
            _save_npz(npz_path, grids, arrays, done_mask, iters, i, False, meta)
            if progress:
                done_now = int(np.count_nonzero(done_mask))
                coverage = 100.0 * (done_now / total)
                elapsed = now - start_time
                elapsed_txt = f"{elapsed:7.1f}s" + ProgressPrinter._hm_text(elapsed)
                print(
                    f"[NPZ] checkpoint @ i={i}/{total}  "
                    f"elapsed={elapsed_txt}  coverage={coverage:5.2f}%"
                )
            next_ckpt = now + checkpoint_every_min * 60.0

        if now >= next_eta and progress:
            done_now = int(np.count_nonzero(done_mask))
            coverage = 100.0 * (done_now / total)
            elapsed = now - start_time
            rate = (done_now - done0) / elapsed if elapsed > 0 else 0.0
            remain = (total - done_now) / rate if rate > 0 else float("nan")
            elapsed_txt = f"{elapsed:7.1f}s" + ProgressPrinter._hm_text(elapsed)
            eta_txt = (f"{remain:7.1f}s" + ProgressPrinter._hm_text(remain)
                       if np.isfinite(remain) else "     n/a")
            print(
                f"[NPZ] ETA update: i={i}/{total}  "
                f"elapsed={elapsed_txt}  ETA={eta_txt}  "
                f"coverage={coverage:5.2f}%"
            )
            next_eta = now + eta_every_min * 60.0

        a, b, c, d, e, f, g = np.unravel_index(i, shape)
        if done_mask[a, b, c, d, e, f, g]:
            i += 1
            continue

        L0   = float(L0_grid[a])
        KdI1 = float(KdI1_grid[b])
        KdA1 = float(KdA1_grid[c])
        KdI2 = float(KdI2_grid[d])
        KdA2 = float(KdA2_grid[e])
        Nt   = float(N_tar_grid[f])
        Ns   = float(N_tsr_grid[g])

        if (L0 <= 0) or (KdI1 <= 0) or (KdA1 <= 0) or (KdI2 <= 0) or (KdA2 <= 0):
            warm_state = None
            done_mask[a, b, c, d, e, f, g] = True
            i += 1
            continue

        ctx = {
            "L0": L0, "KdI1": KdI1, "KdA1": KdA1, "KdI2": KdI2, "KdA2": KdA2,
            "N_tar": Nt, "N_tsr": Ns
        }

        try:
            with WarnContext(ctx):
                res = metrics_at_params_auto_c(
                    L0=L0, KdI1=KdI1, KdA1=KdA1,
                    KdI2=KdI2, KdA2=KdA2,
                    N_tar=Nt, N_tsr=Ns,
                    mode="binary",
                    warm_state=warm_state,
                )

            for k in DEP_VARS:
                arrays[k][a, b, c, d, e, f, g] = np.float32(res.get(k, np.nan))
            iters[a, b, c, d, e, f, g] = int(res.get("iters", 0))

            try:
                c_prev = res.get("ba_c_vals", None)
                pX_prev = res.get("ba_pX", None)
                if c_prev is not None and pX_prev is not None:
                    c_prev = np.asarray(c_prev, float)
                    pX_prev = np.asarray(pX_prev, float)
                    if c_prev.ndim == 1 and pX_prev.ndim == 1 and c_prev.size == pX_prev.size and c_prev.size >= 2:
                        warm_state = BAWarmState(c_vals=c_prev, pX=pX_prev)
                    else:
                        warm_state = None
                else:
                    warm_state = None
            except Exception:
                warm_state = None

            done_mask[a, b, c, d, e, f, g] = True


        except Exception as ex:
            warm_state = None
            with WarnContext(ctx):
                _warn_once("bracket", f"NPZ sweep exception at flat index {i}: {type(ex).__name__}: {ex}")
            done_mask[a, b, c, d, e, f, g] = True

        i += 1

    cursor = i
    complete = bool(np.all(done_mask))
    _save_npz(npz_path, grids, arrays, done_mask, iters, cursor, complete, meta)

    now = time.time()
    elapsed = now - start_time
    done_now = int(np.count_nonzero(done_mask))
    coverage = 100.0 * (done_now / total)
    rate = (done_now - done0) / elapsed if elapsed > 0 else 0.0
    remain = (total - done_now) / rate if rate > 0 else float("nan")
    elapsed_txt = f"{elapsed:7.1f}s" + ProgressPrinter._hm_text(elapsed)
    eta_txt = (f"{remain:7.1f}s" + ProgressPrinter._hm_text(remain)
               if np.isfinite(remain) else "     n/a")

    if progress:
        print(
            f"[NPZ] saved -> {npz_path} | complete={complete} | "
            f"coverage={coverage:5.2f}% | elapsed={elapsed_txt} | "
            f"ETA {eta_txt}"
        )

    return _results_from_npz_arrays(grids, arrays, meta, done_mask, cursor, complete)


# ============================================================================
# TWENTY-STRAIN TABLE (shared across figures and tables)
# ----------------------------------------------------------------------------
# Same 20 operating points as in the manuscript's Table I. Four wild types
# (WT-K, WT-EW, WT-H, WT-C) plus sixteen lab mutants (10 Mello & Tu, 5 Keymer
# receptor-composition variants, 1 Clausznitzer CheB^D56E).
# ============================================================================
WINGREEN_KD = dict(KdI1=0.02, KdA1=0.5, KdI2=100.0, KdA2=1.0e6)
MELLO_KD    = dict(KdI1=0.0492, KdA1=0.1096, KdI2=0.0345, KdA2=0.1099)

WT_STRAINS_20 = [
    ("WT-K",  dict(L0=1.0, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
    ("WT-EW", dict(L0=2.0, N_tar=6.0,  N_tsr=12.0,  **WINGREEN_KD)),
    ("WT-H",  dict(L0=2.0, N_tar=6.0,  N_tsr=13.0,  **WINGREEN_KD)),
    ("WT-C",  dict(L0=1.9, N_tar=7.29, N_tsr=10.21, **WINGREEN_KD)),
]

OTHER_LAB_STRAINS = [
    ("cheR",      dict(L0=20.1,    N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
    ("Tar(EEEE)", dict(L0=4.54e-5, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
    ("Tar(QEEE)", dict(L0=3.06e-7, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
    ("Tar(QEQE)", dict(L0=1.52e-8, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
    ("Tar(QEQQ)", dict(L0=1.25e-9, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
    ("CheB",      dict(L0=1.0,     N_tar=7.29, N_tsr=10.21, **WINGREEN_KD)),
]

N_WT = len(WT_STRAINS_20)


def _mello_lab_strains(name_fmt: str = "Mello {label}") -> list:
    """Return the 10 Mello & Tu strains as (name, params) tuples with the
    caller's naming convention (some scripts label 'Mello 1', some 'Mello & Tu
    2005 strain 1'). Underlying parameters are identical across callers."""
    out = []
    for d in bio_dots_all10():
        p = dict(L0=d["L0"], N_tar=d["N_tar"], N_tsr=d["N_tsr"], **MELLO_KD)
        out.append((name_fmt.format(label=d["label"]), p))
    return out


def _all_strains_20(mello_name_fmt: str = "Mello {label}") -> list:
    """The full 20-strain list: 4 WT + 10 Mello + 6 other lab."""
    return WT_STRAINS_20 + _mello_lab_strains(mello_name_fmt) + OTHER_LAB_STRAINS


# ============================================================================
# STRAIN TABLE  (from scripts/compute_pdf_metrics.py)
# Reproduces: Table III (C_max ceiling saturation), Table V (DR fractions),
# Table VI (|n_eff| fractions), gradient LaTeX tables that fed Table IV.
# Also prints Z-channel ceiling check.
# ============================================================================
def compute_strain_table(npz_mello: Path = NPZ_MELLO_DEFAULT,
                         npz_keymer: Path = NPZ_KEYMER_DEFAULT):
    """Body of scripts/compute_pdf_metrics.py, adapted to use the
    module-level bio_dots_all10 and metrics_at_params_auto_c directly."""

    WT_POINTS = [
        ("Keymer 2006 WT",             dict(L0=1.0,  N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        ("Endres & Wingreen 2006 WT",  dict(L0=2.0,  N_tar=6.0,  N_tsr=12.0,  **WINGREEN_KD)),
        ("Hansen 2008 WT",             dict(L0=2.0,  N_tar=6.0,  N_tsr=13.0,  **WINGREEN_KD)),
        ("Clausznitzer 2010 WT1",      dict(L0=1.9,  N_tar=7.29, N_tsr=10.21, **WINGREEN_KD)),
    ]

    _MELLO_DOTS = bio_dots_all10()
    LAB_POINTS = [
        (f"Mello & Tu 2005 strain {d['label']}",
         dict(L0=d["L0"], N_tar=d["N_tar"], N_tsr=d["N_tsr"], **MELLO_KD))
        for d in _MELLO_DOTS
    ] + [
        ("Keymer 2006 cheR",              dict(L0=20.1,    N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        ("Keymer 2006 Tar(EEEE)",         dict(L0=4.54e-5, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        ("Keymer 2006 Tar(QEEE)",         dict(L0=3.06e-7, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        ("Keymer 2006 Tar(QEQE)",         dict(L0=1.52e-8, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        ("Keymer 2006 Tar(QEQQ)",         dict(L0=1.25e-9, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        ("Clausznitzer 2010 CheB mutant", dict(L0=1.0,     N_tar=7.29, N_tsr=10.21, **WINGREEN_KD)),
    ]

    ALL_POINTS = [(name, p, "WT")  for name, p in WT_POINTS] + \
                 [(name, p, "lab") for name, p in LAB_POINTS]

    def _load_sweep(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        done = data["done_mask"]
        Cm  = float(data["C_bits"][np.isfinite(data["C_bits"]) & done].max())
        DRm = float(data["DR_out"][np.isfinite(data["DR_out"]) & done].max())
        nHm = float(data["nH"][np.isfinite(data["nH"]) & done].max())
        bounds = {
            "L0":    (float(data["L0_grid"].min()),    float(data["L0_grid"].max())),
            "KdI1":  (float(data["KdI1_grid"].min()),  float(data["KdI1_grid"].max())),
            "KdA1":  (float(data["KdA1_grid"].min()),  float(data["KdA1_grid"].max())),
            "KdI2":  (float(data["KdI2_grid"].min()),  float(data["KdI2_grid"].max())),
            "KdA2":  (float(data["KdA2_grid"].min()),  float(data["KdA2_grid"].max())),
            "N_tar": (float(data["N_tar_grid"].min()), float(data["N_tar_grid"].max())),
            "N_tsr": (float(data["N_tsr_grid"].min()), float(data["N_tsr_grid"].max())),
        }
        return Cm, DRm, nHm, bounds

    Cmax_W, DRmax_W, nHmax_W, bounds_W = _load_sweep(npz_keymer)
    Cmax_M, DRmax_M, nHmax_M, bounds_M = _load_sweep(npz_mello)
    cfg = TableBStepConfig()

    print(f"Wingreen-framework sweep: C_max={Cmax_W:.4f}, DR_max={DRmax_W:.4f}, |n_eff|_max={nHmax_W:.3f}")
    print(f"Mello/Tu-framework sweep: C_max={Cmax_M:.4f}, DR_max={DRmax_M:.4f}, |n_eff|_max={nHmax_M:.3f}")
    print()

    def framework_for(params):
        if abs(params["KdI1"] - 0.02) < 1e-6:
            return Cmax_W, DRmax_W, nHmax_W, bounds_W, "Wingreen"
        return Cmax_M, DRmax_M, nHmax_M, bounds_M, "Mello/Tu"

    def C_max_z_channel(A_star):
        if A_star <= 0 or A_star >= 1:
            return 0.0
        return float(np.log2(1.0 + A_star * (1.0 - A_star)**((1.0 - A_star) / A_star)))

    print("=" * 130)
    print(f"{'Point':<38} {'C':>8} {'C/Cmax':>8} {'DR':>8} {'DR/DRm':>8} {'|n_eff|':>9} "
          f"{'||grad C||':>11} {'||grad DR||':>12} {'||grad nH||':>12} {'framework':>10}")
    print("-" * 130)

    results = []
    for name, p, kind in ALL_POINTS:
        out = metrics_at_params_auto_c(**p)
        Cm, DRm, _, bnds, fw = framework_for(p)
        gC  = tableb_grad_norm_fixed_steps(p, "C_bits",  steps=cfg, bounds=bnds)
        gDR = tableb_grad_norm_fixed_steps(p, "DR_out",  steps=cfg, bounds=bnds)
        gnH = tableb_grad_norm_fixed_steps(p, "nH",      steps=cfg, bounds=bnds)
        A_star = 1.0 / (1.0 + p["L0"])
        Cmax_A = C_max_z_channel(A_star)
        row = dict(
            name=name, kind=kind,
            A_star=A_star,
            C=out["C_bits"], DR=out["DR_out"], nH=out["nH"],
            Cmax_A=Cmax_A,
            C_over_Cmax_A=(out["C_bits"]/Cmax_A) if Cmax_A > 0 else 0.0,
            C_frac=out["C_bits"]/Cm, DR_frac=out["DR_out"]/DRm,
            gC=gC, gDR=gDR, gnH=gnH,
            framework=fw,
        )
        results.append(row)
        print(f"{name:<38} {row['C']:>8.4f} {row['C_frac']:>8.4f} "
              f"{row['DR']:>8.4f} {row['DR_frac']:>8.4f} "
              f"{row['nH']:>9.3f} "
              f"{row['gC']:>11.4f} {row['gDR']:>12.4f} {row['gnH']:>12.4f} "
              f"{fw:>10}")

    print()
    print("=" * 100)
    print("Z-channel ceiling check: every C should be <= C_max(A*)")
    print("=" * 100)
    print(f"{'Point':<38} {'A*':>8} {'C':>8} {'C_max(A*)':>11} {'C/C_max':>10} {'gap':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<38} {r['A_star']:>8.4f} {r['C']:>8.4f} "
              f"{r['Cmax_A']:>11.4f} {r['C_over_Cmax_A']:>10.4f} "
              f"{r['Cmax_A']-r['C']:>8.4f}")

    violations = [r for r in results if r['C'] > r['Cmax_A'] + 1e-6]
    if violations:
        print()
        print("!!! VIOLATIONS of C <= C_max(A*) !!!")
        for r in violations:
            print(f"  {r['name']}: C={r['C']:.4f} > C_max={r['Cmax_A']:.4f}")
    else:
        print()
        print(f"OK: all {len(results)} points satisfy C <= C_max(A*) (gap <= {max(r['Cmax_A']-r['C'] for r in results):.4f}).")

    print()
    print("=" * 100)
    print("LaTeX GRADIENT TABLES below")
    print("=" * 100)
    print()

    print(r"% --- WT GRADIENT TABLE ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"WT data point & $\|\nabla C\|_2$ & $\|\nabla \mathrm{DR}\|_2$ & $\|\nabla |n_{\text{eff}}|\|_2$ \\")
    print(r"\midrule")
    for r in results:
        if r["kind"] == "WT":
            print(f"{r['name']} & {r['gC']:.4f} & {r['gDR']:.4f} & {r['gnH']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()
    print(r"% --- LAB-GROWN GRADIENT TABLE ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{lccccl}")
    print(r"\toprule")
    print(r"lab-grown data point & $\|\nabla C\|_2$ & $\|\nabla \mathrm{DR}\|_2$ & $\|\nabla |n_{\text{eff}}|\|_2$ & framework \\")
    print(r"\midrule")
    for r in results:
        if r["kind"] == "lab":
            print(f"{r['name']} & {r['gC']:.4f} & {r['gDR']:.4f} & {r['gnH']:.4f} & {r['framework']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ============================================================================
# CORRELATION MATRIX + GLOBAL MAXIMA  (from scripts/recompute_table_R_and_maxima.py)
# Feeds Table II (C-nH-DR correlation matrix) and the C_max/DR_max/n_eff_max
# global maxima reported in the paper.
# ============================================================================
def compute_correlation_and_maxima(npz_mello: Path = NPZ_MELLO_DEFAULT,
                                   npz_keymer: Path = NPZ_KEYMER_DEFAULT):

    def load_metrics(path):
        z = np.load(path, allow_pickle=True)
        mask = z["done_mask"].astype(bool)
        C   = z["C_bits"][mask].astype(np.float64)
        nH  = z["nH"][mask].astype(np.float64)
        DR  = z["DR_out"][mask].astype(np.float64)
        return C, nH, DR, int(mask.sum()), tuple(z["C_bits"].shape)

    print("Loading Mello/Tu K_d region sweep...")
    C_m, nH_m, DR_m, n_m, shape_m = load_metrics(npz_mello)
    print(f"  grid shape {shape_m}, valid points: {n_m:,}")

    print("Loading Wingreen K_d region sweep...")
    C_k, nH_k, DR_k, n_k, shape_k = load_metrics(npz_keymer)
    print(f"  grid shape {shape_k}, valid points: {n_k:,}")

    print(f"\nUnion total: {n_m + n_k:,} valid grid points\n")

    C   = np.concatenate([C_m,  C_k])
    DR  = np.concatenate([DR_m, DR_k])
    nH  = np.abs(np.concatenate([nH_m, nH_k]))

    finite = np.isfinite(C) & np.isfinite(DR) & np.isfinite(nH)
    print(f"Finite triple: {finite.sum():,} / {C.size:,}")

    C, DR, nH = C[finite], DR[finite], nH[finite]

    R = np.corrcoef(np.vstack([C, nH, DR]))

    print("\nCorrelation matrix R (C, |n_eff|, DR), over the union:")
    labels = ["C", "|n_eff|", "DR"]
    print(f"               {'C':>8} {'|n_eff|':>10} {'DR':>8}")
    for i, lab in enumerate(labels):
        print(f"  {lab:>10}  " + " ".join(f"{R[i,j]:>8.3f}" for j in range(3)))

    print("\nGlobal maxima over the union:")
    print(f"  C_max       = {C.max():.4f} bits")
    print(f"  DR_max      = {DR.max():.4f}")
    print(f"  |n_eff|_max = {nH.max():.4f}")

    print("\nFor reference, per-region maxima (NOT to be reported separately):")
    print(f"  Mello region:    C_max={C_m.max():.4f}, DR_max={DR_m.max():.4f}, |n_eff|_max={np.abs(nH_m).max():.4f}")
    print(f"  Wingreen region: C_max={C_k.max():.4f}, DR_max={DR_k.max():.4f}, |n_eff|_max={np.abs(nH_k).max():.4f}")


# ============================================================================
# GRADIENT-MAX NORMALIZATION  (from scripts/compute_max_gradient.py)
# Feeds Table IV: |grad metric|_strain / |grad metric|_max ratios per strain
# per metric (reviewer concern 4).
# ============================================================================
def compute_gradient_max_and_ratios(npz_mello: Path = NPZ_MELLO_DEFAULT,
                                    npz_keymer: Path = NPZ_KEYMER_DEFAULT):

    AXES_ORDER = ["L0", "KdI1", "KdA1", "KdI2", "KdA2", "N_tar", "N_tsr"]
    LOG_AXES   = {"L0", "KdI1", "KdA1", "KdI2", "KdA2"}

    def load(path):
        z = np.load(path, allow_pickle=True)
        grids = {a: np.asarray(z[f"{a}_grid"], float) for a in AXES_ORDER}
        C  = np.array(z["C_bits"], dtype=float)
        DR = np.array(z["DR_out"], dtype=float)
        nH = np.array(z["nH"],     dtype=float)
        return grids, C, DR, np.abs(nH)

    def grid_spacings(grids):
        return {
            a: np.log10(grids[a]) if a in LOG_AXES else grids[a]
            for a in AXES_ORDER
        }

    def grad_norm(arr, spacings):
        fixed_spacings = {}
        for a in AXES_ORDER:
            g = np.asarray(spacings[a], dtype=float)
            deltas = np.diff(g)
            typical = np.median(np.abs(deltas))
            tol = max(typical * 1e-6, 1e-12)
            tiny = deltas < tol
            if tiny.any():
                print(f"    [dedup] axis {a}: {int(tiny.sum())} sub-tolerance step(s) fixed")
                good = deltas[~tiny]
                replacement = float(np.median(good)) if good.size else 1.0
                deltas = np.where(tiny, replacement, deltas)
                g = np.concatenate([[g[0]], g[0] + np.cumsum(deltas)])
            fixed_spacings[a] = g
        grads = np.gradient(
            arr,
            *[fixed_spacings[a] for a in AXES_ORDER],
            edge_order=1
        )
        stack = np.stack(grads, axis=0)
        return np.sqrt(np.nansum(stack ** 2, axis=0))

    def analyze(name, path):
        grids, C, DR, nH = load(path)
        spacings = grid_spacings(grids)
        print(f"\n== {name} region ==")
        print(f"  Grid shape: {C.shape}")
        for m, label in [(C, "C"), (DR, "DR"), (nH, "|n_eff|")]:
            gn = grad_norm(m, spacings)
            gn_max = float(np.nanmax(gn))
            print(f"  max ||grad {label}||_2 = {gn_max:.3f}")
            yield label, gn_max, gn, grids

    print("Loading and computing gradient norms on both sweep regions...")
    results = {"C": {"max": 0.0}, "DR": {"max": 0.0}, "|n_eff|": {"max": 0.0}}

    for region, path in [("Mello/Tu", npz_mello), ("Wingreen", npz_keymer)]:
        for label, gn_max, gn, grids in analyze(region, path):
            results[label]["max"] = max(results[label]["max"], gn_max)
            results[label][f"{region}_grid"] = grids
            results[label][f"{region}_gn"] = gn

    print("\n== Global maxima over the union of both grids ==")
    for label in ["C", "DR", "|n_eff|"]:
        print(f"  max ||grad {label}||_2 = {results[label]['max']:.3f}")

    WT = [
        ("WT-K",  dict(L0=1.0, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD), "Wingreen"),
        ("WT-EW", dict(L0=2.0, N_tar=6.0,  N_tsr=12.0,  **WINGREEN_KD), "Wingreen"),
        ("WT-H",  dict(L0=2.0, N_tar=6.0,  N_tsr=13.0,  **WINGREEN_KD), "Wingreen"),
        ("WT-C",  dict(L0=1.9, N_tar=7.29, N_tsr=10.21, **WINGREEN_KD), "Wingreen"),
    ]

    _mello = bio_dots_all10()

    LAB = []
    for d in _mello:
        LAB.append((f"Mello {d['label']}",
                    dict(L0=d["L0"], N_tar=d["N_tar"], N_tsr=d["N_tsr"], **MELLO_KD),
                    "Mello/Tu"))
    LAB += [
        ("cheR",      dict(L0=20.1,    N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD), "Wingreen"),
        ("Tar(EEEE)", dict(L0=4.54e-5, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD), "Wingreen"),
        ("Tar(QEEE)", dict(L0=3.06e-7, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD), "Wingreen"),
        ("Tar(QEQE)", dict(L0=1.52e-8, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD), "Wingreen"),
        ("Tar(QEQQ)", dict(L0=1.25e-9, N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD), "Wingreen"),
        ("CheB",      dict(L0=1.0,     N_tar=7.29, N_tsr=10.21, **WINGREEN_KD), "Wingreen"),
    ]
    STRAINS = WT + LAB

    def nearest_idx(grid, val):
        return int(np.argmin(np.abs(grid - val)))

    print("\n== Strain-point gradient norms (nearest grid point) and ratios ==")
    print(f"{'strain':14} {'|grad C|':>10} {'ratio':>7} {'|grad DR|':>11} {'ratio':>7} {'|grad neff|':>13} {'ratio':>7}")
    for name, p, region in STRAINS:
        grids = results["C"][f"{region}_grid"]
        idx = tuple(nearest_idx(grids[a], p[a] if a in {"N_tar", "N_tsr", "L0"}
                                              else p[a]) for a in AXES_ORDER)
        row_C   = float(results["C"][f"{region}_gn"][idx])
        row_DR  = float(results["DR"][f"{region}_gn"][idx])
        row_nH  = float(results["|n_eff|"][f"{region}_gn"][idx])
        rC = row_C / results["C"]["max"]
        rDR = row_DR / results["DR"]["max"]
        rnH = row_nH / results["|n_eff|"]["max"]
        print(f"{name:14} {row_C:10.3f} {rC:7.3f} {row_DR:11.3f} {rDR:7.3f} {row_nH:13.3f} {rnH:7.3f}")


# ============================================================================
# MAIN-TEXT FIGURES 3 AND 4  (from scripts/regenerate_main_figures.py)
# Produces: all_strains_pactive_vs_c.png, all_strains_OptimalInput_vs_c.png,
# combined_pY_grouped.png, C_vs_KA1.png, DR_NtarNtsr_two_panels.png,
# neff_NtarNtsr_two_panels.png.
# ============================================================================
def regenerate_fig3_fig4(output_dir: Path,
                         npz_mello: Path = NPZ_MELLO_DEFAULT,
                         npz_keymer: Path = NPZ_KEYMER_DEFAULT):

    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    WT = list(WT_STRAINS_20)
    _mello = bio_dots_all10()
    LAB = []
    for d in _mello:
        LAB.append((f"Mello {d['label']}",
                    dict(L0=d["L0"], N_tar=d["N_tar"], N_tsr=d["N_tsr"], **MELLO_KD)))
    LAB += list(OTHER_LAB_STRAINS)

    STRAINS = WT + LAB
    N_WT_local  = len(WT)
    N_LAB_local = len(LAB)
    assert N_WT_local == 4 and N_LAB_local == 16

    WT_COLORS  = [cm.tab10(i) for i in range(N_WT_local)]
    LAB_COLORS = [cm.viridis(x) for x in np.linspace(0.05, 0.92, N_LAB_local)]

    def style_for(idx):
        if idx < N_WT_local:
            return WT_COLORS[idx], 2.5, 1.0, 3
        return LAB_COLORS[idx - N_WT_local], 1.1, 0.55, 1

    WT_MARKERS  = ['o', 's', '^', 'D']
    LAB_MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '*', '<', '>',
                   'h', 'd', 'H', 'p', '8', 'o']

    def marker_for(idx):
        if idx < N_WT_local:
            return WT_MARKERS[idx]
        return LAB_MARKERS[(idx - N_WT_local) % len(LAB_MARKERS)]

    print("Computing Blahut-Arimoto at each of the 20 strain operating points...")
    results = []
    for name, p in STRAINS:
        r = ba_at_params(p["L0"], p["KdI1"], p["KdA1"],
                         p["KdI2"], p["KdA2"],
                         p["N_tar"], p["N_tsr"])
        results.append({"name": name, "params": p,
                        "c_vals": r["c_vals"], "pa": r["pa"],
                        "pX": r["pX"], "pY": r["pY"], "C_bits": r["C_bits"]})
        print(f"  {name:14s}  C = {r['C_bits']:.3f} bits")

    def fig_pa_vs_c():
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        order = list(range(N_WT_local, N_WT_local + N_LAB_local)) + list(range(N_WT_local))
        for idx in order:
            r = results[idx]
            c, lw, a, z = style_for(idx)
            ax.plot(r["c_vals"], r["pa"], color=c, lw=lw, alpha=a, zorder=z,
                    label=r["name"])
        ax.set_xscale("log")
        ax.set_xlabel(r"Ligand Concentration $c$ (mM)", fontsize=14)
        ax.set_ylabel(r"$p(c)$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="both", linestyle=":", alpha=0.3)
        leg = ax.legend(ncol=2, fontsize=9, loc="center left",
                        bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.95,
                        title="Strain", title_fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "all_strains_pactive_vs_c.png", dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {OUT / 'all_strains_pactive_vs_c.png'}")

    def fig_pX_vs_c():
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        order = list(range(N_WT_local, N_WT_local + N_LAB_local)) + list(range(N_WT_local))
        for idx in order:
            r = results[idx]
            c, lw, a, z = style_for(idx)
            ax.plot(r["c_vals"], r["pX"], color=c, lw=lw, alpha=a, zorder=z,
                    label=r["name"])
        ax.set_xscale("log")
        ax.set_xlabel(r"Ligand Concentration $c$ (mM)", fontsize=14)
        ax.set_ylabel(r"$p_{\mathrm{in}}^*(c)$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.text(-0.12, 1.03, "(a)", transform=ax.transAxes,
                fontsize=14, va="bottom", ha="left")
        ax.grid(True, which="both", linestyle=":", alpha=0.3)
        ax.legend(ncol=2, fontsize=9, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.95,
                  title="Strain", title_fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "all_strains_OptimalInput_vs_c.png",
                    dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {OUT / 'all_strains_OptimalInput_vs_c.png'}")

    def fig_pY_grouped():
        names  = [r["name"] for r in results]
        p_act  = np.array([r["pY"][1] for r in results])
        p_inact = 1.0 - p_act

        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        x = np.arange(len(names))
        w = 0.38
        bar_colors = []
        for idx in range(len(results)):
            c, _, _, _ = style_for(idx)
            bar_colors.append(c)
        ax.bar(x - w/2, p_act,   width=w, label=r"$p^*(s{=}1)$ active",
               color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, p_inact, width=w, label=r"$p^*(s{=}0)$ inactive",
               color=bar_colors, edgecolor="black", linewidth=0.5,
               alpha=0.45, hatch="///")
        ax.axvline(N_WT_local - 0.5, color="black", linewidth=1.0, linestyle="--",
                   alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel(r"$p^*(s)$", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 1.14)
        ax.text(N_WT_local/2 - 0.5, 1.04, "wild type", ha="center", fontsize=11)
        ax.text(N_WT_local + N_LAB_local/2 - 0.5, 1.04, "lab mutants",
                ha="center", fontsize=11)
        ax.text(-0.055, 1.03, "(b)", transform=ax.transAxes,
                fontsize=14, va="bottom", ha="left")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "combined_pY_grouped.png", dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {OUT / 'combined_pY_grouped.png'}")

    def fig_C_vs_KA1():
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        KA1_grid = np.logspace(-3, 7, 24)
        order = list(range(N_WT_local, N_WT_local + N_LAB_local)) + list(range(N_WT_local))
        for idx in order:
            r = results[idx]
            p = r["params"]
            Cs = []
            for KA1 in KA1_grid:
                if KA1 <= p["KdI1"]:
                    Cs.append(np.nan)
                    continue
                res = ba_at_params(p["L0"], p["KdI1"], KA1,
                                   p["KdI2"], p["KdA2"],
                                   p["N_tar"], p["N_tsr"])
                Cs.append(res["C_bits"])
            c, lw, a, z = style_for(idx)
            ax.plot(KA1_grid, Cs, color=c, lw=lw, alpha=a, zorder=z, label=r["name"])
        ax.set_xscale("log")
        ax.set_xlabel(r"$K_{d}^{(A),1}$ (mM)", fontsize=12)
        ax.set_ylabel(r"Channel capacity $C$ (bits)", fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="both", linestyle=":", alpha=0.3)
        ax.legend(ncol=2, fontsize=7.5, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.95,
                  title="strain", title_fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "C_vs_KA1.png", dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {OUT / 'C_vs_KA1.png'}")

    def _load_region(path):
        return np.load(path, allow_pickle=True)

    def heatmap_two_panel(metric_key, cbar_label, fname, vmax_override=None,
                          take_abs=False):
        mello  = _load_region(npz_mello)
        keymer = _load_region(npz_keymer)

        def _slice_LN(z, abs_):
            arr = z[metric_key]
            if abs_:
                arr = np.abs(arr)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            return np.nanmedian(arr, axis=(1, 2, 3, 4, 5))

        Mslice = _slice_LN(mello,  take_abs)
        Kslice = _slice_LN(keymer, take_abs)

        Mx = mello["L0_grid"];   My = mello["N_tsr_grid"]
        Kx = keymer["L0_grid"];  Ky = keymer["N_tsr_grid"]

        vmax = vmax_override if vmax_override is not None else float(
            np.nanmax([np.nanmax(Mslice), np.nanmax(Kslice)]))
        vmin = 0.0

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0))

        def _draw(ax, slice_arr, xg, yg, dots, title):
            lxg = np.log10(xg)
            Xe = np.concatenate([[lxg[0] - (lxg[1] - lxg[0]) / 2],
                                  (lxg[:-1] + lxg[1:]) / 2,
                                  [lxg[-1] + (lxg[-1] - lxg[-2]) / 2]])
            Xe = 10.0 ** Xe
            Ye = np.concatenate([[yg[0] - (yg[1] - yg[0]) / 2],
                                  (yg[:-1] + yg[1:]) / 2,
                                  [yg[-1] + (yg[-1] - yg[-2]) / 2]])
            pcm = ax.pcolormesh(Xe, Ye, slice_arr.T, shading="flat",
                                cmap="viridis", vmin=vmin, vmax=vmax)
            for idx, (name, p) in dots:
                mk = marker_for(idx)
                color, _, _, _ = style_for(idx)
                ax.scatter([p["L0"]], [p["N_tsr"]],
                           s=140, marker=mk, c=[color],
                           edgecolors="white", linewidths=1.4, zorder=5,
                           label=name)
            ax.set_xscale("log")
            ax.set_xlabel(r"Allosteric constant $L_0$", fontsize=12)
            ax.set_ylabel(r"$N_2$ (Tsr count)", fontsize=12)
            ax.set_title(title, fontsize=11)
            return pcm

        mello_dots = [(N_WT_local + i, STRAINS[N_WT_local + i]) for i in range(10)]
        pcmL = _draw(axL, Mslice, Mx, My, mello_dots, "Mello/Tu K$_d$ region")
        axL.legend(ncol=2, fontsize=7, loc="upper right", framealpha=0.95)

        wing_dots = [(i, STRAINS[i]) for i in range(N_WT_local)]
        wing_dots += [(N_WT_local + 10 + i, STRAINS[N_WT_local + 10 + i]) for i in range(6)]
        pcmR = _draw(axR, Kslice, Kx, Ky, wing_dots, "Wingreen K$_d$ region")
        axR.legend(ncol=2, fontsize=7, loc="lower left", framealpha=0.95)

        cbar = fig.colorbar(pcmR, ax=[axL, axR], shrink=0.85, pad=0.02)
        cbar.set_label(cbar_label, fontsize=12)

        fig.suptitle("Two windows on the same 7D landscape "
                     r"(median over $K_d$ dimensions; shared color scale)",
                     fontsize=10, y=1.00)
        fig.savefig(OUT / fname, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {OUT / fname}")

    fig_pa_vs_c()
    fig_pX_vs_c()
    fig_pY_grouped()
    fig_C_vs_KA1()
    heatmap_two_panel("DR_out", r"Dynamic range $\mathrm{DR}$",
                       "DR_NtarNtsr_two_panels.png", vmax_override=1.0)
    heatmap_two_panel("nH",     r"Effective Hill coefficient $|n_{\mathrm{eff}}|$",
                       "neff_NtarNtsr_two_panels.png", take_abs=True)

    print(f"\nAll six figures saved to: {OUT}")


# ============================================================================
# MAIN-TEXT FIGURE 5  (from scripts/regenerate_new_figs.py fig5())
# Produces: cmax_curve.png (C vs baseline activity p_0 with Z-channel ceiling)
# ============================================================================
def regenerate_fig5(output_dir: Path):
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    WT = list(WT_STRAINS_20)
    _mello = bio_dots_all10()
    LAB = [(f"Mello {d['label']}",
            dict(L0=d["L0"], N_tar=d["N_tar"], N_tsr=d["N_tsr"], **MELLO_KD))
           for d in _mello] + list(OTHER_LAB_STRAINS)
    STRAINS = WT + LAB
    N_WT_local, N_LAB_local = len(WT), len(LAB)

    WT_COLORS  = [cm.tab10(i) for i in range(N_WT_local)]
    LAB_COLORS = [cm.viridis(x) for x in np.linspace(0.05, 0.92, N_LAB_local)]

    def compute_all():
        rows = []
        for i, (name, p) in enumerate(STRAINS):
            r = ba_at_params(p["L0"], p["KdI1"], p["KdA1"],
                             p["KdI2"], p["KdA2"],
                             p["N_tar"], p["N_tsr"])
            m = metrics_at_params_auto_c(
                L0=p["L0"], KdI1=p["KdI1"], KdA1=p["KdA1"],
                KdI2=p["KdI2"], KdA2=p["KdA2"],
                N_tar=p["N_tar"], N_tsr=p["N_tsr"])
            L0 = p["L0"]
            Astar = 1.0 / (1.0 + L0)
            log_prod = (p["N_tar"] * (np.log10(p["KdA1"]) - np.log10(p["KdI1"]))
                        + p["N_tsr"] * (np.log10(p["KdA2"]) - np.log10(p["KdI2"])))
            log_denom = np.log10(1.0 + L0 * 10**log_prod) if log_prod < 300 else np.log10(L0) + log_prod
            log_p_inf = -log_denom
            rows.append(dict(
                idx=i, name=name,
                Astar=Astar,
                C=r["C_bits"],
                neff=abs(m.get("nH", np.nan)),
                log_p_inf=log_p_inf,
                N_tot=p["N_tar"] + p["N_tsr"],
                N_tar=p["N_tar"], N_tsr=p["N_tsr"],
            ))
        return rows

    print("Computing metrics at each strain...")
    data = compute_all()
    for r in data:
        r["f_Tsr"] = r["N_tsr"] / r["N_tot"] if r["N_tot"] > 0 else 0.5
        print(f"  {r['name']:12s}  A*={r['Astar']:.3f}  C={r['C']:.3f}  |n_eff|={r['neff']:.2f}  log10(p_inf)={r['log_p_inf']:6.1f}  N_tot={r['N_tot']:.1f}  f_Tsr={r['f_Tsr']:.2f}")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    A_grid = np.linspace(0.001, 0.999, 400)
    Cmax = np.log2(1.0 + A_grid * (1.0 - A_grid) ** ((1.0 - A_grid) / A_grid))
    ax.plot(A_grid, Cmax, color="black", lw=2.4, zorder=1)
    label_A = 0.50
    label_C = np.log2(1.0 + label_A * (1.0 - label_A) ** ((1.0 - label_A) / label_A))
    ax.text(label_A, label_C + 0.12, r"$C_{\max}(p_0)$",
            fontsize=17, ha="center", va="bottom")

    handles = []
    for r in data:
        i = r["idx"]
        color = WT_COLORS[i] if i < N_WT_local else LAB_COLORS[i - N_WT_local]
        alpha = 1.0 if i < N_WT_local else 0.85
        h = ax.scatter([r["Astar"]], [r["C"]], s=110, marker="o",
                       facecolor=color, edgecolor="black", linewidths=0.7,
                       alpha=alpha, zorder=3, label=r["name"])
        handles.append(h)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel(r"Baseline Activity $p_0$", fontsize=15)
    ax.set_ylabel(r"Channel Capacity $C$ (bits)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    ax.legend(handles=handles, ncol=2, fontsize=10, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.95,
              title="Strain", title_fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT / "cmax_curve.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT / 'cmax_curve.png'}")


# ============================================================================
# SI FULL-PANEL HEATMAPS (Figs S1-S60)  (from scripts/generate_new_strain_heatmaps.py)
# Writes 30 strain heatmap PDFs (10 strains x 3 metrics) for strains 11-20.
# ============================================================================
def regenerate_si_heatmaps(overleaf_dest: Path,
                           staging_dir: Path,
                           npz_keymer: Path = NPZ_KEYMER_DEFAULT):

    NEW_STRAINS = [
        (11, "WT-K",      dict(L0=1.0,      N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        (12, "WT-EW",     dict(L0=2.0,      N_tar=6.0,  N_tsr=12.0,  **WINGREEN_KD)),
        (13, "WT-H",      dict(L0=2.0,      N_tar=6.0,  N_tsr=13.0,  **WINGREEN_KD)),
        (14, "WT-C",      dict(L0=1.9,      N_tar=7.29, N_tsr=10.21, **WINGREEN_KD)),
        (15, "cheR",      dict(L0=20.1,     N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        (16, "Tar(EEEE)", dict(L0=4.54e-5,  N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        (17, "Tar(QEEE)", dict(L0=3.06e-7,  N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        (18, "Tar(QEQE)", dict(L0=1.52e-8,  N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        (19, "Tar(QEQQ)", dict(L0=1.25e-9,  N_tar=5.0,  N_tsr=10.0,  **WINGREEN_KD)),
        (20, "CheB",      dict(L0=1.0,      N_tar=7.29, N_tsr=10.21, **WINGREEN_KD)),
    ]

    print(f"[LOAD] {npz_keymer}")
    results = load_results_from_npz(npz_keymer)

    bio_list = [{"label": str(idx), "name": name, **params}
                for idx, name, params in NEW_STRAINS]

    STAGING = Path(staging_dir)
    STAGING.mkdir(parents=True, exist_ok=True)
    export_full_panel_pages_per_strain(
        results,
        bio_list,
        STAGING,
        fmt="pdf",
        style=FULL_PANEL_STYLE,
        norm_mode="percentile",
        low_pct=2.0,
        high_pct=98.0,
        vmin=None,
        vmax=None,
    )

    src = STAGING / "plots" / "full_panels"
    OVERLEAF_DEST = Path(overleaf_dest)
    OVERLEAF_DEST.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(src.glob("strain*_full.pdf")):
        dst = OVERLEAF_DEST / pdf.name
        shutil.copy2(pdf, dst)
        print(f"[COPY] {dst}")

    print("[DONE] wrote 30 new strain heatmap panels")


# ============================================================================
# 7D SWEEP LAUNCHERS (from scripts/run_overnight_keymer_sweep.py and the old
# main() in 7D_Sweep_Code.py). SLOW: multi-hour, resumable. Only run to
# reproduce the NPZ data files themselves.
# ============================================================================
def run_mello_sweep(npz_path: Path = NPZ_MELLO_DEFAULT,
                    time_budget_hours: float = 12.0):
    """Reproduce the Mello/Tu K_d region sweep (7D_Sweep_Results.npz).
    Anchored at Mello & Tu strain 2, grid expanded to cover all 10 Mello
    strains. Same parameters as the submitted paper's sweep."""

    bio_list = bio_dots_all10()

    anchor = next(
        (d for d in bio_list
         if str(d.get("name", "")).lower() == "strain2"
         or str(d.get("label", "")) == "2"),
        bio_list[0]
    )

    grids = build_grids_pilot(
        anchor,
        points_log=6,
        points_N=6,
        L0_span_dec=2.0,
        K_span_dec=2.0
    )
    grids = expand_grids_to_cover_biodots(grids, bio_list,
                                          L0_pad=2.0, N_pad=2)

    print()
    print("=" * 72)
    print("Mello/Tu K_d region 7D sweep")
    print("=" * 72)
    shape = tuple(len(grids[k]) for k in INDEP_VARS)
    total = int(np.prod(shape))
    print(f"Output:  {npz_path}")
    print(f"Anchor:  strain 2 (L0={anchor['L0']:.3g})")
    print(f"Grid shape: {shape}  (= {total:,} points)")
    print(f"Time budget: {time_budget_hours}h")
    print("=" * 72)

    results = run_resumable_sweep_npz(
        npz_path=npz_path,
        grids=grids,
        bio_list=bio_list,
        anchor=anchor,
        time_budget_hours=time_budget_hours,
        checkpoint_every_min=30,
        eta_every_min=10,
        progress=True
    )

    print()
    print("=" * 72)
    print("Sweep complete (or hit time budget). Results saved to:")
    print(f"  {npz_path}")
    print("=" * 72)


def run_keymer_sweep(npz_path: Path = NPZ_KEYMER_DEFAULT,
                     time_budget_hours: float = 11.0):
    """Reproduce the Wingreen K_d region sweep (7D_Sweep_Results_keymer.npz).
    Anchored at Keymer WT with the six Keymer receptor-composition strains
    as bio dots."""

    KDI1 = 0.02
    KDA1 = 0.5
    KDI2 = 100.0
    KDA2 = 1.0e6

    KEYMER_STRAINS = [
        ("WT",         0.0,  0.0),
        ("cheR",       0.2,  0.2),
        ("Tar_EEEE",   1.0, -1.5),
        ("Tar_QEEE",   0.0, -1.5),
        ("Tar_QEQE", -0.6, -1.5),
        ("Tar_QEQQ", -1.1, -1.5),
    ]

    N_TAR_NOMINAL = 5.0
    N_TSR_NOMINAL = 10.0

    def keymer_bio_dots():
        dots = []
        for name, eps_a, eps_s in KEYMER_STRAINS:
            L0 = math.exp(N_TAR_NOMINAL * eps_a + N_TSR_NOMINAL * eps_s)
            dots.append({
                "name":  name,
                "label": name,
                "L0":    L0,
                "KdI1":  KDI1,
                "KdA1":  KDA1,
                "KdI2":  KDI2,
                "KdA2":  KDA2,
                "N_tar": N_TAR_NOMINAL,
                "N_tsr": N_TSR_NOMINAL,
            })
        return dots

    ANCHOR = {
        "L0":    1.0,
        "KdI1":  KDI1,
        "KdA1":  KDA1,
        "KdI2":  KDI2,
        "KdA2":  KDA2,
        "N_tar": N_TAR_NOMINAL,
        "N_tsr": N_TSR_NOMINAL,
    }

    bio_dots = keymer_bio_dots()

    print()
    print("=" * 72)
    print("Wingreen K_d region 7D sweep (Keymer framework)")
    print("=" * 72)
    print(f"Output:  {npz_path}")
    print(f"Anchor:  L0={ANCHOR['L0']}, KdI1={ANCHOR['KdI1']}, "
          f"KdA1={ANCHOR['KdA1']}, KdI2={ANCHOR['KdI2']}, "
          f"KdA2={ANCHOR['KdA2']}, N_tar={ANCHOR['N_tar']}, "
          f"N_tsr={ANCHOR['N_tsr']}")
    print(f"Bio dots: {len(bio_dots)} Keymer strains")
    for d in bio_dots:
        print(f"   {d['name']:<12}  L0={d['L0']:.3e}")
    print("=" * 72)

    grids = build_grids_pilot(
        bio=ANCHOR,
        points_log=7,
        points_N=9,
        L0_span_dec=3.0,
        K_span_dec=3.0,
    )

    grids = expand_grids_to_cover_biodots(
        grids, bio_dots, L0_pad=2.0, N_pad=2
    )

    grids["N_tar"] = np.linspace(0.0, 20.0, 10)
    grids["N_tsr"] = np.linspace(0.0, 25.0, 10)

    shape = tuple(len(grids[k]) for k in INDEP_VARS)
    total = int(np.prod(shape))
    print(f"Final grid shape: {shape}  (= {total:,} points)")
    print()

    results = run_resumable_sweep_npz(
        npz_path,
        grids,
        bio_list=bio_dots,
        anchor=ANCHOR,
        time_budget_hours=time_budget_hours,
        checkpoint_every_min=30.0,
        eta_every_min=10.0,
        progress=True,
    )

    print()
    print("=" * 72)
    print("Sweep complete (or hit time budget). Results saved to:")
    print(f"  {npz_path}")
    print("=" * 72)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce figures and tables for Cui & Marzen (2026).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Reproduction targets (fast, need only the two NPZ files)
    p.add_argument("--all", action="store_true",
                   help="Regenerate every main-text figure and print every "
                        "stdout table (fast; requires both NPZ files).")
    p.add_argument("--fig3-4", action="store_true", dest="fig3_4",
                   help="Regenerate all_strains_pactive_vs_c, "
                        "all_strains_OptimalInput_vs_c, combined_pY_grouped, "
                        "C_vs_KA1, DR_NtarNtsr_two_panels, "
                        "neff_NtarNtsr_two_panels.")
    p.add_argument("--fig5", action="store_true",
                   help="Regenerate cmax_curve (Fig 5).")
    p.add_argument("--strain-table", action="store_true", dest="strain_table",
                   help="Compute per-strain (C, DR, |n_eff|, grad-norms) and "
                        "Z-channel ceiling check; print LaTeX-paste-ready "
                        "gradient tables.")
    p.add_argument("--correlation", action="store_true",
                   help="Compute Table II (C-|n_eff|-DR correlation matrix) "
                        "and global maxima over the union of both sweeps.")
    p.add_argument("--gradient-max", action="store_true", dest="gradient_max",
                   help="Compute |grad metric|_max over both sweep grids and "
                        "the per-strain ratios (feeds Table IV).")
    p.add_argument("--si-heatmaps", action="store_true", dest="si_heatmaps",
                   help="Regenerate 30 SI heatmap PDFs (Figs S1-S60) into "
                        "--overleaf-dest and staged in --staging-dir.")

    # Long-running: reproduce the sweep NPZ files from scratch
    p.add_argument("--run-mello-sweep", action="store_true", dest="run_mello_sweep",
                   help="SLOW (~10h). Reproduce 7D_Sweep_Results.npz.")
    p.add_argument("--run-keymer-sweep", action="store_true", dest="run_keymer_sweep",
                   help="SLOW (~11h). Reproduce 7D_Sweep_Results_keymer.npz.")

    # Paths
    p.add_argument("--output-dir", default=str(REPO_DIR / "sweep_figures"),
                   help="Directory for main-text figure PNGs "
                        "(default: %(default)s).")
    p.add_argument("--overleaf-dest",
                   default=str(REPO_DIR / "si_heatmaps"),
                   help="Directory to place SI heatmap PDFs "
                        "(default: %(default)s).")
    p.add_argument("--staging-dir",
                   default=str(REPO_DIR / "_heatmap_staging"),
                   help="Staging directory for SI heatmap generation "
                        "(default: %(default)s).")
    p.add_argument("--npz-mello", default=str(NPZ_MELLO_DEFAULT),
                   help="Path to Mello/Tu K_d region sweep NPZ.")
    p.add_argument("--npz-keymer", default=str(NPZ_KEYMER_DEFAULT),
                   help="Path to Wingreen K_d region sweep NPZ.")
    p.add_argument("--sweep-time-budget", type=float, default=None,
                   help="Hours to spend in a --run-*-sweep call before "
                        "checkpointing and exiting (defaults: Mello=12, "
                        "Keymer=11).")

    return p


def main(argv: list[str] | None = None):
    args = _build_arg_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    overleaf_dest = Path(args.overleaf_dest)
    staging_dir = Path(args.staging_dir)
    npz_mello = Path(args.npz_mello)
    npz_keymer = Path(args.npz_keymer)

    if not any([args.all, args.fig3_4, args.fig5, args.strain_table,
                args.correlation, args.gradient_max, args.si_heatmaps,
                args.run_mello_sweep, args.run_keymer_sweep]):
        _build_arg_parser().print_help()
        print("\nERROR: no reproduction target specified. Pass at least one "
              "of --all, --fig3-4, --fig5, --strain-table, --correlation, "
              "--gradient-max, --si-heatmaps, --run-mello-sweep, "
              "--run-keymer-sweep.")
        sys.exit(1)

    if args.run_mello_sweep:
        tb = args.sweep_time_budget if args.sweep_time_budget is not None else 12.0
        run_mello_sweep(npz_path=npz_mello, time_budget_hours=tb)

    if args.run_keymer_sweep:
        tb = args.sweep_time_budget if args.sweep_time_budget is not None else 11.0
        run_keymer_sweep(npz_path=npz_keymer, time_budget_hours=tb)

    if args.all or args.correlation:
        print("\n" + "#" * 78)
        print("# TABLE II: correlation matrix + global maxima")
        print("#" * 78)
        compute_correlation_and_maxima(npz_mello=npz_mello, npz_keymer=npz_keymer)

    if args.all or args.gradient_max:
        print("\n" + "#" * 78)
        print("# TABLE IV: gradient normalization")
        print("#" * 78)
        compute_gradient_max_and_ratios(npz_mello=npz_mello, npz_keymer=npz_keymer)

    if args.all or args.strain_table:
        print("\n" + "#" * 78)
        print("# TABLES III / V / VI + WT/lab gradient LaTeX")
        print("#" * 78)
        compute_strain_table(npz_mello=npz_mello, npz_keymer=npz_keymer)

    if args.all or args.fig3_4:
        print("\n" + "#" * 78)
        print("# FIGURES 3 AND 4")
        print("#" * 78)
        regenerate_fig3_fig4(output_dir=output_dir,
                             npz_mello=npz_mello, npz_keymer=npz_keymer)

    if args.all or args.fig5:
        print("\n" + "#" * 78)
        print("# FIGURE 5")
        print("#" * 78)
        regenerate_fig5(output_dir=output_dir)

    if args.si_heatmaps:
        print("\n" + "#" * 78)
        print("# SI HEATMAPS")
        print("#" * 78)
        regenerate_si_heatmaps(overleaf_dest=overleaf_dest,
                               staging_dir=staging_dir,
                               npz_keymer=npz_keymer)


if __name__ == "__main__":
    main()
