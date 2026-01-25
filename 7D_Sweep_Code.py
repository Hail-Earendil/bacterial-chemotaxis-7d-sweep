# Apparent Selection Pressure for Channel Capacity In E. coli Bacterial Chemotactic Sensor
# 7D Sweep Code

from __future__ import annotations
import json
import re
import time
import os
from dataclasses import dataclass
from datetime import datetime
from math import log
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from scipy.optimize import root_scalar
from datetime import date
from typing import Dict, Tuple, Sequence

REPO_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("SWEEP_BASE_DIR", REPO_DIR / "outputs")).expanduser().resolve()
run_tag = os.environ.get("SWEEP_RUN_TAG", date.today().strftime("%Y%m%d"))

# Global numeric guard for "flat curves"
FLAT_DELTA_P_THRESH = 1e-32

# Warning System
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

def reset_warning_logs() -> None:
    global _WARN_STATS, _WARN_EVENTS, _WARN_DROPPED_EVENTS
    _WARN_STATS = {}
    _WARN_EVENTS = []
    _WARN_DROPPED_EVENTS = 0
    for k in list(_ERR_COUNT.keys()):
        _ERR_COUNT[k] = 0

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
        # sort param sets by frequency
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

# Core MWC Functions
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

# Effective Hill Coefficient Calculation
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

# Channel Capacity Calculator
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
            f"Flat curve: Δp={delta_p:.2e} for "
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

def blahut_arimoto(P: np.ndarray,
                   *, tol: float = 1e-8, max_iter: int = 100000,
                   r_init: np.ndarray | None = None
                   ) -> tuple[float, np.ndarray, np.ndarray, int]:
    EPS = 1e-300
    P = np.asarray(P, float)
    if P.ndim != 2 or not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("Invalid P.")
    rowsum = P.sum(axis=1, keepdims=True)
    if np.any(rowsum <= 0):
        raise ValueError("Some rows of P sum to zero.")
    P = P / rowsum
    M, K = P.shape
    r = (np.full(M, 1.0 / M, float)
         if r_init is None
         else np.asarray(r_init, float) / np.sum(r_init))

    for it in range(1, max_iter + 1):
        q = r @ P
        q = np.maximum(q, EPS)
        D = np.log(P + EPS) - np.log(q[None, :] + EPS)
        z = np.exp((P * D).sum(axis=1))
        s = z.sum()
        r_new = (z / s) if (np.isfinite(s) and s > 0) else r.copy()
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            n_iter = it
            break
        r = r_new
    else:
        n_iter = max_iter
        _warn_once(
            "ba",
            "Blahut–Arimoto hit max_iter; results may be slightly under the true capacity."
        )

    pX = r
    pY = np.maximum(pX @ P, EPS)
    C_bits = float(np.sum(
        pX[:, None] * P * (np.log2(P + EPS) - np.log2(pY[None, :] + EPS))
    ))
    C_bits = max(0.0, min(1.0, C_bits))
    return C_bits, pX, pY, n_iter

# All Three Metric Wrapper Functions
def _interp_c_at_p(pa: np.ndarray,
                   c_vals: np.ndarray,
                   target_p: float) -> float | np.nan:
    inc = pa[0] < pa[-1]
    pgrid, cgrid = (pa, c_vals) if inc else (pa[::-1], c_vals[::-1])
    try:
        return float(np.interp(target_p, pgrid, cgrid))
    except Exception:
        return float('nan')

def metrics_at_params_auto_c(
    *,
    L0: float,
    KdI1: float,
    KdA1: float,
    KdI2: float,
    KdA2: float,
    N_tar: float,
    N_tsr: float,
    mode: str = "binary"
) -> dict:
    c_vals, pa, info = pick_c_grid_from_params(
        L0=L0, KdI1=KdI1, KdA1=KdA1,
        KdI2=KdI2, KdA2=KdA2,
        N_tar=N_tar, N_tsr=N_tsr
    )

    p0, pinf = float(info["p0"]), float(info["pinf"])
    p_min, p_max = (p0, pinf) if p0 <= pinf else (pinf, p0)
    DR_p_signed = float(pinf - p0)
    DR_out_mag  = float(abs(DR_p_signed))

    heff = c50 = np.nan
    if DR_out_mag >= FLAT_DELTA_P_THRESH and np.all(np.isfinite(pa)):
        p_star = 0.5 * (p_min + p_max)
        c50 = info.get("c50", np.nan)
        if not (np.isfinite(c50) and c50 > 0):
            c50 = _interp_c_at_p(pa, c_vals, p_star)

        if np.isfinite(c50) and c50 > 0:
            heff = heff_at_cstar(
                c50, p_star,
                p_min, p_max,
                N_tar, N_tsr,
                KdI1, KdA1, KdI2, KdA2,
                return_abs=True
            )

    if mode != "binary":
        print("[WARN] metrics_at_params_auto_c: only 'binary' channel implemented; using binary.")
    P = build_channel_matrix_binary(pa)
    C_bits, pX_opt, pY, iters = blahut_arimoto(P)

    return {
        "C_bits": float(C_bits),
        "nH": float(heff) if np.isfinite(heff) else np.nan,
        "DR_p": float(DR_p_signed),
        "DR_out": float(DR_out_mag),
        "c50": float(c50) if np.isfinite(c50) else np.nan,
        "iters": int(iters),
    }

# Sweep machinery
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


def _ensure_contains(grid: np.ndarray, value: float,
                     *, as_int: bool = False) -> np.ndarray:
    if as_int:
        g = np.asarray(grid, int)
        v = int(round(float(value)))
        if not np.any(g == v):
            g = np.unique(np.append(g, v)).astype(int)
        return np.sort(g)
    else:
        g = np.asarray(grid, float)
        v = float(value)
        if not (g.min() <= v <= g.max()) or not np.any(
            np.isclose(g, v, rtol=0, atol=1e-12)
        ):
            g = np.unique(np.append(g, v))
        return np.sort(g)

# Heatmap Utilities
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

# Naming System
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

def plot_heatmap_nd(
    results: dict, dep: str,
    xvar: str, yvar: str, *,
    agg: str = "fixed",
    fixed: dict | None = None,
    dots: list[dict] | None = None,
    title: str | None = None,
    savepath: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    label_fs: float = 15,
    tick_fs: float = 12,
    tick_len: float = 4,
    tick_wid: float = 1,
    dot_size: float = 150,
    dot_edge_wid: float = 1,
    legend_fs: float = 15,
    legend_title_fs: float = 15,
    cbar_label_fs: float = 15,
    cbar_tick_fs: float = 12
):
    if dots is None:
        meta = _meta_of(results)
        if isinstance(meta, dict) and "bio_dots" in meta:
            dots = meta["bio_dots"]

    X, Y, Z = slice2_nd(
        results, dep, xvar, yvar,
        agg=agg, fixed=fixed
    )
    use_logx = xvar in LOG_VARS
    use_logy = yvar in LOG_VARS
    Xe = _log_edges(X) if use_logx else np.linspace(X.min(), X.max(), len(X) + 1)
    Ye = _log_edges(Y) if use_logy else np.linspace(Y.min(), Y.max(), len(Y) + 1)

    Zm = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.2)
    pcm = ax.pcolormesh(
        Xe, Ye, Zm,
        shading="flat",
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )

    if use_logx:
        ax.set_xscale("log")
    if use_logy:
        ax.set_yscale("log")

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    if use_logx:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    if use_logy:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))

    ax.set_xlabel(_vname(xvar), fontsize=label_fs)
    ax.set_ylabel(_vname(yvar), fontsize=label_fs)

    ax.tick_params(axis="both", which="major",
                   labelsize=tick_fs, length=tick_len, width=tick_wid)
    ax.tick_params(axis="both", which="minor",
                   labelsize=tick_fs, length=0.75 * tick_len, width=tick_wid)

    if show_colorbar:
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(_dname(dep), fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs,
                            length=tick_len, width=tick_wid)

    show_legend = dots and (len(dots) > 1)
    if dots:
        markers = ["o", "s", "^", "D", "P", "X", "v", "*", "h", "<"]
        lines, labels = [], []
        for i, d in enumerate(dots):
            if xvar not in d or yvar not in d:
                continue
            x, y = float(d[xvar]), float(d[yvar])
            h = ax.scatter(
                [x], [y],
                s=dot_size, marker=markers[i % len(markers)],
                edgecolors="k", facecolors="w",
                linewidths=dot_edge_wid, zorder=5
            )
            if show_legend:
                lines.append(h)
                labels.append(str(d.get("label", d.get("name", f"{i+1}"))))
        if show_legend and lines:
            leg = ax.legend(
                lines, labels,
                title="strain",
                frameon=True,
                loc="best",
                fontsize=legend_fs,
                title_fontsize=legend_title_fs
            )

    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240)
        plt.close(fig)
    else:
        plt.show()

def plot_heatmap_nd_into_ax(
    ax,
    results: dict, dep: str,
    xvar: str, yvar: str, *,
    agg: str = "fixed",
    fixed: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_ticks: bool = False,
):

    X, Y, Z = slice2_nd(results, dep, xvar, yvar, agg=agg, fixed=fixed)

    use_logx = xvar in LOG_VARS
    use_logy = yvar in LOG_VARS
    Xe = _log_edges(X) if use_logx else np.linspace(X.min(), X.max(), len(X) + 1)
    Ye = _log_edges(Y) if use_logy else np.linspace(Y.min(), Y.max(), len(Y) + 1)

    Zm = np.ma.masked_invalid(Z)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.2)

    pcm = ax.pcolormesh(
        Xe, Ye, Zm,
        shading="flat",
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )

    if use_logx:
        ax.set_xscale("log")
    if use_logy:
        ax.set_yscale("log")

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.tick_params(axis="both", which="both", labelsize=8, length=2, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return pcm

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

def export_63_heatmaps_per_strain(
    results: dict,
    bio_list: list[dict],
    outdir: str | Path,
    *,
    show_colorbar: bool = True,
    norm_mode: str = "fixed",
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:

    outdir = Path(outdir) / run_tag / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    vC = _global_range(results["C_bits"], norm_mode=norm_mode, low_pct=low_pct, high_pct=high_pct, vmin=vmin, vmax=vmax)
    vH = _global_range(results["nH"],     norm_mode=norm_mode, low_pct=low_pct, high_pct=high_pct, vmin=vmin, vmax=vmax)
    vD = _global_range(results["DR_out"], norm_mode=norm_mode, low_pct=low_pct, high_pct=high_pct, vmin=vmin, vmax=vmax)

    pairs = [
        (INDEP_VARS[i], INDEP_VARS[j])
        for i in range(len(INDEP_VARS))
        for j in range(i + 1, len(INDEP_VARS))
    ]

    for d in bio_list:
        label = str(d.get("label", d.get("name", "")))
        subdir = outdir / f"strain{label}"
        subdir.mkdir(parents=True, exist_ok=True)

        fixed = {k: d[k] for k in INDEP_VARS}
        this_dot = [d]

        for dep, (vmin_dep, vmax_dep) in (
            ("C_bits", vC),
            ("nH",     vH),
            ("DR_out", vD),
        ):
            for xv, yv in pairs:
                fn = subdir / f"s{label}_{_dfile(dep)}_{_vfile(xv)}_{_vfile(yv)}.png"
                plot_heatmap_nd(
                    results, dep, xv, yv,
                    agg="fixed", fixed=fixed, dots=this_dot,
                    savepath=fn,
                    vmin=vmin_dep, vmax=vmax_dep,
                    show_colorbar=show_colorbar
                )

        print(f"[PLOTS] Strain {label}: wrote 63 heatmaps -> {subdir.resolve()}")

# R Coefficient
def _flatten_finite(arr: np.ndarray) -> np.ndarray:
    v = np.asarray(arr, float).ravel()
    return v[np.isfinite(v)]

def r2_matrix_dependent(results: dict,
                        deps=("C_bits", "nH", "DR_out"),
                        joint_mask: bool = True
                        ) -> tuple[np.ndarray, list[str]]:
    
    vals = [np.asarray(results[d], float).ravel() for d in deps]
    if joint_mask:
        m = np.ones_like(vals[0], dtype=bool)
        for v in vals:
            m &= np.isfinite(v)
        vals = [v[m] for v in vals]
    else:
        vals = [_flatten_finite(v) for v in vals]

    N = len(vals)
    R = np.full((N, N), np.nan, float)
    for i in range(N):
        for j in range(N):
            x, y = vals[i], vals[j]
            if not joint_mask:
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
            if x.size >= 2 and y.size >= 2:
                r = np.corrcoef(x, y)[0, 1]
                R[i, j] = float(r)
    return R, list(deps)

def plot_r2_dependent(results: dict,
                      deps=("C_bits", "nH", "DR_out"),
                      title="Pairwise correlation (R) among dependent variables",
                      savepath: str | Path | None = None,
                      show: bool = True) -> np.ndarray:
    
    R, labels = r2_matrix_dependent(
        results, deps=deps, joint_mask=True
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(
        R,
        vmin=-1.0, vmax=+1.0,
        cmap="coolwarm",
        origin="upper"
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='R')

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            val = R[i, j]
            if np.isfinite(val):
                txt = f"{val:+.3f}"
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=11, color=color
                )

    fig.tight_layout()
    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240)
        plt.close(fig)
        print(f"[PLOT] wrote {savepath.resolve()}")
    elif show:
        plt.show()
    return R

def export_r2_plot(results: dict, outdir: str | Path,
                   deps=("C_bits", "nH", "DR_out")) -> np.ndarray:
    outdir = Path(outdir) / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    savepath = outdir / "r_coefficient.png"
    R = plot_r2_dependent(
        results,
        deps=deps,
        title="Pairwise correlation (R) among dependent variables",
        savepath=savepath,
        show=False
    )
    print("[R  ] Matrix:\n", np.array2string(R, precision=3, suppress_small=True))
    return R

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

def pick_even_indices(n: int, k: int) -> np.ndarray:

    n = int(n)
    k = int(k)
    if n <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    # Ensure endpoints
    if idx.size == 0 or idx[0] != 0:
        idx = np.unique(np.append(idx, 0))
    if idx[-1] != (n - 1):
        idx = np.unique(np.append(idx, n - 1))
    return np.sort(idx)

def make_subsample_indices(grids: dict,
                           *, k_log: int, k_N: int) -> dict[str, np.ndarray]:

    idx = {}
    for name in INDEP_VARS:
        n = len(grids[name])
        k = k_N if name in {"N_tar", "N_tsr"} else k_log
        idx[name] = pick_even_indices(n, k)
    return idx

def subselect_results(results: dict,
                      idx: dict[str, np.ndarray],
                      deps: tuple[str, ...] = DEP_DEFAULT) -> dict:
    
    grids = results["grids"]

    slicer = np.ix_(
        idx["L0"],
        idx["KdI1"],
        idx["KdA1"],
        idx["KdI2"],
        idx["KdA2"],
        idx["N_tar"],
        idx["N_tsr"],
    )

    out = {"grids": {k: grids[k][idx[k]] for k in INDEP_VARS}}
    for dep in deps:
        out[dep] = np.asarray(results[dep], float)[slicer]

    if "DR_out" not in deps and "DR_out" in results:
        out["DR_out"] = np.asarray(results["DR_out"], float)[slicer]

    return out

def assert_same_region(full_grids: dict, sub_grids: dict) -> None:
    for name in INDEP_VARS:
        f = np.asarray(full_grids[name], float)
        s = np.asarray(sub_grids[name], float)
        fmin, fmax = float(f.min()), float(f.max())
        smin, smax = float(s.min()), float(s.max())
        if not (np.isclose(fmin, smin, rtol=0, atol=0) and np.isclose(fmax, smax, rtol=0, atol=0)):
            raise AssertionError(
                f"Region mismatch for {name}: full=[{fmin}, {fmax}] vs sub=[{smin}, {smax}]"
            )

def corr_R(results_sub: dict,
           deps: tuple[str, ...] = DEP_DEFAULT,
           *, responsive_only: bool = True,
           dr_eps: float = 1e-12) -> tuple[np.ndarray, int, int]:
    vals = [np.asarray(results_sub[d], float).ravel() for d in deps]
    n_total = int(vals[0].size)

    mask = np.ones_like(vals[0], dtype=bool)
    for v in vals:
        mask &= np.isfinite(v)

    if responsive_only:
        dr = np.asarray(results_sub["DR_out"], float).ravel()
        mask &= np.isfinite(dr) & (dr > float(dr_eps))

    vals = [v[mask] for v in vals]
    n_used = int(vals[0].size)

    if n_used < 3:
        return np.full((len(deps), len(deps)), np.nan, float), n_used, n_total

    X = np.vstack(vals)
    R = np.corrcoef(X)
    return R, n_used, n_total

def select_diagonal_path(
    rows: list[dict],
    *,
    k_list: tuple[int, ...] = (2, 3, 4, 5, 7, 10),
    prefer_largest_used: bool = True,
) -> list[dict]:

    picked: dict[int, dict] = {}
    for r in rows:
        klog = int(r["k_log"])
        kN = int(r["k_N"])
        if klog != kN or klog not in k_list:
            continue

        if klog not in picked:
            picked[klog] = r
        else:
            if prefer_largest_used and int(r.get("n_used", 0)) > int(picked[klog].get("n_used", 0)):
                picked[klog] = r

    return [picked[k] for k in sorted(picked.keys())]


def plot_R_convergence_diagonal_vs_nused(
    path_rows: list[dict],
    *,
    savepath: str | Path,
) -> None:

    if not path_rows:
        raise ValueError("path_rows is empty. Did you run select_diagonal_path(...) ?")

    xs = np.array([int(r["n_used"]) for r in path_rows], float)

    r1 = np.array([r.get("R_C_nH", np.nan) for r in path_rows], float)
    r2 = np.array([r.get("R_C_DR", np.nan) for r in path_rows], float)
    r3 = np.array([r.get("R_nH_DR", np.nan) for r in path_rows], float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(xs, r1, marker="o", linestyle="-", label=r"$R(C,\ n_{\mathrm{eff}})$")
    ax.plot(xs, r2, marker="o", linestyle="-", label=r"$R(C,\ \mathrm{DR})$")
    ax.plot(xs, r3, marker="o", linestyle="-", label=r"$R(n_{\mathrm{eff}},\ \mathrm{DR})$")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Grid Points Used")
    ax.set_ylabel(r"Pairwise Pearson Correlation Coefficient $R$")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=240)
    plt.close(fig)


def export_r_convergence_diagonal_vs_nused(
    results: dict,
    outdir: str | Path,
    *,
    k_list: tuple[int, ...] = (2, 3, 4, 5, 7, 10),
    deps: tuple[str, ...] = ("C_bits", "nH", "DR_out"),
    responsive_only: bool = True,
    dr_eps: float = 1e-12,
    verify_region: bool = True,
) -> Path:

    outdir = Path(outdir)
    plots_dir = outdir / "plots"

    full_grids = results["grids"]
    rows: list[dict] = []

    for k in k_list:
        idx = make_subsample_indices(full_grids, k_log=int(k), k_N=int(k))
        sub = subselect_results(results, idx, deps=deps)

        if verify_region:
            assert_same_region(full_grids, sub["grids"])

        R, n_used, n_total = corr_R(
            sub,
            deps=deps,
            responsive_only=responsive_only,
            dr_eps=dr_eps,
        )

        entry = {
            "k_log": int(k),
            "k_N": int(k),
            "n_total": int(n_total),
            "n_used": int(n_used),
            "R_matrix": R,
        }

        if deps == ("C_bits", "nH", "DR_out") and R.shape == (3, 3):
            entry.update({
                "R_C_nH": float(R[0, 1]) if np.isfinite(R[0, 1]) else np.nan,
                "R_C_DR": float(R[0, 2]) if np.isfinite(R[0, 2]) else np.nan,
                "R_nH_DR": float(R[1, 2]) if np.isfinite(R[1, 2]) else np.nan,
            })

        rows.append(entry)

    path_rows = select_diagonal_path(rows, k_list=k_list)

    savepath = plots_dir / "R_convergence_diagonal_vs_nused.png"
    plot_R_convergence_diagonal_vs_nused(path_rows, savepath=savepath)
    return savepath


# Biological Parameter
def bio_dots_all10() -> list[dict]:
    KdI1 = 0.0492
    C1 = 0.449
    KdA1 = KdI1 / C1  # Tar
    KdI2 = 0.0345
    C2 = 0.314
    KdA2 = KdI2 / C2  # Tsr
    ell0_bar, ell1, ell2 = 0.826, 1.23, 1.54

    strains = [
        ("strain1",  4.95, 16.5),
        ("strain2",  4.00,  8.0),
        ("strain3",  4.39,  4.39),
        ("strain4", 18.70,  6.24),
        ("strain5", 14.00,  0.0),
        ("strain6", 29.80,  0.0),
        ("strain7", 73.50,  0.0),
        ("strain8",  0.0,   9.85),
        ("strain9",  0.0,  15.20),
        ("strain10", 0.0,  32.30),
    ]

    dots = []
    for name, Nt, Ns in strains:
        L_paper = float(
            ell0_bar * (ell1 ** Nt) * (ell2 ** Ns)
        )
        L0 = 1.0 / L_paper
        dots.append({
            "name": name,
            "label": name.replace("strain", ""),
            "L0": L0,
            "KdI1": KdI1, "KdA1": KdA1,
            "KdI2": KdI2, "KdA2": KdA2,
            "N_tar": Nt,  "N_tsr": Ns,
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

def L0_from_components(N_tar: float, N_tsr: float,
                       ell0_bar: float = 0.826,
                       ell1: float = 1.23,
                       ell2: float = 1.54) -> float:
    L_paper = float(ell0_bar * (ell1 ** N_tar) * (ell2 ** N_tsr))
    return 1.0 / L_paper


def L0_from_p0(p0: float) -> float:
    p0 = float(np.clip(p0, 1e-12, 1-1e-12))
    return float((1.0 - p0) / p0)


def biological_anchor_strain2(use_p0: float | None = None) -> dict:
    N_tar = 4.00
    N_tsr = 8.00
    KdI1 = 0.0492   # mM
    KdA1 = 0.1096   # mM
    KdI2 = 0.0345   # mM
    KdA2 = 0.1099   # mM
    if use_p0 is not None:
        L0 = L0_from_p0(use_p0)
    else:
        L0 = L0_from_components(N_tar, N_tsr)  # ≈ 59.8
    return dict(
        L0=L0,
        KdI1=KdI1, KdA1=KdA1,
        KdI2=KdI2, KdA2=KdA2,
        N_tar=N_tar, N_tsr=N_tsr
    )

def get_bio_list(results: dict) -> list[dict]:
    meta = results.get("meta", {})
    if isinstance(meta, np.ndarray):
        try:
            meta = meta.item()
        except Exception:
            meta = {}
    if isinstance(meta, dict) and meta.get("bio_dots"):
        return meta["bio_dots"]
    return bio_dots_all10()

# Grid Construction
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

# BA visualization exports
def ba_at_params(L0: float, KdI1: float, KdA1: float,
                 KdI2: float, KdA2: float,
                 N_tar: float, N_tsr: float) -> dict:
    c_vals, pa, info = pick_c_grid_from_params(
        L0=L0, KdI1=KdI1, KdA1=KdA1,
        KdI2=KdI2, KdA2=KdA2,
        N_tar=N_tar, N_tsr=N_tsr
    )
    P = build_channel_matrix_binary(pa)
    C_bits, pX, pY, iters = blahut_arimoto(P)
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

def plot_px_vs_c(c_vals: np.ndarray, pX: np.ndarray,
                 info: dict | None = None,
                 savepath: str | Path | None = None,
                 show: bool = False,
                 title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(c_vals, pX, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Ligand concentration $c$ (mM)")
    ax.set_ylabel(r"Capacity-achieving input distribution $p_{\mathrm{in}}^*(c)$")
    if title:
        ax.set_title(title)

    if info:
        labels, lines = [], []
        for name, style in (("c10", "--"), ("c50", "-."), ("c90", ":")):
            v = info.get(name, np.nan)
            if np.isfinite(v) and v > 0:
                line = ax.axvline(
                    v, linestyle=style, linewidth=1.6, alpha=0.9
                )
                lines.append(line)
                labels.append(name)
        if lines:
            ax.legend(["$p_X^*(c)$"] + labels, loc="best")

    fig.tight_layout()
    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240)
        plt.close(fig)
    elif show:
        plt.show()

def plot_pa_vs_c(c_vals: np.ndarray, pa: np.ndarray,
                 savepath: str | Path | None = None,
                 show: bool = False,
                 title: str | None = r"$p(c)$") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(c_vals, pa, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Ligand concentration $c$ (mM)")
    ax.set_ylabel(r"$p(c)$")
    fig.tight_layout()
    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240)
        plt.close(fig)
    elif show:
        plt.show()

def plot_py(pY: np.ndarray,
            savepath: str | Path | None = None,
            show: bool = False,
            title: str | None = "$p^*(s)$ (Inactive/Active)") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.bar([0, 1], pY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"])
    ax.set_ylabel(r"Optimal output probability $p^*(s)$")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240)
        plt.close(fig)
    elif show:
        plt.show()

def export_ba_px_py_per_strain(bio_list: list[dict],
                               outdir: str | Path) -> None:
    
    base = Path(outdir) / "plots"
    base.mkdir(parents=True, exist_ok=True)

    overlay_data = []

    for d in bio_list:
        label = str(d.get("label", d.get("name", "")))
        subdir = base / f"strain{label}"
        subdir.mkdir(parents=True, exist_ok=True)

        res = ba_at_params(
            d["L0"], d["KdI1"], d["KdA1"],
            d["KdI2"], d["KdA2"],
            d["N_tar"], d["N_tsr"]
        )

        plot_px_vs_c(
            res["c_vals"], res["pX"], info=res["info"],
            savepath=subdir / f"{label}_OptimalInput_vs_c.png",
            title=f"Strain {label}: $C^*={res['C_bits']:.3f}$ bits"
        )
        plot_pa_vs_c(
            res["c_vals"], res["pa"],
            savepath=subdir / f"{label}_pactive_vs_c.png",
            title=f"Strain {label}: $p(c)$"
        )
        plot_py(
            res["pY"],
            savepath=subdir / f"{label}_pY.png",
            title=f"Strain {label}: $p_Y^*$ (Inactive/Active)"
        )
        print(
            f"[BA ] Strain {label}: C* = {res['C_bits']:.3f}  "
            f"→ plots in {subdir.resolve()}"
        )

        overlay_data.append({
            "label": label,
            "c_vals": res["c_vals"],
            "pa": res["pa"],
            "pX": res["pX"],
            "C_bits": res["C_bits"],
        })

    if overlay_data:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for item in overlay_data:
            ax.plot(
                item["c_vals"],
                item["pa"],
                lw=1.8,
                label=f"{item['label']}"
            )
        ax.set_xscale("log")
        ax.set_xlabel("Ligand c (mM)")
        ax.set_ylabel(r"$p(c)$")
        ax.legend(title="strain", fontsize=8, frameon=True, loc="best")
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(base / "all_strains_pactive_vs_c.png", dpi=240)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for item in overlay_data:
            ax.plot(
                item["c_vals"],
                item["pX"],
                lw=1.8,
                label=f"{item['label']}"
            )
        ax.set_xscale("log")
        ax.set_xlabel("Ligand concentration $c$ (mM)")
        ax.set_ylabel(r"Capacity-achieving input distribution $p_{\mathrm{in}}^*(c)$")
        ax.legend(title="strain", fontsize=8, frameon=True, loc="best")
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(base / "all_strains_OptimalInput_vs_c.png", dpi=240)
        plt.close(fig)

def export_combined_pY_grouped(bio_list: list[dict],
                               outdir: str | Path) -> None:
    
    base = Path(outdir) / "plots"
    base.mkdir(parents=True, exist_ok=True)

    pY_list = []
    strain_labels = []

    for d in bio_list:
        label = str(d.get("label", d.get("name", "")))
        strain_labels.append(label)

        res = ba_at_params(
            d["L0"], d["KdI1"], d["KdA1"],
            d["KdI2"], d["KdA2"],
            d["N_tar"], d["N_tsr"]
        )
        pY_list.append(res["pY"])

    pY_arr = np.vstack(pY_list)

    n_strains = pY_arr.shape[0]
    x = np.arange(n_strains)
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.bar(
        x - width / 2,
        pY_arr[:, 0],
        width,
        label="Inactive"
    )
    ax.bar(
        x + width / 2,
        pY_arr[:, 1],
        width,
        label="Active"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(strain_labels)
    ax.set_xlabel("Strain")
    ax.set_ylabel(r"Optimal output probability $p^*(s)$")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.tight_layout()
    outpath = base / "combined_pY_grouped.png"
    fig.savefig(outpath, dpi=240)
    plt.close(fig)
    print(f"[PLOT] wrote grouped pY* bar chart → {outpath.resolve()}")

# 1D Curves Export
def slice1_nd(results: dict, dep: str, xvar: str,
              *, fixed: dict, agg: str = "median"
              ) -> tuple[np.ndarray, np.ndarray]:
    
    if dep not in DEP_VARS:
        raise ValueError(f"Unknown dep '{dep}'. Options: {DEP_VARS}")
    if xvar not in INDEP_VARS:
        raise ValueError(f"xvar must be in {INDEP_VARS}.")

    Z = np.asarray(results[dep], float)
    ax_map = {name: i for i, name in enumerate(INDEP_VARS)}
    ax_x = ax_map[xvar]

    indexer: list[object] = [slice(None)] * Z.ndim
    for name, ax in ax_map.items():
        if name == xvar:
            continue
        grid = _grid_of(results, name)
        val  = fixed.get(name, grid[len(grid)//2])
        idx  = _nearest_index(grid, float(val))
        indexer[ax] = idx

    sub = Z[tuple(indexer)]
    surviving_axes = [ax for ax, sel in enumerate(indexer)
                      if isinstance(sel, slice)]
    pos_x = 0 if len(surviving_axes) == 1 else surviving_axes.index(ax_x)

    if sub.ndim > 1 and pos_x != (sub.ndim - 1):
        sub = np.moveaxis(sub, pos_x, -1)

    if sub.ndim > 1:
        reduce_axes = tuple(range(sub.ndim - 1))
        if agg == "median":
            sub = np.nanmedian(sub, axis=reduce_axes)
        elif agg == "mean":
            sub = np.nanmean(sub, axis=reduce_axes)
        else:
            raise ValueError("agg must be 'median' or 'mean'.")

    X = _grid_of(results, xvar)
    Y = np.asarray(sub, float)
    if Y.shape != X.shape:
        Y = Y.reshape(X.shape)
    return X, Y

@dataclass
class CurveStyle:
    figsize: tuple[float, float] = (8.0, 5.6)
    label_fs: float = 24
    tick_fs: float = 20
    tick_len: float = 8.0
    tick_wid: float = 2.0
    log_numticks: int = 5
    line_w: float = 4.0
    dot_size: float = 120
    dot_edge_wid: float = 2.4
    dot_marker: str = "o"
    legend_fs: float = 18
    legend_title_fs: float = 18
    legend_markerscale: float = 2.0
    legend_loc: str = "best"
    grid: bool = True
    grid_alpha: float = 0.35
    grid_ls: str = ":"
    title: str | None = None

def _apply_log_ticks(ax, *, which: str, numticks: int, tick_len: float, tick_wid: float):

    locator_major = LogLocator(base=10.0, numticks=int(numticks))
    locator_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    formatter = LogFormatterSciNotation(base=10.0)

    if which == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(locator_major)
        ax.xaxis.set_minor_locator(locator_minor)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", which="major", length=tick_len, width=tick_wid)
        ax.tick_params(axis="x", which="minor", length=0.75 * tick_len, width=tick_wid)
    elif which == "y":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(locator_major)
        ax.yaxis.set_minor_locator(locator_minor)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="y", which="major", length=tick_len, width=tick_wid)
        ax.tick_params(axis="y", which="minor", length=0.75 * tick_len, width=tick_wid)
    else:
        raise ValueError("which must be 'x' or 'y'")

def export_metric_vs_vars_for_strains(
    results: dict,
    bio_list: list[dict],
    outdir: str | Path,
    *,
    dep: str,
    out_subdir: str,
    y_label: str | None = None,
    agg: str = "median",
    style: CurveStyle | None = None,
) -> None:

    if dep not in DEP_VARS:
        raise ValueError(f"Unknown dep '{dep}'. Options: {DEP_VARS}")

    st = style if style is not None else CurveStyle()

    outdir = Path(outdir) / out_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    for xvar in INDEP_VARS:
        fig, ax = plt.subplots(figsize=st.figsize)

        for bio in bio_list:
            fixed = {
                k: float(bio[k])
                for k in INDEP_VARS if k != xvar
            }

            X, Y = slice1_nd(
                results, dep, xvar,
                fixed=fixed, agg=agg
            )

            ax.plot(
                X, Y,
                linewidth=st.line_w,
                label=str(bio.get("label", bio.get("name", "")))
            )

            x_anchor = float(bio[xvar])
            idx = int(np.argmin(np.abs(X - x_anchor)))
            ax.scatter(
                [X[idx]], [Y[idx]],
                s=st.dot_size,
                marker=st.dot_marker,
                facecolors="white",
                edgecolors="black",
                linewidths=st.dot_edge_wid,
                zorder=5
            )

        if xvar in LOG_VARS:
            _apply_log_ticks(
                ax,
                which="x",
                numticks=st.log_numticks,
                tick_len=st.tick_len,
                tick_wid=st.tick_wid,
            )
        else:
            ax.tick_params(axis="x", which="major",
                           labelsize=st.tick_fs, length=st.tick_len, width=st.tick_wid)
            ax.tick_params(axis="x", which="minor",
                           labelsize=st.tick_fs, length=0.75 * st.tick_len, width=st.tick_wid)

        ax.tick_params(axis="y", which="major",
                       labelsize=st.tick_fs, length=st.tick_len, width=st.tick_wid)
        ax.tick_params(axis="y", which="minor",
                       labelsize=st.tick_fs, length=0.75 * st.tick_len, width=st.tick_wid)

        ax.set_xlabel(_vname(xvar), fontsize=st.label_fs)

        if y_label is None:
            ax.set_ylabel(_dname(dep), fontsize=st.label_fs)
        else:
            ax.set_ylabel(y_label, fontsize=st.label_fs)

        if st.title:
            ax.set_title(st.title, fontsize=st.label_fs)

        ax.legend(
            title="Strain",
            frameon=True,
            fontsize=st.legend_fs,
            title_fontsize=st.legend_title_fs,
            markerscale=st.legend_markerscale,
            loc=st.legend_loc,
        )

        if st.grid:
            ax.grid(True, which="both", linestyle=st.grid_ls, alpha=st.grid_alpha)

        fig.tight_layout()
        fname = f"{_dfile(dep)}_vs_{_vfile(xvar)}.png"
        fig.savefig(outdir / fname, dpi=240)
        plt.close(fig)

def export_capacity_vs_vars_for_strains(
    results: dict,
    bio_list: list[dict],
    outdir: str | Path,
    *,
    style: CurveStyle | None = None,
    agg: str = "median",
) -> None:
    export_metric_vs_vars_for_strains(
        results, bio_list, outdir,
        dep="C_bits",
        out_subdir="capacity_curves",
        y_label=r"Channel capacity (bits)",
        agg=agg,
        style=style,
    )

def export_dynamic_range_vs_vars_for_strains(
    results: dict,
    bio_list: list[dict],
    outdir: str | Path,
    *,
    style: CurveStyle | None = None,
    agg: str = "median",
) -> None:
    export_metric_vs_vars_for_strains(
        results, bio_list, outdir,
        dep="DR_out",
        out_subdir="dynamic_range_curves",
        y_label=r"Dynamic range",
        agg=agg,
        style=style,
    )

def export_neff_vs_vars_for_strains(
    results: dict,
    bio_list: list[dict],
    outdir: str | Path,
    *,
    style: CurveStyle | None = None,
    agg: str = "median",
) -> None:
    export_metric_vs_vars_for_strains(
        results, bio_list, outdir,
        dep="nH",
        out_subdir="neff_curves",
        y_label=r"Effective Hill coefficient $n_{\mathrm{eff}}$",
        agg=agg,
        style=style,
    )

# CSV Output
def save_table_csv(rows: list[dict], path: str | Path) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            pass
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# Tables Export
def _nearest_idx_on_grids(grids: dict, point: dict) -> tuple[int, ...]:
    idx = []
    for name in INDEP_VARS:
        g = np.asarray(grids[name], float)
        idx.append(int(np.argmin(np.abs(g - float(point[name])))))
    return tuple(idx)

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

def build_table_A_capacity(results: dict,
                           bio_list: list[dict]) -> list[dict]:
    C = np.asarray(results["C_bits"], float)
    grids = results["grids"]
    Cmax7D = float(np.nanmax(C))
    rows = []
    for d in bio_list:
        idx = _nearest_idx_on_grids(grids, d)
        C_bio = float(C[idx])
        rows.append({
            "strain": d.get("name", d.get("label", "?")),
            "value_at_bio": C_bio,
            "theoretical_max": 1.0,
            "max_over_7D": Cmax7D,
            "ratio_bio_over_7Dmax": float(C_bio / Cmax7D) if Cmax7D > 0 else np.nan,
        })
    return rows

def build_table_A_metric(results: dict,
                         bio_list: list[dict],
                         dep_key: str,
                         theoretical_max: float | None = None
                         ) -> list[dict]:
    A = np.asarray(results[dep_key], float)
    grids = results["grids"]
    Amax7D = float(np.nanmax(A))
    rows = []
    for d in bio_list:
        idx = _nearest_idx_on_grids(grids, d)
        Aval = float(A[idx])
        rows.append({
            "strain": d.get("name", d.get("label", "?")),
            "value_at_bio": Aval,
            "theoretical_max": (float(theoretical_max)
                                if theoretical_max is not None else np.nan),
            "max_over_7D": Amax7D,
            "ratio_bio_over_7Dmax": float(Aval / Amax7D) if Amax7D > 0 else np.nan,
        })
    return rows

def build_table_B_gradnorm_combined(
    results: dict,
    bio_list: list[dict],
    *,
    steps: TableBStepConfig
) -> list[dict]:
    grids = results["grids"]
    bounds = _tableb_bounds_from_grids(grids)

    rows: list[dict] = []
    for d in bio_list:
        p = {k: float(d[k]) for k in INDEP_VARS}

        gC  = tableb_grad_norm_fixed_steps(p, "C_bits",  steps=steps, bounds=bounds)
        gDR = tableb_grad_norm_fixed_steps(p, "DR_out",  steps=steps, bounds=bounds)
        gH  = tableb_grad_norm_fixed_steps(p, "nH",      steps=steps, bounds=bounds)

        rows.append({
            "strain": d.get("name", d.get("label", "?")),
            "grad_L2_C": float(gC),
            "grad_L2_DR": float(gDR),
            "grad_L2_neff": float(gH),
        })

    return rows

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
    deadline   = start_time + time_budget_hours * 3600.0
    next_ckpt  = start_time + checkpoint_every_min * 60.0
    next_eta   = start_time + eta_every_min * 60.0

    i = int(cursor)
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
                    mode="binary"
                )

            for k in DEP_VARS:
                arrays[k][a, b, c, d, e, f, g] = np.float32(res.get(k, np.nan))
            iters[a, b, c, d, e, f, g] = int(res.get("iters", 0))
            done_mask[a, b, c, d, e, f, g] = True

        except Exception as ex:
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
            f"[NPZ] saved → {npz_path} | complete={complete} | "
            f"coverage={coverage:5.2f}% | elapsed={elapsed_txt} | "
            f"ETA {eta_txt}"
        )

    return _results_from_npz_arrays(grids, arrays, meta, done_mask, cursor, complete)

# Full Panel Export
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

    # ---- DEFAULTS (now 7x3) ----
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
    
    outdir = Path(outdir) / "full_panels"
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

#BA Algorithm Convergence Check
def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))

def analyze_ba_convergence_from_npz(
    npz_path: str | Path,
    *,
    max_iter: int = 100000,
    responsive_only: bool = False,
    dr_eps: float = 1e-12,
) -> dict:

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if "iters" not in data or "done_mask" not in data:
        raise KeyError("NPZ must contain 'iters' and 'done_mask' arrays.")

    iters = np.asarray(data["iters"], dtype=int)
    done_mask = np.asarray(data["done_mask"], dtype=bool)

    mask = done_mask.copy()

    if responsive_only:
        if "DR_out" not in data:
            raise KeyError("responsive_only=True requires 'DR_out' in NPZ.")
        dr = np.asarray(data["DR_out"], dtype=float)
        mask &= np.isfinite(dr) & (dr > float(dr_eps))

    it_used = iters[mask]
    it_used = it_used[np.isfinite(it_used)]

    n = int(it_used.size)
    n_cap = int(np.count_nonzero(it_used == int(max_iter)))
    frac_cap = float(n_cap / n) if n > 0 else float("nan")

    it_noncap = it_used[it_used < int(max_iter)]
    max_noncap = float(np.max(it_noncap)) if it_noncap.size > 0 else float("nan")

    out = {
        "npz": str(npz_path),
        "scope": "responsive_only" if responsive_only else "all_done",
        "max_iter_cap": int(max_iter),
        "n_points": int(n),
        "n_hit_cap": int(n_cap),
        "frac_hit_cap": float(frac_cap),
        "iters_min": float(np.min(it_used)) if n > 0 else float("nan"),
        "iters_median": _safe_quantile(it_used, 0.50),
        "iters_p90": _safe_quantile(it_used, 0.90),
        "iters_p99": _safe_quantile(it_used, 0.99),
        "iters_max": float(np.max(it_used)) if n > 0 else float("nan"),
        "iters_max_noncap": float(max_noncap),
    }
    return out

def export_ba_convergence_reports(
    npz_path: str | Path,
    outdir: str | Path,
    *,
    max_iter: int = 100000,
    dr_eps: float = 1e-12,
    verbose: bool = True,
) -> dict:

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    s_all = analyze_ba_convergence_from_npz(
        npz_path, max_iter=max_iter, responsive_only=False, dr_eps=dr_eps
    )
    s_resp = analyze_ba_convergence_from_npz(
        npz_path, max_iter=max_iter, responsive_only=True, dr_eps=dr_eps
    )

    rows = []
    for s in (s_all, s_resp):
        rows.append({
            "scope": s["scope"],
            "max_iter_cap": s["max_iter_cap"],
            "n_points": s["n_points"],
            "n_hit_cap": s["n_hit_cap"],
            "frac_hit_cap": s["frac_hit_cap"],
            "iters_min": s["iters_min"],
            "iters_median": s["iters_median"],
            "iters_p90": s["iters_p90"],
            "iters_p99": s["iters_p99"],
            "iters_max": s["iters_max"],
            "iters_max_noncap": s["iters_max_noncap"],
        })

    save_table_csv(rows, outdir / "ba_convergence_summary.csv")

    if verbose:
        for s in (s_all, s_resp):
            pct = 100.0 * s["frac_hit_cap"] if np.isfinite(s["frac_hit_cap"]) else float("nan")
            print(
                f"[BA] scope={s['scope']} | n={s['n_points']} | "
                f"hit_cap={s['n_hit_cap']} ({pct:.3f}%) | "
                f"median={s['iters_median']:.0f} | p99={s['iters_p99']:.0f} | "
                f"max_noncap={s['iters_max_noncap']:.0f}"
            )
        print(f"[BA] wrote { (outdir / 'ba_convergence_summary.csv').resolve() }")

    return {"all_done": s_all, "responsive_only": s_resp}

# Main Function
def main():
    bio_list = bio_dots_all10()

    anchor = next(
        (d for d in bio_list
         if str(d.get("name", "")).lower() == "strain2"
         or str(d.get("label", "")) == "2"),
        bio_list[0]
    )

    grids = build_grids_pilot(
        anchor,
        points_log=7,
        points_N=7,
        L0_span_dec=2.0,
        K_span_dec=2.0
    )
    grids = expand_grids_to_cover_biodots(grids, bio_list,
                                          L0_pad=2.0, N_pad=2)

    for name in INDEP_VARS:
        for d in bio_list:
            grids[name] = _ensure_contains(
                grids[name], d[name],
                as_int=False
            )

    base = BASE_DIR / run_tag
    base.mkdir(parents=True, exist_ok=True)
    npz_path = base / "sweep_7d_long.npz"

    results = run_resumable_sweep_npz(
        npz_path=npz_path,
        grids=grids,
        bio_list=bio_list,
        anchor=anchor,
        time_budget_hours=12,
        checkpoint_every_min=30,
        eta_every_min=1,
        progress=True
    )

    logs_dir = base / "logs"
    write_warning_logs(logs_dir, tag=run_tag)

    export_ba_convergence_reports(
        npz_path=npz_path,
        outdir=base / "tables",
        max_iter=100000,
        dr_eps=1e-12,
        verbose=True,
    )

    if not results.get("complete", False):
        print("[INFO] Partial sweep saved to NPZ. Warning logs written to /logs.")
        return

    print("[INFO] Sweep complete — running post-processing.")

    HEATMAP_NORM_MODE = "percentile"
    HEATMAP_LOW_PCT   = 2.0
    HEATMAP_HIGH_PCT  = 98.0
    HEATMAP_VMIN = None
    HEATMAP_VMAX = None

    export_63_heatmaps_per_strain(
        results,
        bio_list=bio_list,
        outdir=BASE_DIR,
        show_colorbar=True,
        norm_mode="percentile",
        low_pct=2.0,
        high_pct=98.0,
    )

    export_full_panel_pages_per_strain(
        results, bio_list, base,
        fmt="pdf",
        style=FULL_PANEL_STYLE,
        norm_mode=HEATMAP_NORM_MODE,
        low_pct=HEATMAP_LOW_PCT,
        high_pct=HEATMAP_HIGH_PCT,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
    )

    curve_style = CurveStyle(
        figsize=(8.0, 5.6),
        label_fs=15,
        tick_fs=12,
        tick_len=4.0,
        tick_wid=1.0,
        log_numticks=5,
        line_w=4.0,
        dot_size=80,
        dot_edge_wid=1,
        legend_fs=12,
        legend_title_fs=12,
        legend_markerscale=2.0,
        grid=True,
        grid_alpha=0.35,
        title=None,
    )

    export_ba_px_py_per_strain(bio_list, base)
    export_combined_pY_grouped(bio_list, base)

    export_capacity_vs_vars_for_strains(results, bio_list, base, style=curve_style)
    export_dynamic_range_vs_vars_for_strains(results, bio_list, base, style=curve_style)
    export_neff_vs_vars_for_strains(results, bio_list, base, style=curve_style)

    export_r2_plot(results, base, deps=("C_bits", "nH", "DR_out"))

    export_r_convergence_diagonal_vs_nused(
        results,
        base,
        k_list=(2, 3, 4, 5, 7, 10),
        deps=("C_bits", "nH", "DR_out"),
        responsive_only=True,
        dr_eps=1e-12,
    )

    tableA_C  = build_table_A_capacity(results, bio_list)
    tableA_nH = build_table_A_metric(results, bio_list,
                                     dep_key="nH", theoretical_max=None)
    tableA_DR = build_table_A_metric(results, bio_list,
                                     dep_key="DR_out",
                                     theoretical_max=1.0)
        
    steps = TableBStepConfig(dlog10=0.01, dN=1.0)

    reset_tableb_metrics_cache()
    tableB_grad = build_table_B_gradnorm_combined(results, bio_list, steps=steps)
    outdir = base
    save_table_csv(tableA_C,  outdir / "tables" / "table_A_capacity.csv")
    save_table_csv(tableA_nH, outdir / "tables" / "table_A_neff.csv")
    save_table_csv(tableA_DR, outdir / "tables" / "table_A_DR.csv")
    save_table_csv(tableB_grad, outdir / "tables" / "table_B_gradnorm.csv")


    print("[DONE] All plots/tables written.") 

if __name__ == "__main__":
    main()