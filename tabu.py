"""
tabu.py

Core computations for the TAN span-tension table:
- Forward span tension propagation (Turbo-C TAN.EXE compatible form)
- Multiple H0 solvers (bruteforce, bisection, safeguarded Newton)
- TAN-compatible sag and "diorthosi" (dda) calculations
- PARAM1.DAT parsing (binary struct dump from TAN)

This module is intended to be used by app.py (Streamlit) and by tan_solver.py (TAN-style step-halving solver).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math
import struct

import numpy as np


PROSEGISH = 0.001  # TAN: tolerance on total length difference (same length units as spans)
THERM = 5          # number of temperature points in PARAM1.DAT tension tables
AGVGOI = 10         # number of conductor types in PARAM1.DAT tension tables


@dataclass(frozen=True)
class Conductor:
    """One conductor option (matches TAN's ityp index 0..9)."""

    ityp: int
    name: str
    tensions: np.ndarray  # shape (5,), float64
    w: float
    sec: float
    E: float


def _decode_cp737(b: bytes) -> str:
    # PARAM1.DAT strings are typically OEM Greek (CP737)
    try:
        return b.decode("cp737", errors="replace").rstrip("\x00").strip()
    except Exception:
        return b.decode("latin-1", errors="replace").rstrip("\x00").strip()


def load_param1(path: str = "PARAM1.DAT") -> List[Conductor]:
    """Parse PRM/PARAM1.DAT (binary struct param dump from Turbo C) into Conductor objects."""

    with open(path, "rb") as f:
        data = f.read()

    if len(data) != 254:
        raise ValueError(f"Unexpected PARAM1.DAT size: {len(data)} bytes (expected 254).")

    off = 0
    xarakt_raw = data[off:off + 40]
    off += 40
    xarakt = [_decode_cp737(xarakt_raw[i * 8:(i + 1) * 8]) for i in range(5)]

    # tash[5], basan[2] (unused in our calculations)
    off += 10
    off += 4

    tq = struct.unpack_from("<" + "h" * (AGVGOI * THERM), data, off)
    off += 2 * (AGVGOI * THERM)
    tq = np.array(tq, dtype=np.float64).reshape((AGVGOI, THERM))

    # typos[5] (unused), baros[5], diatomh[5], m_elast[5]
    off += 20
    baros = np.array(struct.unpack_from("<5f", data, off), dtype=np.float64)
    off += 20
    diatomh = np.array(struct.unpack_from("<5f", data, off), dtype=np.float64)
    off += 20
    m_elast = np.array(struct.unpack_from("<5d", data, off), dtype=np.float64)
    off += 40

    conductors: List[Conductor] = []
    for ityp in range(AGVGOI):
        g = ityp // 2
        gname = xarakt[g] if xarakt[g] else f"Group {g}"
        mode = "A" if (ityp % 2 == 0) else "B"
        name = f"{gname} ({mode})"

        w = float(baros[g])
        sec = float(diatomh[g] / 10000.0)
        E = float(m_elast[g] * 1e10)

        conductors.append(
            Conductor(
                ityp=ityp,
                name=name,
                tensions=tq[ityp].copy(),
                w=w,
                sec=sec,
                E=E,
            )
        )

    return conductors


def _as_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def forward_tensions_tan(spans_m, dh_m, w, H0):
    """TAN-compatible forward propagation of horizontal tensions across spans."""

    a = _as_f32(spans_m)
    dh = _as_f32(dh_m)
    w = np.float32(w)
    H0 = np.float32(H0)

    n = a.size
    tor = np.empty(n, dtype=np.float32)
    tsp = np.empty(n, dtype=np.float32)
    tsa = np.empty(n, dtype=np.float32)
    at = np.empty(n, dtype=np.float32)
    f = np.empty(n, dtype=np.float32)

    tor[0] = H0
    xx = tor[0] / w
    at[0] = a[0] - 2.0 * dh[0] * xx / a[0]
    f[0] = at[0] * at[0] / (8.0 * xx)
    tsa[0] = tor[0] + w * f[0]
    tsp[0] = tor[0] + w * (f[0] + dh[0])

    for j in range(1, n):
        tsa[j] = tsp[j - 1]
        ax = dh[j] / a[j]
        aa = np.float32(0.5) * ax * ax + np.float32(1.0)
        bb = -tsa[j] - np.float32(0.5) * w * dh[j]
        cc = w * w * a[j] * a[j] / np.float32(8.0)
        disc = bb * bb - np.float32(4.0) * aa * cc
        if disc < 0:
            raise ValueError(f"Negative discriminant at span {j}: {float(disc)}")
        dd = np.float32(math.sqrt(float(disc)))
        tor[j] = (dd - bb) / (np.float32(2.0) * aa)

        xx = tor[j] / w
        at[j] = a[j] - 2.0 * dh[j] * xx / a[j]
        f[j] = at[j] * at[j] / (8.0 * xx)
        tsp[j] = tor[j] + w * (f[j] + dh[j])

    return tor


def total_length_tan(spans_m, dh_m, w, H):
    """TAN span length approximation."""

    a = _as_f32(spans_m)
    dh = _as_f32(dh_m)
    w = np.float32(w)
    H = _as_f32(H)

    b2 = a * a + dh * dh
    term = (a ** 4) * (w * w) / (np.float32(12.0) * (H * H))
    L = np.sqrt(b2 + term, dtype=np.float32)
    return float(np.sum(L, dtype=np.float32))


def sags_tan(tensions, spans_m, dh_m, w):
    """TAN print-time sag formula."""

    H = np.asarray(tensions, dtype=np.float64)
    a = np.asarray(spans_m, dtype=np.float64)
    dh = np.asarray(dh_m, dtype=np.float64)
    b = np.sqrt(a * a + dh * dh)
    xx = H / float(w)
    return b * a / (8.0 * xx) + b * (a / xx) ** 3 / 384.0


def diorthosi_tan(H_solution, T_ref, spans_m, dh_m, w, sec, E):
    """TAN da/dda (includes elastic-extension term)."""

    H = np.asarray(H_solution, dtype=np.float64)
    tt = np.asarray(T_ref, dtype=np.float64)
    a = np.asarray(spans_m, dtype=np.float64)
    dh = np.asarray(dh_m, dtype=np.float64)
    b = np.sqrt(a * a + dh * dh)

    aaa = (a ** 4) * (float(w) ** 2) / 12.0
    xc = np.sqrt(b * b + aaa / (H * H)) - np.sqrt(b * b + aaa / (tt * tt))
    da = (b ** 3) * (H - tt) / (float(sec) * float(E) * (a ** 2)) - xc
    return np.cumsum(da)


def solve_horizontal_tensions_bruteforce(spans_m, dh_m, w, T_ref, step=0.1, atol_m=PROSEGISH, max_iter=200000):
    """Walk H0 up/down by 'step' until total_length matches target within atol_m."""

    target = total_length_tan(spans_m, dh_m, w, np.full(len(spans_m), T_ref, dtype=np.float32))

    def err(H0_):
        H = forward_tensions_tan(spans_m, dh_m, w, H0_)
        return total_length_tan(spans_m, dh_m, w, H) - target, H

    H0 = float(T_ref)
    e0, H = err(H0)
    best = (abs(e0), H0, H)
    if abs(e0) <= atol_m:
        return H

    direction = -1.0 if e0 > 0 else 1.0

    for _ in range(max_iter):
        H0 += direction * step
        e, H = err(H0)
        if abs(e) < best[0]:
            best = (abs(e), H0, H)
        if abs(e) <= atol_m:
            return H
        if (e0 > 0 and e < 0) or (e0 < 0 and e > 0):
            direction *= -1.0
            step *= 0.5
        e0 = e

    return best[2]


def solve_horizontal_tensions_bisect(spans_m, dh_m, w, T_ref, atol_m=PROSEGISH, max_iter=200):
    """Bracketed bisection on H0 to match target length."""

    target = total_length_tan(spans_m, dh_m, w, np.full(len(spans_m), T_ref, dtype=np.float32))

    def F(H0):
        H = forward_tensions_tan(spans_m, dh_m, w, H0)
        return total_length_tan(spans_m, dh_m, w, H) - target

    lo = float(T_ref) * 0.5
    hi = float(T_ref) * 2.0
    flo = F(lo)
    fhi = F(hi)
    expand = 0
    while flo * fhi > 0 and expand < 50:
        lo *= 0.7
        hi *= 1.3
        flo = F(lo)
        fhi = F(hi)
        expand += 1

    if flo * fhi > 0:
        return solve_horizontal_tensions_bruteforce(spans_m, dh_m, w, T_ref, step=0.1, atol_m=atol_m)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = F(mid)
        if abs(fmid) <= atol_m:
            return forward_tensions_tan(spans_m, dh_m, w, mid)
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return forward_tensions_tan(spans_m, dh_m, w, 0.5 * (lo + hi))


def solve_horizontal_tensions_newton(spans_m, dh_m, w, T_ref, atol_m=PROSEGISH, max_iter=50, safeguard=True):
    """Safeguarded Newton (finite difference derivative) with bisection fallback."""

    target = total_length_tan(spans_m, dh_m, w, np.full(len(spans_m), T_ref, dtype=np.float32))

    def F(H0):
        H = forward_tensions_tan(spans_m, dh_m, w, H0)
        return total_length_tan(spans_m, dh_m, w, H) - target

    lo = float(T_ref) * 0.5
    hi = float(T_ref) * 2.0
    flo = F(lo)
    fhi = F(hi)
    expand = 0
    while flo * fhi > 0 and expand < 50:
        lo *= 0.7
        hi *= 1.3
        flo = F(lo)
        fhi = F(hi)
        expand += 1

    if flo * fhi > 0:
        return solve_horizontal_tensions_bisect(spans_m, dh_m, w, T_ref, atol_m=atol_m)

    x = float(T_ref)
    fx = F(x)

    for _ in range(max_iter):
        if abs(fx) <= atol_m:
            return forward_tensions_tan(spans_m, dh_m, w, x)

        h = max(1e-3, 1e-4 * abs(x))
        f1 = F(x + h)
        d = (f1 - fx) / h

        if d == 0 or not math.isfinite(d):
            x_new = 0.5 * (lo + hi)
        else:
            x_new = x - fx / d
            if safeguard and not (lo < x_new < hi):
                x_new = 0.5 * (lo + hi)

        x = x_new
        fx = F(x)

        if flo * fx <= 0:
            hi = x
            fhi = fx
        else:
            lo = x
            flo = fx

    return forward_tensions_tan(spans_m, dh_m, w, x)
