"""tan_solver.py

Implements the *exact* outer-loop solver strategy used by TAN.EXE (Turbo C):
- Start from H0 = T_ref
- Use dt = 99
- Adjust H0 by +/- dt based on the sign of the length error
- Halve dt only when the adjustment direction flips
- Stop when |sl - slt| <= PROSEGISH

The forward propagation and length model are imported from tabu.py and executed
using float32 wherever practical.
"""

from __future__ import annotations

import numpy as np

from tabu import (
    PROSEGISH,
    forward_tensions_tan,
    total_length_tan,
)


def solve_horizontal_tensions_tan_style(
    spans_m,
    dh_m,
    w,
    T_ref,
    atol_m: float = PROSEGISH,
    dt0: float = 99.0,
    max_iter: int = 9999,
):
    """Return per-span horizontal tensions (tor[]) using TAN.EXE's dt-halving scheme."""

    # Reference total length using constant tension = T_ref (same as TAN's sl)
    spans_m = np.asarray(spans_m, dtype=np.float32)
    dh_m = np.asarray(dh_m, dtype=np.float32)
    w = float(w)
    T_ref = float(T_ref)

    sl = total_length_tan(spans_m, dh_m, w, np.full(spans_m.shape[0], T_ref, dtype=np.float32))

    tor0 = T_ref
    dt = float(dt0)
    iso = 0

    for it in range(max_iter):
        tor = forward_tensions_tan(spans_m, dh_m, w, tor0)
        slt = total_length_tan(spans_m, dh_m, w, tor)

        if abs(sl - slt) <= atol_m:
            return tor

        if slt <= sl:
            if it == 0:
                iso = -1
            isn = -1
            if isn != iso:
                dt /= 2.0
            tor0 -= dt
            iso = isn
        else:
            if it == 0:
                iso = 1
            isn = 1
            if isn != iso:
                dt /= 2.0
            tor0 += dt
            iso = isn

    # Return best-effort result after max_iter
    return forward_tensions_tan(spans_m, dh_m, w, tor0)
