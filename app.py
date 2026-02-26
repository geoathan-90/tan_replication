import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import tabu
import tan_solver


st.set_page_config(page_title="TAN Table Replicator", layout="wide")

st.title("TAN span-tension table (TAN.EXE replication)")

# --- Load conductors from PARAM1.DAT ---
param1_candidates = [
    Path("PARAM1.DAT"),
    Path("PRM") / "PARAM1.DAT",
    Path(__file__).resolve().parent / "PARAM1.DAT",
]
conductors = None
param1_path = None
for cand in param1_candidates:
    if cand.exists():
        try:
            conductors = tabu.load_param1(str(cand))
            param1_path = cand
            break
        except Exception:
            pass

if conductors is None:
    st.warning("PARAM1.DAT not found. Using a minimal placeholder conductor (you can still run with manual inputs).")
    conductors = [
        tabu.Conductor(ityp=0, name="Manual", tensions=np.array([2000, 2000, 2000, 2000, 2000], dtype=float), w=1.0, sec=1e-4, E=1e11)
    ]
else:
    st.caption(f"Loaded conductor parameters from: {param1_path}")

# --- Inputs ---
with st.sidebar:
    st.header("Inputs")

    conductor_names = [f"{c.ityp}: {c.name}" for c in conductors]
    idx = st.selectbox("Conductor type (ityp)", list(range(len(conductors))), format_func=lambda i: conductor_names[i])
    cond = conductors[idx]

    st.markdown("**Span geometry**")
    spans_str = st.text_area(
        "Spans (m), comma-separated",
        value="100, 120, 95",
        help="Enter span lengths as comma-separated numbers. Example: 100,120,95",
        height=80,
    )

    mode = st.radio("Heights input", ["Tower heights (N+1)", "Height differences dh (N)"], index=0)
    if mode == "Tower heights (N+1)":
        heights_str = st.text_area(
            "Tower heights (m), comma-separated",
            value="0, 5, 2, 6",
            help="Enter N+1 heights. dh[i] = h[i+1] - h[i].",
            height=80,
        )
    else:
        heights_str = st.text_area(
            "dh per span (m), comma-separated",
            value="5, -3, 4",
            help="Enter N values where dh[i] corresponds to span i height difference.",
            height=80,
        )

    st.markdown("**Solver**")
    solver_method = st.selectbox(
        "Solver method",
        [
            "TAN-style (C-compatible)",
            "Bisection",
            "Newton (safeguarded)",
            "Bruteforce",
        ],
        index=0,
    )

    bruteforce_step = None
    if solver_method == "Bruteforce":
        bruteforce_step = st.number_input("Bruteforce step (kg)", min_value=0.001, max_value=50.0, value=0.1)

    atol = st.number_input("Length tolerance (m)", min_value=1e-6, max_value=0.1, value=0.001, format="%.6f")

    st.markdown("**Conductor parameters (from PARAM1.DAT)**")
    st.write({
        "w": cond.w,
        "sec": cond.sec,
        "E": cond.E,
        "T_ref vector": cond.tensions.tolist(),
    })


def _parse_csv_floats(s: str):
    parts = [p.strip() for p in s.replace("\n", ",").split(",") if p.strip()]
    return [float(p) for p in parts]


def build_tables(spans, dh, cond: tabu.Conductor, solver_method: str, atol_m: float, bruteforce_step: float | None):
    spans = np.asarray(spans, dtype=float)
    dh = np.asarray(dh, dtype=float)

    tables = []
    for k in range(tabu.THERM):
        T_ref = float(cond.tensions[k])

        if solver_method == "TAN-style (C-compatible)":
            tor = tan_solver.solve_horizontal_tensions_tan_style(spans, dh, cond.w, T_ref, atol_m=atol_m)
        elif solver_method == "Bisection":
            tor = tabu.solve_horizontal_tensions_bisect(spans, dh, cond.w, T_ref, atol_m=atol_m)
        elif solver_method == "Newton (safeguarded)":
            tor = tabu.solve_horizontal_tensions_newton(spans, dh, cond.w, T_ref, atol_m=atol_m, safeguard=True)
        else:
            step = bruteforce_step if bruteforce_step is not None else 0.1
            tor = tabu.solve_horizontal_tensions_bruteforce(spans, dh, cond.w, T_ref, step=step, atol_m=atol_m)

        tor = np.asarray(tor, dtype=float)
        Tref_vec = np.full_like(tor, T_ref, dtype=float)

        sag_ref = tabu.sags_tan(Tref_vec, spans, dh, cond.w)
        sag_sol = tabu.sags_tan(tor, spans, dh, cond.w)
        dda = tabu.diorthosi_tan(tor, Tref_vec, spans, dh, cond.w, cond.sec, cond.E)

        df = pd.DataFrame({
            "span_idx": np.arange(1, len(spans) + 1),
            "span_m": spans,
            "dh_m": dh,
            "T_ref": Tref_vec,
            "tor": tor,
            "sag_ref": sag_ref,
            "sag_sol": sag_sol,
            "dda": dda,
        })

        tables.append((k + 1, df))

    return tables


def build_xlsx_bytes(tables):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for temp_idx, df in tables:
            sheet_name = f"Temp_{temp_idx}"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()


# --- Run ---
try:
    spans = _parse_csv_floats(spans_str)
    if len(spans) < 1:
        st.error("Please enter at least one span length.")
        st.stop()

    if mode == "Tower heights (N+1)":
        heights = _parse_csv_floats(heights_str)
        if len(heights) != len(spans) + 1:
            st.error(f"Expected N+1 heights ({len(spans)+1}), but got {len(heights)}.")
            st.stop()
        dh = [heights[i + 1] - heights[i] for i in range(len(spans))]
    else:
        dh = _parse_csv_floats(heights_str)
        if len(dh) != len(spans):
            st.error(f"Expected N dh values ({len(spans)}), but got {len(dh)}.")
            st.stop()

    tables = build_tables(spans, dh, cond, solver_method, atol, bruteforce_step)

    st.subheader("Results")
    tabs = st.tabs([f"Temp {i}" for i, _ in tables])
    for tab, (temp_idx, df) in zip(tabs, tables):
        with tab:
            st.dataframe(df, use_container_width=True)

    xlsx = build_xlsx_bytes(tables)
    st.download_button(
        "Download XLSX",
        data=xlsx,
        file_name="tan_table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Also provide a combined CSV (stacked, with temp index)
    all_df = []
    for temp_idx, df in tables:
        dft = df.copy()
        dft.insert(0, "temp_idx", temp_idx)
        all_df.append(dft)
    combined = pd.concat(all_df, ignore_index=True)
    st.download_button(
        "Download CSV (all temps)",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name="tan_table_all_temps.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Error: {e}")
