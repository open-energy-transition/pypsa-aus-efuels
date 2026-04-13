# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Streamlit application to interactively manage PyPSA-Earth networks,
adjust some key economic parameters, and run optimizations afterwards.
"""

import os
import tempfile

import numpy as np
import pypsa

import streamlit as st

solver_name = "highs"


# -------------------- Helper functions --------------------
def get_week_per_month_snapshots(network, start_day=1):
    sns_all = network.snapshots
    periodic_index = sns_all[
        (sns_all.strftime("%d").astype(int) <= (start_day + 6))
        & (sns_all.strftime("%d").astype(int) >= start_day)
    ]
    return periodic_index


def replace_nan(x, def_value=0):
    return x if not np.isnan(x) else def_value


def round_multiple(number, multiple=50):
    return multiple * round(number / multiple)


def print_statistics():
    if st.session_state.n is not None:
        st.header("📊 Network Statistics")
        for c in n.components.keys() - ["Network", "SubNetwork"]:
            if len(getattr(n, n.components[c]["list_name"])):
                st.write(f"**{c}**: {len(getattr(n, n.components[c]['list_name']))}")
        st.write(f"**Snapshot**: {len(n.snapshots)}")


def toggle_expander():
    st.session_state.expander_open = not st.session_state.expander_open


st.set_page_config(page_title="PyPSA-Earth UI", layout="wide")

# --- SESSION STATE ---
if "n" not in st.session_state:
    st.session_state.n = None
if "network_loaded" not in st.session_state:
    st.session_state.network_loaded = False
if "opt_runs" not in st.session_state:
    st.session_state.opt_runs = 0
if "expander_open" not in st.session_state:
    st.session_state.expander_open = False
if "results" not in st.session_state:
    st.session_state.results = None

# --- SIDEBAR ---
with st.sidebar:
    st.sidebar.header("Networks")

    uploaded_file = st.file_uploader(
        "Choose a PyPSA NetCDF file", type=["nc"], max_upload_size=10  # 10 MB limit
    )
    if uploaded_file is not None and st.session_state.network_loaded is False:
        # PyPSA needs a file path, so we save the uploaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # 2. Load the Network
        with st.spinner("Loading network..."):
            n = pypsa.Network(tmp_path)
            st.session_state.n = n
            st.session_state.network_loaded = True
            st.success("Network loaded successfully!")
            print_statistics()

        # Cleanup the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if st.sidebar.button("Load Example 'model_energy'"):
        n = pypsa.examples.scigrid_de()
        st.session_state.n = n
        st.success("Example network loaded!")
        print_statistics()

# --- MAIN PAGE ---
st.title("PyPSA-AUS-eFuels Interactive Manager")

if st.session_state.n is None:
    st.info("Please load a network ...")
else:
    st.header("💰 Economic Adjustments")

    with st.expander("View Economic Adjustment Options"):
        st.write("Choose Capital Cost and Marginal Cost to be used for your case:")
        n = st.session_state.n

        col1, col2, col3, col4 = st.columns(4, vertical_alignment="center")
        with col2:
            st.write("**Percent (%)**")

        col1, col2, col3, col4 = st.columns(4, vertical_alignment="center")
        with col1:
            st.write("**Discount Rate**")
        with col2:
            discount_rate = st.slider(
                "Discount Rate",
                min_value=1,
                max_value=20,
                value=8,
                label_visibility="collapsed",
            )
        dr = discount_rate / 100

        # read current values and provide default values if non exist yet
        # unique carriers: 'offwind-ac', 'offwind-dc', 'onwind', 'solar', 'solar rooftop'
        # cc ~ capital_cost; mc ~ marginal_cost
        data = {
            "solar": {"cc": 750, "mc": 1, "label": "Solar PV"},
            "onwind": {"cc": 1500, "mc": 2, "label": "Onshore Wind"},
            "offwind-ac": {"cc": 3000, "mc": 4, "label": "Offshore Wind (AC)"},
            "offwind-dc": {"cc": 3500, "mc": 6, "label": "Offshore Wind (DC)"},
            #    'electrolysis': { 'cc':  750, 'mc': 1, 'label': "Electrolysis" },
        }
        old_cc = {}
        old_mc = {}
        new_cc = {}
        new_mc = {}
        for d in data:
            old_cc[d] = replace_nan(
                n.generators.loc[
                    n.generators.carrier.str.startswith(d), "capital_cost"
                ].mean(),
                data[d]["cc"] * dr * 1000,
            )
            old_mc[d] = replace_nan(
                n.generators.loc[
                    n.generators.carrier.str.startswith(d), "marginal_cost"
                ].mean(),
                data[d]["mc"],
            )

        col1, col2, col3, col4 = st.columns(4, vertical_alignment="center")
        with col2:
            st.write("**Capital Cost (kAUD/MW)**")
        with col3:
            st.write("**Marginal Cost (AUD/MWh)**")

        for d in data:
            col1, col2, col3, col4 = st.columns(4, vertical_alignment="center")
            with col1:
                st.write(f"**{data[d]['label']}**")
            with col2:
                new_cc[d] = st.select_slider(
                    label=f"Capital Cost {data[d]['label']}",
                    label_visibility="collapsed",
                    options=[
                        *range(100, 8000, 50)
                    ],  # fixed range for better UX with rounding to 50k AUD/MW steps
                    value=round_multiple(old_cc[d] / dr / 1000, 50),
                )
            with col3:
                new_mc[d] = st.slider(
                    label=f"Marginal Cost {data[d]['label']}",
                    label_visibility="collapsed",
                    min_value=0,
                    max_value=20,
                    value=data[d]["mc"],
                )

        if st.button("Apply New Costs", on_click=toggle_expander):
            if "discount_rate" not in n.generators.columns:
                n.generators["discount_rate"] = dr

            n.generators.loc[n.generators.discount_rate.isnull(), "discount_rate"] = dr
            for d in data:
                if len(n.generators.carrier[n.generators.carrier == d]):
                    n.generators.loc[
                        n.generators.carrier.str.startswith(d), "discount_rate"
                    ] = dr
                    n.generators.loc[
                        n.generators.carrier.str.startswith(d), "capital_cost"
                    ] = (new_cc[d] * 1000 * dr)
                    n.generators.loc[
                        n.generators.carrier.str.startswith(d), "marginal_cost"
                    ] = new_mc[d]

            st.success("Updated details for mentioned technologies ...")
            st.write(n.generators[["capital_cost", "marginal_cost", "discount_rate"]])

    # --- D) OPTIMIZATION ---
    st.header("⚙️ Run Optimization")
    run_mode = st.radio("Select Snapshots", ["Full Year", "Week per Month (Heuristic)"])

    if st.button("Run LOPF"):
        n2 = n.copy()

        if run_mode == "Week per Month (Heuristic)":
            sns_before = len(n2.snapshots)
            sns_subset = get_week_per_month_snapshots(n2, start_day=10)
            sns_after = len(sns_subset)
            n2.set_snapshots(sns_subset)
            n2.snapshot_weightings = n2.snapshot_weightings * sns_before / sns_after

        st.info(f"Optimizing for {len(n2.snapshots)} snapshots.")
        st.session_state.opt_runs += 1
        with st.spinner("Solving Network ..."):
            n2.consistency_check()
            status, condition = n2.optimize(
                solver_name=solver_name,
                assign_all_duals=True,
                include_objective_constant=True,
            )

        if status == "ok":
            # Show Results
            st.success(f"Optimization finished: {condition}")
            st.metric("Total System Cost", f"${n2.objective:,.2f}")
            st.subheader("Results: Generation Dispatch")
            dispatch = n2.generators_t.p  # .sum()
            st.bar_chart(dispatch, y_label="MW")
            expanded_cap = n2.statistics.expanded_capacity().round(1)
            if st.session_state.results is None:
                cap_df = expanded_cap.to_frame(name=f"run {st.session_state.opt_runs}")
            else:
                cap_df = st.session_state.results.join(
                    expanded_cap.to_frame(name=f"run {st.session_state.opt_runs}")
                )

            st.subheader("Expanded Capacity Data")
            st.dataframe(cap_df)
            st.session_state.results = cap_df

        else:
            st.error(f"Solver failed: {condition}")
