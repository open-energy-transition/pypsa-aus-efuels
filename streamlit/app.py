# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Streamlit application to interactively manage a PyPSA-Earth network,
adjust some key economic parameters, and run optimizations afterwards
to assess the impact of the adjustments on the network's costs.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pypsa
from importlib.metadata import version
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import streamlit as st

solver_name = "highs"

# read current values and provide default values if non exist yet
# "lt": lifetime; "cc": capital_cost; "mc": marginal_cost; "dr": discount_rate; "label": label for UI
tech_data: dict[str, dict[str, int | str]] = {
    "solar": {"lt": 25, "cc": 750, "mc": 1, "dr": 0.05, "label": "Solar PV"},
    "solar rooftop": {"lt": 25, "cc": 1250, "mc": 1, "dr": 0.05, "label": "Solar PV Rooftop"},
    "onwind": {"lt": 20, "cc": 1500, "mc": 2, "dr": 0.05, "label": "Onshore Wind"},
    "offwind-ac": {"lt": 20, "cc": 3000, "mc": 4, "dr": 0.05, "label": "Offshore Wind (AC)"},
    "offwind-dc": {"cc": 3500, "mc": 6, "dr": 0.05, "label": "Offshore Wind (DC)"},
    "electrolysis": { "cc":  750, "mc": 1, "dr": 0.05, "label": "Electrolysis" },
}

load_data: dict[str, dict[str, int | str]] = {
    "demand": {"multiplier": 1, "label": "Electricity Demand"},
    "electricity": {"multiplier": 1, "label": "Electricity"},
    "hydrogen": {"multiplier": 1, "label": "Hydrogen"},
    "ammonia": {"multiplier": 1, "label": "Ammonia"},
    "methanol": {"multiplier": 1, "label": "Methanol"},
}

# -------------------- Helper functions --------------------
def get_week_per_month_snapshots(
    network: pypsa.Network, start_day: int = 1
) -> np.ndarray:
    sns_all = network.snapshots
    periodic_index = sns_all[
        (sns_all.strftime("%d").astype(int) <= (start_day + 6))
        & (sns_all.strftime("%d").astype(int) >= start_day)
    ]
    return periodic_index

def replace_nan(x: float, def_value: int = 0):
    return x if not np.isnan(x) else def_value

def round_multiple(number: float, multiple: float = 50.0):
    return multiple * round(number / multiple)

def print_statistics(n: pypsa.Network):
    if st.session_state.n is not None:
        st.header("Network Statistics")
        for c in n.components.keys() - ["Network", "SubNetwork"]:
            if len(getattr(n, n.components[c]["list_name"])):
                st.write(f"**{c}**: {len(getattr(n, n.components[c]['list_name']))}")
        st.write(f"**Snapshot**: {len(n.snapshots)}")

title = "AUS eFuels"
st.set_page_config(layout="wide", page_title=title)
st.set_page_config(page_title=f"{title} UI", layout="wide")
st.title(f"{title} Interactive Manager")

if "n" not in st.session_state:
    st.session_state.n = None
if "opt_runs" not in st.session_state:
    st.session_state.opt_runs = 0
if "network_loaded" not in st.session_state:
    st.session_state.network_loaded = False
if "results" not in st.session_state:
    st.session_state.results = None

# --- SIDEBAR ---
with st.sidebar:
    st.sidebar.header("Networks")
    with st.expander("Selected PyPSA Network", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a PyPSA NetCDF file", type=["nc"], max_upload_size=20  # 20 MB limit
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

        # Cleanup the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if st.sidebar.button("Load Example 'scigrid_de'"):
        n = pypsa.examples.scigrid_de()
        st.session_state.n = n
        st.success("Example network loaded!")

    if st.session_state.network_loaded:
        print_statistics(st.session_state.n)

    st.write("---")
    for pkg in ["pypsa", "linopy", "highspy"]:
        pkg_version = version(pkg)
        st.write(f"{pkg}: {pkg_version}")

# --- Tabs
tabs = [
    "| - Welcome", 
    "| - Economic Parameters", 
    "| - Demand Parameters ", 
    "| - Optimization", 
    "| - View Results",
]
t_welcome, t_economic, t_demand, t_optimization, t_results = st.tabs(
    tabs, 
    on_change="rerun"
)

# --- TAB WeLCOME
with t_welcome:
    st.subheader("Welcome to the PyPSA-AUS-eFuels Interactive Manager!")
    st.write(
        """
        This application allows you to load a PyPSA network, adjust key economic parameters, and run optimizations to see how those adjustments impact the network's costs and performance.
        
        Use the sidebar to load your network and set project targets. Then, navigate through the tabs to manage different aspects of your project.

        This application has been developed during a project between Open Energy Transition (OET) and 
        """
    )

# --- TAB ECONOMIC PARAMETERS
with t_economic:
    if st.session_state.n is None:
        st.info("Please load a network ...")
        st.write(
            """
            After loading a network, you are able to adjust the economic parameters.
            """)
    else:
        st.header("Economic Parameters")

        n = st.session_state.n
        g = n.generators
        with st.expander("Selected Economic Parameters", expanded=True):
            st.write("Choose Capital Cost and Marginal Cost to be used for your case:")

            col1, col2, col3 = st.columns(3, vertical_alignment="center")
            with col2:
                st.write("**Percent (%)**")

            col1, col2, col3 = st.columns(3, vertical_alignment="center")
            with col1:
                st.write("**Discount Rate**")
            with col2:
                discount_rate_slider = st.slider(
                    "Discount Rate",
                    min_value=1,
                    max_value=20,
                    value=8,
                    label_visibility="collapsed",
                )
            dr = discount_rate_slider / 100

            old_cc = {}
            old_mc = {}
            new_cc = {}
            new_mc = {}
            for d in tech_data:
                old_cc[d] = replace_nan(
                    g.loc[
                        g.carrier.str.startswith(d), "capital_cost"
                    ].mean(),
                    tech_data[d]["cc"] * dr * 1000,
                )
                old_mc[d] = replace_nan(
                    g.loc[
                        g.carrier.str.startswith(d), "marginal_cost"
                    ].mean(),
                    tech_data[d]["mc"],
                )

            col1, col2, col3 = st.columns(3, vertical_alignment="center")
            with col2:
                st.write("**Capital Cost (kAUD/MW)**")
            with col3:
                st.write("**Marginal Cost (AUD/MWh)**")

            for d in tech_data:
                col1, col2, col3 = st.columns(3, vertical_alignment="center")
                with col1:
                    st.write(f"**{tech_data[d]['label']}**")
                with col2:
                    new_cc[d] = st.select_slider(
                        label=f"cc_{tech_data[d]['label']}",
                        label_visibility="collapsed",
                        options=[
                            *range(100, 8000, 50)
                        ],  # fixed range for better UX with rounding to 50k AUD/MW steps
                        value=round_multiple(old_cc[d] / dr / 1000, 50),
                    )
                with col3:
                    new_mc[d] = st.slider(
                        label=f"mc_{tech_data[d]['label']}",
                        label_visibility="collapsed",
                        min_value=0,
                        max_value=20,
                        value=tech_data[d]["mc"],
                    )

        if st.button("Apply New Costs"):
            g["discount_rate"] = g.get("discount_rate", dr)
            g.loc[g.discount_rate.astype(str).isnull(), "discount_rate"] = dr

            for d in tech_data:
                if len(g.carrier[g.carrier == d]):
                    g.loc[
                        g.carrier.str.startswith(d), "discount_rate"
                    ] = dr
                    g.loc[
                        g.carrier.str.startswith(d), "capital_cost"
                    ] = (new_cc[d] * 1000 * dr)
                    g.loc[
                        g.carrier.str.startswith(d), "marginal_cost"
                    ] = new_mc[d]

            st.success("Updated details for mentioned technologies ...")
            st.write(g[["capital_cost", "marginal_cost", "discount_rate"]])

# --- TAB DEMAND PARAMETERS
with t_demand:
    if st.session_state.n is None:
        st.info("Please load a network ...")
        st.write(
            """
            After loading a network, you are able to adjust the demand parameters.
            """)
    else:
        st.header("Demand Parameters")
        n = st.session_state.n
        with st.expander("Selected Demand Parameters", expanded=True):
            st.write("Choose Load Multipliers to be used for your case:")
            old_multiplier = {}
            new_multiplier = {}
            for l in load_data:
                if l in n.loads.index:
                    n.loads.at[l, 'p_set'] = n.loads_t.p_set[l].sum() / 1e6
                    if 'q_set' in n.loads_t and l in n.loads_t.q_set.index and len(n.loads_t.q_set[l]):
                        n.loads.at[l, 'q_set'] = n.loads_t.q_set[l].sum()
                    old_multiplier[l] = replace_nan(
                        n.loads.loc[
                            n.loads.index.str.startswith(l), "p_set"
                        ].sum(),
                        load_data[l]["multiplier"]
                    )
                else:
                    old_multiplier[l] = 0

            col1, col2, col3 = st.columns(3, vertical_alignment="center")
            with col2:
                st.write("**Demand Multiplier (* 1e6)**")

            for l in load_data:
                col1, col2, col3 = st.columns(3, vertical_alignment="center")
                with col1:
                    st.write(f"**{load_data[l]['label']}**")
                with col2:
                    new_multiplier[l] = st.select_slider(
                        label=f"Demand Multiplier {l}",
                        label_visibility="collapsed",
                        options=[
                            *range(0, 501, 1)
                        ],  # fixed range for better UX with rounding to full percent
                        value=round_multiple(old_multiplier[l], 1.0),
                    )

        if st.button("Apply New Demand"):
            n.loads["carrier"] = n.loads.get("carrier", "electricity")
            for l in load_data:
                if l in n.loads.index:
                    n.loads.loc[
                        n.loads.index.str.startswith(l), "p_set"
                    ] = new_multiplier[l] * 1e6

            st.success("Updated details for mentioned carriers ...")
            st.write(n.loads[["carrier", "p_set"]])

# --- TAB OPTIMIZATION
with t_optimization:
    if st.session_state.n is None:
        st.info("Please load a network ...")
        st.write(
            """
            After loading a network, you are able to optimize the network.
            """)
    else:
        st.header("Run Optimization")
        with st.expander("Snapshot Options", expanded=True):
            run_mode = st.radio("Select the number of desired optimization snapshots:", 
                                ["Full Year (8760 h)", "Week per Month (2016 h)"], index=1)

        if st.button("Run LOPF"):
            n2 = n.copy()
            g = n2.generators

            if run_mode == "Week per Month (2016 h)":
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
                    include_objective_constant=False,
                )

            if status == "ok":
                # Show Results
                st.success(f"Optimization finished: {condition}")
                # st.metric("Total System Cost", f"${n2.objective:,.2f}")
                st.subheader("Results: Generation Dispatch")
                dispatch = n2.generators_t.p  # .sum()
                st.bar_chart(dispatch, y_label="MW")
                # TODO needs update :-)
                old_total_cost = 20*10e6 * 750 # assuming 100 AUD / unit
                new_total_cost = 20*10e6 * 0 + n2.objective # assuming 100 AUD / unit
                expanded_cap = n2.statistics.expanded_capacity().round(1)
                expanded_cap[('Economics', 'Old Annuity')] = round(old_total_cost / 1e6, 1) # convert to million AUD
                expanded_cap[('Economics', 'New Annuity')] = round(n2.objective / 1e6, 1) # convert to million AUD
                expanded_cap[('Economics', 'Savings')] = round((old_total_cost - new_total_cost)/1e6, 1) # convert to million AUD
                if st.session_state.results is None:
                    cap_df = expanded_cap.to_frame(name=f"run {st.session_state.opt_runs}")
                else:
                    cap_df = st.session_state.results.join(
                        expanded_cap.to_frame(name=f"run {st.session_state.opt_runs}")
                    )
                st.session_state.results = cap_df
                st.write(
                    """
                    Successfully optimized the network with the new parameters.
                    
                    Check the 'View Results' tab for details.
                    """)
            else:
                st.error(f"Solver failed: {condition}")

# ---TAB RESULTS
if t_results.open:
    with t_results:
        if st.session_state.results is not None:
            st.header("Result Details")
            df = st.session_state.results
            df = df[~df.index.get_level_values(0).str.contains('Economics')]
            st.dataframe(df.style.format("{:.1f}"))
            #
            st.header("Economic Comparison")
            df = st.session_state.results
            df = df / 1e3  # convert to million AUD
            df = df[df.index.get_level_values(0).str.contains('Economics')].round(1)
            df = df.reset_index().drop(columns=['component'])
            df = df.set_index('carrier')
            st.bar_chart(
                df.T, 
                y_label="Runs", 
                x_label="Million AUD", 
                horizontal=True
            )
        else:
            st.info("Please load a network and run an optimization to see results here ...")
            st.write(
                """
                After running an optimization, you will see a detailed breakdown of the expanded capacities and economic outcomes for each technology, allowing you to assess the impact of your parameter adjustments on the network's performance and costs.
                """)
