# %%
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import datetime
import utils as ut
import optimization as opti
import base64
import os
import requests
import streamlit.components.v1 as components
import httplib2

# from ga import get_ga_values

# spacial scope

import streamlit_elements as se


# Default Dates
if "demand_reduction_date" not in st.session_state:
    st.session_state.demand_reduction_date = datetime.datetime.now()

if "import_stop_date" not in st.session_state:
    st.session_state.import_stop_date = datetime.datetime.now() + datetime.timedelta(
        weeks=4
    )

if "lng_increase_date" not in st.session_state:
    st.session_state.lng_increase_date = datetime.datetime.now() + datetime.timedelta(
        weeks=6
    )

if "spacial_scope" not in st.session_state:
    st.session_state.spacial_scope = "EU"  # EU DE

# status_quo_data = se.StatusQuoData(st.session_state.spacial_scope)
# status_quo_data = se.get_status_quo_data(st.session_state.spacial_scope)

# Default inputs
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("static/results/default_results.csv", index_col=0)
if "input_data" not in st.session_state:
    st.session_state.input_data = pd.read_csv("static/default_inputs.csv", index_col=0)


# Streamlit App
icon_dict = {"EU": "🇪🇺", "DE": "🇩🇪"}
st.set_page_config(
    page_title="No Stream",
    page_icon="🇪🇺",  # icon_dict.get(st.session_state.spacial_scope),
    layout="wide",
    initial_sidebar_state="expanded",  # wide centered
)


hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("# No Stream: Erdgas Energy Dashboard")

st.markdown("### Sichere Energie für Europa (EU27) ohne russische Erdgasimporte")


with st.sidebar:
    # Logo
    se.centered_fzj_logo()
    st.text("")

    # Settings
    st.markdown("### Einstellungen")

    # Spacial scope
    status_quo_data = se.setting_spacial_scope(allow_region_selection=False)

    # Embargo
    reduction_import_russia = se.setting_embargo()

    # Compensation
    (
        red_ind_dem,
        red_elec_dem,
        red_ghd_dem,
        red_dom_dem,
        red_exp_dem,
        add_lng_import,
        add_pl_import,
    ) = se.setting_compensation(status_quo_data)

    # Storage
    # if "reserve_dates" not in st.session_state:
    st.session_state.reserve_dates = status_quo_data.reserve_dates
    # if "reserve_soc_val" not in st.session_state:
    st.session_state.reserve_soc_val_abs = status_quo_data.reserve_soc_val_abs

    consider_gas_reserve = se.setting_storage(status_quo_data)

    # Status Quo
    st.markdown("### Status Quo")

    # Supply
    se.setting_statusQuo_supply(status_quo_data, expanded=False)

    # Demand
    se.setting_statusQuo_demand(status_quo_data)

    # Further links and information
    st.text("")
    se.sidebar_further_info()


# Energiebilanz

# Embargo und Kompensation
cols = st.columns(2)

# Importlücke
se.plot_import_gap(
    reduction_import_russia,
    red_exp_dem,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    add_lng_import,
    add_pl_import,
    status_quo_data,
    streamlit_object=cols[0],
)

# Stauts Quo
se.plot_status_quo(status_quo_data, streamlit_object=cols[1])

se.message_embargo_compensation(
    add_lng_import,
    add_pl_import,
    reduction_import_russia,
    red_exp_dem,
    red_elec_dem,
    red_ind_dem,
    red_ghd_dem,
    red_dom_dem,
    status_quo_data,
)

scen_code = ut.get_scen_code(
    st.session_state.spacial_scope,
    status_quo_data.total_ng_import,
    status_quo_data.total_lng_import,
    status_quo_data.total_ng_production,
    status_quo_data.total_pl_import_russia,
    status_quo_data.total_domestic_demand,
    status_quo_data.total_ghd_demand,
    status_quo_data.total_electricity_demand,
    status_quo_data.total_industry_demand,
    status_quo_data.total_exports_and_other,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    red_exp_dem,
    st.session_state.import_stop_date,
    st.session_state.demand_reduction_date,
    st.session_state.lng_increase_date,
    reduction_import_russia,
    add_lng_import,
    add_pl_import,
    consider_gas_reserve,
)


st.markdown("## Optimierungsergebnisse")
start_opti = st.button("Optimierung ausführen")


if start_opti:
    # Optimization
    st.session_state.df, st.session_state.input_data = se.start_optimization(
        add_lng_import,
        add_pl_import,
        red_ind_dem,
        red_elec_dem,
        red_ghd_dem,
        red_dom_dem,
        red_exp_dem,
        reduction_import_russia,
        consider_gas_reserve,
        status_quo_data,
    )

    # Plotting
    se.plot_optimization_results(st.session_state.df, status_quo_data)
    short_hash = int(abs(hash(scen_code)))
    st.download_button(
        "💾 Optimierungsergebnisse herunterladen",
        st.session_state.df.to_csv(),
        file_name=f"Optimierungsergebnisse_{short_hash}.csv",
        mime="text/csv",
    )

    # Download
    st.download_button(
        "💾 Input-Daten herunterladen",
        st.session_state.input_data.to_csv(),
        file_name=f"Input_Daten_{short_hash}.csv",
        mime="text/csv",
    )

st.text("")
st.text("")
se.centered_fzj_logo()
st.text("")
