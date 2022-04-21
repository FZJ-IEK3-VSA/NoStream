#%%
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import datetime
import utils as ut
import optimization as opti
import base64
import os
import streamlit_elements as se


total_lng_import = 914
total_industry_demand = 1110
total_exports_and_other = 988
total_domestic_demand = 926
total_ghd_demand = 421
total_electricity_demand = 1515
total_ng_import = 4190
total_pl_import_russia = 1752
total_lng_import_russia = 160
total_ng_production = 608


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


# Default inputs
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("static/results/default_results.csv", index_col=0)
if "input_data" not in st.session_state:
    st.session_state.input_data = pd.read_csv("static/default_inputs.csv", index_col=0)


# Default SOC min
reserve_dates = [
    datetime.datetime(2022, 8, 1, 0, 0),
    datetime.datetime(2022, 9, 1, 0, 0),
    datetime.datetime(2022, 10, 1, 0, 0),
    datetime.datetime(2022, 11, 1, 0, 0),
    datetime.datetime(2023, 2, 1, 0, 0),
    datetime.datetime(2023, 5, 1, 0, 0),
    datetime.datetime(2023, 7, 1, 0, 0),
]
reserve_soc_val = [0.63, 0.68, 0.74, 0.80, 0.43, 0.33, 0.52]
storage_cap = 1100
reserve_soc_val = [x * storage_cap for x in reserve_soc_val]

if "reserve_dates" not in st.session_state:
    st.session_state.reserve_dates = reserve_dates
if "reserve_soc_val" not in st.session_state:
    st.session_state.reserve_soc_val = reserve_soc_val



### Streamlit App
st.set_page_config(
    page_title="No Stream",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded",  # wide centered
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
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

    ## Embargo
    reduction_import_russia = se.setting_embargo()

    ## Compensation
    (
        red_ind_dem,
        red_elec_dem,
        red_ghd_dem,
        red_dom_dem,
        red_exp_dem,
        add_lng_import,
        add_pl_import,
    ) = se.setting_compensation()

    ## Storage
    consider_gas_reserve = se.setting_storage()

    # Status Quo
    st.markdown("### Status Quo")

    ## Supply
    se.setting_statusQuo_supply(expanded=False)

    ## Demand
    se.setting_statusQuo_demand()

    # Further links and information
    st.text("")
    se.sidebar_further_info()


# Energiebilanz

# Embargo und Kompensation
cols = st.columns(2)

## Importlücke
se.plot_import_gap(
    reduction_import_russia,
    red_exp_dem,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    add_lng_import,
    add_pl_import,
    streamlit_object=cols[0],
)

# Stauts Quo
se.plot_status_quo(streamlit_object=cols[1])

se.message_embargo_compensation(
    add_lng_import,
    add_pl_import,
    reduction_import_russia,
    red_exp_dem,
    red_elec_dem,
    red_ind_dem,
    red_ghd_dem,
    red_dom_dem,
)

scen_code = ut.get_scen_code(
    total_ng_import,
    total_ng_production,
    total_pl_import_russia,
    total_domestic_demand,
    total_ghd_demand,
    total_electricity_demand,
    total_industry_demand,
    total_exports_and_other,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    red_exp_dem,
    st.session_state.import_stop_date,
    st.session_state.demand_reduction_date,
    st.session_state.lng_increase_date,
    reduction_import_russia,
    total_lng_import,
    add_lng_import,
    add_pl_import,
    consider_gas_reserve,
)

# default_scen_code = "419000608001752009260042100151500111000988001320881420220416000000202203160000002022050100000010091400908000"
# st.write(scen_code)

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
    )

    # Plotting
    se.plot_optimization_results(st.session_state.df)
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
