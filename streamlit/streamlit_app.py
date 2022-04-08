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

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("static/results/default_results.csv", index_col=0)
if "input_data" not in st.session_state:
    st.session_state.input_data = pd.read_csv("static/default_inputs.csv", index_col=0)


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

    st.markdown("### Einstellungen")

    # Embargo
    import_stop_date, reduction_import_russia = se.setting_embargo()

    # Compensation - Demand reduction
    (
        demand_reduction_date,
        red_ind_dem,
        red_elec_dem,
        red_ghd_dem,
        red_dom_dem,
        red_exp_dem,
        add_lng_import,
        lng_increase_date,
        add_pl_import,
    ) = se.setting_compensation()

    st.markdown("### Status Quo")

    # Supply
    se.setting_statusQuo_supply(expanded=False)

    # Demand
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


def get_scen_code(val_list):
    res = ""
    for s in val_list:
        try:
            s = int(100 * s)
        except:
            pass
        s = str(s)
        rep_list = [":", "-", ".", " "]
        for rep in rep_list:
            s = s.replace(rep, "")
        res += s
    return res


scen_code = get_scen_code(
    [
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
        import_stop_date,
        demand_reduction_date,
        lng_increase_date,
        reduction_import_russia,
        total_lng_import,
        add_lng_import,
        add_pl_import,
    ]
)
default_scen_code = "419000608001752009260042100151500111000988001320881420220416000000202203160000002022050100000010091400908000"
# st.write(scen_code)

st.markdown("## Optimierungsergebnisse")
start_opti = False

if scen_code != default_scen_code:
    start_opti = st.button("Optimierung ausführen")

show_results = True
if start_opti:
    # Start new calculation
    st.session_state.df, st.session_state.input_data = se.start_optimization(
        demand_reduction_date,
        import_stop_date,
        lng_increase_date,
        add_lng_import,
        add_pl_import,
        red_ind_dem,
        red_elec_dem,
        red_ghd_dem,
        red_dom_dem,
        red_exp_dem,
        reduction_import_russia,
    )
elif scen_code == default_scen_code:
    # Load default results
    pass
else:
    show_results = False


if show_results:
    se.plot_optimization_results(st.session_state.df)
    short_hash = int(abs(hash(scen_code)))
    st.download_button(
        "💾 Optimierungsergebnisse herunterladen",
        st.session_state.df.to_csv(),
        file_name=f"Optimierungsergebnisse_{short_hash}.csv",
        mime="text/csv",
    )

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
