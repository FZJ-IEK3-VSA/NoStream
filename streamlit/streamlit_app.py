import pandas as pd
import streamlit as st
import datetime
import utils as ut
import streamlit_elements as se

# Default values
if "demand_reduction_date" not in st.session_state:
    st.session_state.demand_reduction_date = datetime.datetime.now()
if "import_stop_date" not in st.session_state:
    st.session_state.import_stop_date = datetime.datetime.now()
if "lng_increase_date" not in st.session_state:
    st.session_state.lng_increase_date = datetime.datetime.now()
if "spacial_scope" not in st.session_state:
    st.session_state.spacial_scope = "EU"

# 10 %
# if "red_ind_dem" not in st.session_state:
#     st.session_state.red_ind_dem = 0.06
# if "red_elec_dem" not in st.session_state:
#     st.session_state.red_elec_dem = 0.13
# if "red_ghd_dem" not in st.session_state:
#     st.session_state.red_ghd_dem = 0.06
# if "red_dom_dem" not in st.session_state:
#     st.session_state.red_dom_dem = 0.10

# 15 %
if "red_ind_dem" not in st.session_state:
    st.session_state.red_ind_dem = 0.09
if "red_elec_dem" not in st.session_state:
    st.session_state.red_elec_dem = 0.21
if "red_ghd_dem" not in st.session_state:
    st.session_state.red_ghd_dem = 0.09
if "red_dom_dem" not in st.session_state:
    st.session_state.red_dom_dem = 0.14

# 20 %
# if "red_ind_dem" not in st.session_state:
#     st.session_state.red_ind_dem = 0.12
# if "red_elec_dem" not in st.session_state:
#     st.session_state.red_elec_dem = 0.28
# if "red_ghd_dem" not in st.session_state:
#     st.session_state.red_ghd_dem = 0.12
# if "red_dom_dem" not in st.session_state:
#     st.session_state.red_dom_dem = 0.19

if "red_exp_dem" not in st.session_state:
    st.session_state.red_exp_dem = 0.0
if "add_lng_import" not in st.session_state:
    st.session_state.add_lng_import = 390
if "add_pl_import" not in st.session_state:
    st.session_state.add_pl_import = 0
if "reduction_import_russia" not in st.session_state:
    st.session_state.reduction_import_russia = 1
if "status_quo_data" not in st.session_state:
    st.session_state.status_quo_data = se.setting_spacial_scope(
        allow_region_selection=False
    )

# Default inputs
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("static/results/default_results.csv", index_col=0)
if "input_data" not in st.session_state:
    st.session_state.input_data = pd.read_csv("static/default_inputs.csv", index_col=0)

# Streamlit App
icon_dict = {"EU": "🇪🇺", "DE": "🇩🇪"}
st.set_page_config(
    page_title="NoStream",
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

st.markdown("# NoStream: Erdgas Energy Dashboard")

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
    eduction_import_russia = se.setting_embargo()

    # Compensation
    se.setting_compensation(status_quo_data)

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
se.plot_import_gap(status_quo_data, streamlit_object=cols[0])

# Stauts Quo
se.plot_status_quo(status_quo_data, streamlit_object=cols[1])

se.message_embargo_compensation(status_quo_data)

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
    st.session_state.red_dom_dem,
    st.session_state.red_elec_dem,
    st.session_state.red_ghd_dem,
    st.session_state.red_ind_dem,
    st.session_state.red_exp_dem,
    st.session_state.import_stop_date,
    st.session_state.demand_reduction_date,
    st.session_state.lng_increase_date,
    st.session_state.reduction_import_russia,
    st.session_state.add_lng_import,
    st.session_state.add_pl_import,
    consider_gas_reserve,
)

st.markdown("## Optimierungsergebnisse")
start_opti = st.button("Optimierung ausführen")

if start_opti:
    # Optimization
    print(
        [
            st.session_state.add_lng_import,
            st.session_state.add_pl_import,
            st.session_state.red_ind_dem,
            st.session_state.red_elec_dem,
            st.session_state.red_ghd_dem,
            st.session_state.red_dom_dem,
            st.session_state.red_exp_dem,
            st.session_state.reduction_import_russia,
            consider_gas_reserve,
            status_quo_data,
        ]
    )
    try:
        st.session_state.df, st.session_state.input_data = se.start_optimization(
            consider_gas_reserve, status_quo_data,
        )
    except Exception as e:
        st.write(e)

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
