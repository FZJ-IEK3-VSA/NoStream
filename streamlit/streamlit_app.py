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


FZJcolor = ut.get_fzjColor()
legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)

font_dict = dict(size=16)


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
    cols = st.columns([2, 6])
    svg_image = (
        r'<a href="https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html">'
        + ut.render_svg("static/FJZ IEK-3.svg")
        + r"</a>"
    )
    cols[0].write(svg_image, unsafe_allow_html=True)
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

    # (
    #     demand_reduction_date,
    #     red_ind_dem,
    #     red_elec_dem,
    #     red_ghd_dem,
    #     red_dom_dem,
    #     red_exp_dem,
    # ) = se.setting_compensation_demand()

    # # Compensation - Natural gas import
    # (
    #     add_lng_import,
    #     lng_increase_date,
    #     add_pl_import,
    # ) = se.setting_compensation_import()

    st.markdown("### Status Quo")

    # Supply
    se.setting_statusQuo_supply(expanded=False)

    # Demand
    se.setting_statusQuo_demand()

    st.text("NoStream 0.2")
    st.markdown(
        "⛲ [Dokumentation und Quellcode](https://github.com/FZJ-IEK3-VSA/NoStream)"
    )

    st.markdown(
        "🌎 [Zur Institutsseite (IEK-3)](https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html)"
    )

    st.markdown(
        "📜 [Impressum](https://www.fz-juelich.de/portal/DE/Service/Impressum/impressum_node.html)"
    )

    st.markdown(
        "💡 [Verbesserungsvorschläge?](https://github.com/FZJ-IEK3-VSA/NoStream/issues)"
    )


# Energiebilanz

# Embarge und Kompensation
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
    font_dict=font_dict,
    streamlit_object=cols[0],
)

# Stauts Quo
se.plot_status_quo(
    font_dict=font_dict, streamlit_object=cols[1],
)

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
#st.write(scen_code)

st.markdown("## Optimierungsergebnisse")
start_opti = False

if scen_code != default_scen_code:
    start_opti = st.button("Optimierung ausführen")


if start_opti:
    with st.spinner(
        text="Starte Optimierung. Rechenzeit kann 3-5 Minuten in Anspruch nehmen ☕ ..."
    ):
        try:
            df, input_data = opti.run_scenario(
                total_ng_import=total_ng_import,
                total_pl_import_russia=total_pl_import_russia,
                total_ng_production=total_ng_production,
                total_lng_import=total_lng_import,
                total_lng_import_russia=total_lng_import_russia,
                total_domestic_demand=total_domestic_demand,
                total_ghd_demand=total_ghd_demand,
                total_electricity_demand=total_electricity_demand,
                total_industry_demand=total_industry_demand,
                total_exports_and_other=total_exports_and_other,
                red_dom_dem=red_dom_dem,
                red_elec_dem=red_elec_dem,
                red_ghd_dem=red_ghd_dem,
                red_ind_dem=red_ind_dem,
                red_exp_dem=red_exp_dem,
                import_stop_date=import_stop_date,
                demand_reduction_date=demand_reduction_date,
                lng_increase_date=lng_increase_date,
                reduction_import_russia=reduction_import_russia,
                add_lng_import=add_lng_import,
                add_pl_import=add_pl_import,
            )
            se.plot_optimization_results(df, legend_dict=None, font_dict=None)
        except Exception as e:
            st.write(e)

if scen_code == default_scen_code:
    if not start_opti:
        with st.spinner(text="Lade Ergebnisse des Standardszenarios..."):
            df = pd.read_csv("static/results/default_results.csv", index_col=0)
            se.plot_optimization_results(df, legend_dict=None, font_dict=None)
            input_data = pd.read_csv("static/default_inputs.csv", index_col=0)

if start_opti or scen_code == default_scen_code:
    short_hash = int(abs(hash(scen_code)))
    ut.download_df(
        df,
        f"Optimierungsergebnisse_{short_hash}.csv",
        "💾 Optimierungsergebnisse speichern",
    )
    ut.download_df(
        input_data, f"Input_Daten_{short_hash}.csv", "💾 Input-Daten speichern",
    )

st.text("")

st.markdown("## Analyse: Energieversorgung ohne russisches Erdgas")
st.markdown(
    "🖨️ [Vollständige Analyse herunterladen](https://www.fz-juelich.de/iek/iek-3/DE/_Documents/Downloads/energySupplyWithoutRussianGasAnalysis.pdf?__blob=publicationFile)"
)
