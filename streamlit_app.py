#%%
import pandas as pd
import streamlit as st
import numpy as np

import plotly.graph_objects as go
import datetime
import get_data as gdta
import storage_sim as opti
import base64
# from PIL import Image

import os


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


FZJcolor = gdta.get_fzjColor()
legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)

font_dict = dict(size=16)

write_image = False  # True False
scale = 2
width = 3000 / scale
height = 1000 / scale


### Streamlit App
st.set_page_config(
    page_title="Energy Independence", page_icon="🇪🇺", layout="wide"  # wide centered
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.text("")
st.markdown("# Reduktion Russischer Erdgas-Importe")
st.markdown("## Auswirkungen auf die Versorgungssicherheit in Europa")

st.text("")
# st.markdown("Dashboard:")


def render_svg(figDir):
    f = open(figDir,"r")
    lines = f.readlines()
    svg =''.join(lines)
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    return html


with st.sidebar:
    cols = st.columns([2,6])
    svg_image = render_svg("Input/FJZ IEK-3.svg")
    cols[0].write(svg_image, unsafe_allow_html=True)
    st.text("")

    st.markdown("### Einstellungen")
    with st.expander("Importstopp", expanded=False):
        # cols = st.columns(2)
        total_import = st.number_input(
            "Erdgas-Import gesamt¹ [TWh/a]", min_value=0, max_value=None, value=4190
        )

        total_production = st.number_input(
            "Innländische Erdgasproduktion¹ [TWh/a]", min_value=0, max_value=None, value=608
        )

        cols = st.columns(2)
        total_import_russia = st.number_input(
            "Erdgas-Import aus Russland¹ [TWh/a]", min_value=0, max_value=None, value=1752
        )

        pl_reduction = st.slider(
            "Reduktion der russischen Erdgas-Importe um [%]",
            min_value=100,
            max_value=0,
            value=100,
            step=1,
        )/100
        russ_share = 1 - pl_reduction

        cols = st.columns(2)
        import_stop_date = st.date_input(
            "Beginn der Importreduktion",
            value=datetime.date(2022, 4, 16),
            min_value=datetime.date(2022, 3, 15),
            max_value=datetime.date(2023, 12, 31),
        )
        import_stop_date = datetime.datetime.fromordinal(import_stop_date.toordinal())

        st.markdown(
            "¹ Voreingestellte Werte entsprechen Erdgas-Import EU27, 2019 (Quelle: [Eurostat Energy Balance](https://ec.europa.eu/eurostat/databrowser/view/NRG_TI_GAS__custom_2316821/default/table?lang=en), 2022)"
        )

    with st.expander("Nachfrageredutkion", expanded=False):
        cols = st.columns(2)
        total_domestic_demand = st.number_input(
            "Nachfrage Haushalte¹ [TWh/a]", min_value=0, max_value=None, value=926
        )
        red_dom_dem = (
            st.slider(
                "Reduktion der Nachfrage Haushalte um [%]",
                key="red_dom_dem",
                min_value=0,
                max_value=100,
                value=13,
                step=1,
            )
            / 100
        )

        cols = st.columns(2)
        total_ghd_demand = st.number_input(
            "Nachfrage GHD¹ [TWh/a]", min_value=0, max_value=None, value=421
        )
        red_ghd_dem = (
            st.slider(
                "Reduktion der Nachfrage GHD um [%]",
                key="red_ghd_dem",
                min_value=0,
                max_value=100,
                value=8,
                step=1,
            )
            / 100
        )

        cols = st.columns(2)
        total_electricity_demand = st.number_input(
            "Nachfrage Energie-Sektor¹ [TWh/a]", min_value=0, max_value=None, value=1515
        )
        red_elec_dem = (
            st.slider(
                "Reduktion der Nachfrage Energie um [%]",
                key="red_elec_dem",
                min_value=0,
                max_value=100,
                value=20,
                step=1,
            )
            / 100
        )

        cols = st.columns(2)
        total_industry_demand = st.number_input(
            "Nachfrage Industrie¹ [TWh/a]", min_value=0, max_value=None, value=1110
        )
        red_ind_dem = (
            st.slider(
                "Reduktion der Nachfrage Industrie um [%]",
                key="red_ind_dem",
                min_value=0,
                max_value=100,
                value=8,
                step=1,
            )
            / 100
        )

        cols = st.columns(2)
        total_exports_and_other = st.number_input(
            "Export und sonstige Nachfragen¹ [TWh/a]",
            min_value=0,
            max_value=None,
            value=988,
        )
        red_exp_dem = (
            st.slider(
                "Reduktion der Exporte um [%]",
                key="red_exp_dem",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )
            / 100
        )

        cols = st.columns(2)
        demand_reduction_date = st.date_input(
            "Beginn der Nachfragereduktion",
            value=datetime.date(2022, 3, 16),
            min_value=datetime.date(2022, 3, 15),
            max_value=datetime.date(2023, 12, 31),
        )
        demand_reduction_date = datetime.datetime.fromordinal(
            demand_reduction_date.toordinal()
        )

        st.markdown(
            "² Voreingestellte Werte: Erdgas-Bedarf EU27, 2019 (Quelle: [Eurostat Databrowser](https://ec.europa.eu/eurostat/cache/sankey/energy/sankey.html?geos=EU27_2020&year=2019&unit=GWh&fuels=TOTAL&highlight=_2_&nodeDisagg=1111111111111&flowDisagg=true&translateX=15.480270462412136&translateY=135.54626885696325&scale=0.6597539553864471&language=EN), 2022)"
        )

    with st.expander("LNG Kapazitäten", expanded=False):
        cols = st.columns(2)
        lng_capacity = st.slider(
            "Genutzte LNG Import Kapazität² [TWh/a]",
            min_value=0,
            max_value=2025,
            value=965 + 875,
        )
        lng_base_capacity = 875
        lng_add_capacity = lng_capacity - lng_base_capacity

        lng_increase_date = st.date_input(
            "Beginn der LNG Kapazität-Erhöhung",
            value=datetime.date(2022, 5, 1),
            min_value=datetime.date(2022, 1, 1),
            max_value=datetime.date(2023, 12, 30),
        )
        lng_increase_date = datetime.datetime.fromordinal(lng_increase_date.toordinal())

        st.markdown(
            "³ Genutzte LNG-Kapazitäten EU27 (2021): 875 TWh/a (43% Auslastung) (Quelle: [GIE](https://www.gie.eu/transparency/databases/lng-database/), 2022)"
        )
    st.markdown("[View code for optimization](https://github.com/ToniGustavson/eu_energy_independence/blob/3a5a78ca33caec7777fb3b04afea66a423182335/storage_sim.py)")


use_soc_slack = False

# Energiebilanz
st.markdown("## Erdgas-Bilanz")

fig = go.Figure()
xval = ["Bedarfe", "Import & Produktion", "Importlücke Russland", "Kompensation"]
yempty = [0, 0, 0, 0]

## Bedarfe
ypos = 0
yvals = yempty.copy()
yvals[ypos] = total_domestic_demand
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Bedarfe",
        legendgrouptitle_text="Bedarfe",
        name="Haushalte",
    )
)

yvals[ypos] = total_ghd_demand
fig.add_trace(go.Bar(x=xval, y=yvals, legendgroup="Bedarfe", name="GHD",))

yvals[ypos] = total_industry_demand
fig.add_trace(go.Bar(x=xval, y=yvals, legendgroup="Bedarfe", name="Industrie",))

yvals[ypos] = total_electricity_demand
fig.add_trace(go.Bar(x=xval, y=yvals, legendgroup="Bedarfe", name="Energie",))

yvals[ypos] = total_exports_and_other
fig.add_trace(go.Bar(x=xval, y=yvals, legendgroup="Bedarfe", name="Export etc.",))

## Import & Produktion
ypos = 1
yvals = yempty.copy()
yvals[ypos] = total_import
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Import & Produktion",
        legendgrouptitle_text="Import & Produktion",
        name="Import",
    )
)

yvals[ypos] = total_production
fig.add_trace(
    go.Bar(x=xval, y=yvals, legendgroup="Import & Produktion", name="Produktion",)
)


## Importlücke
ypos = 2
yvals = yempty.copy()
yvals[ypos] = total_import_russia * pl_reduction

fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Importlücke",
        legendgrouptitle_text="Importlücke",
        name="Import Russland",
    )
)

## Kompensation
ypos = 3
yvals = yempty.copy()
yvals[ypos] = total_domestic_demand * red_dom_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        legendgrouptitle_text="Kompensation",
        name="Haushalte (Nachfragereduktion)",
    )
)

yvals[ypos] = total_ghd_demand * red_ghd_dem
fig.add_trace(
    go.Bar(
        x=xval, y=yvals, legendgroup="Kompensation", name="GHD (Nachfragereduktion)",
    )
)

yvals[ypos] = total_industry_demand * red_ind_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="Industrie (Nachfragereduktion)",
    )
)

yvals[ypos] = total_electricity_demand * red_elec_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="Energie (Nachfragereduktion)",
    )
)

yvals[ypos] = total_exports_and_other * red_exp_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="Export etc. (Nachfragereduktion)",
    )
)

yvals[ypos] = lng_add_capacity
fig.add_trace(
    go.Bar(x=xval, y=yvals, legendgroup="Kompensation", name="LNG Kapazitätserhöhung",)
)


fig.update_layout(
    title=f"Status Quo, Import-Lücke und Kompensationsmöglichkeiten ",
    yaxis_title="Erdgas [TWh/a]",
    barmode="stack",
    font=font_dict,
    # legend=legend_dict,
)
# fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)



def plot_optimization_results(df):
    # Demand
    total_demand = df.dom_Dem + df.elec_Dem + df.ind_Dem + df.ghd_Dem + df.exp_n_oth
    total_demand_served = (
        df.dom_served
        + df.elec_served
        + df.ind_served
        + df.ghd_served
        + df.exp_n_oth_served
    )
    unserved_demand = total_demand - total_demand_served

    fig = go.Figure()
    xvals = df.time

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.dom_served,
            stackgroup="one",
            legendgroup="bedarf",
            name="Haushalte",
            mode="none",
            fillcolor=FZJcolor.get("green")
            # marker=marker_dict,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.ghd_served,
            stackgroup="one",
            legendgroup="bedarf",
            name="GHD",
            mode="none",
            fillcolor=FZJcolor.get("purple2"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.elec_served,
            stackgroup="one",
            legendgroup="bedarf",
            name="Energie",
            mode="none",
            fillcolor=FZJcolor.get("blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.ind_served,
            stackgroup="one",
            legendgroup="bedarf",
            legendgrouptitle_text="Erdgasbedarfe",
            name="Industrie",
            mode="none",
            fillcolor=FZJcolor.get("grey2"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.exp_n_oth_served,
            stackgroup="one",
            legendgroup="bedarf",
            name="Export und sonstige",
            mode="none",
            fillcolor=FZJcolor.get("blue2"),
        )
    )

    if sum(unserved_demand) > 0.001:
        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=total_demand - total_demand_served,
                stackgroup="one",
                legendgroup="bedarf",
                name=f"Abgeregelt ({int(sum(unserved_demand))} TWh)",
                mode="none",
                fillcolor=FZJcolor.get("red"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.lngServed,
            stackgroup="two",
            line=dict(color=FZJcolor.get("yellow3"), width=3.5),
            legendgroup="import",
            legendgrouptitle_text="Erdgasimport",
            name="LNG Import",
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.pipeServed,
            stackgroup="two",
            line=dict(color=FZJcolor.get("orange"), width=3.5),
            legendgroup="import",
            name="Pipeline Import",
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    fig.update_layout(
        title=f"Erdgasbedarfe und Import",
        font=font_dict,
        yaxis_title="Erdgas [TWh/h]",
        # legend=legend_dict,
    )
    # fig.update_layout(showlegend=False)

    if write_image:
        fig.write_image(
            f"Output/Optimierung_Erdgasbedarf_{scenario_name}.png",
            width=width,
            height=height,
            # scale=scale,
        )

    st.plotly_chart(fig, use_container_width=True)

    # st.markdown(
    #     f"Ungedeckter, abgeregelter Erdgasbedarf: {int(sum(unserved_demand))} TWh"
    # )

    ## SOC
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.soc,
            stackgroup="one",
            name="Füllstand",
            mode="none",
            fillcolor=FZJcolor.get("orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=np.ones(len(xvals)) * 1100,
            name="Maximale Kapazität",
            line=dict(color=FZJcolor.get("black"), width=2),
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    fig.update_layout(
        title=f"Speicherfüllstand",
        font=font_dict,
        yaxis_title="Erdgas [TWh]",
        legend=legend_dict,
    )
    # fig.update_layout(showlegend=False)

    if write_image:
        fig.write_image(
            f"Output/Optimierung_Speicher_{scenario_name}.png",
            width=width,
            height=height,
            # scale=scale,
        )

    st.plotly_chart(fig, use_container_width=True)

    ##  Pipeline Import
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.pipeServed,
            stackgroup="one",
            name="Pipeline Import",
            mode="none",
            fillcolor=FZJcolor.get("orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.lngServed,
            stackgroup="one",
            name="LNG Import",
            mode="none",
            fillcolor=FZJcolor.get("yellow3"),
        )
    )

    fig.update_layout(
        title=f"Erdgasimporte", yaxis_title="Erdgas [TWh/h]", legend=legend_dict,
    )
    st.plotly_chart(fig, use_container_width=True)

    ## Storage Charge and discharge
    storage_operation = df.lngServed + df.pipeServed - total_demand_served
    storage_discharge = [min(0, x) for x in storage_operation]
    storage_charge = np.array([max(0, x) for x in storage_operation])

    storage_operation_pl = df.pipeServed - total_demand_served
    storage_charge_pl = np.array([max(0, x) for x in storage_operation_pl])

    storage_operation_lng = storage_charge - storage_charge_pl
    storage_charge_lng = np.array([max(0, x) for x in storage_operation_lng])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=storage_discharge,
            stackgroup="two",
            name="Ausspeicherung",
            mode="none",
            fillcolor=FZJcolor.get("red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=storage_charge_pl,
            stackgroup="one",
            name="Speicherung (Pipeline)",
            mode="none",
            fillcolor=FZJcolor.get("orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=storage_charge_lng,
            stackgroup="one",
            name="Speicherung (LNG)",
            mode="none",
            fillcolor=FZJcolor.get("yellow3"),
        )
    )

    fig.update_layout(
        title=f"Ein- und Ausspeicherung Gasspeicher",
        yaxis_title="Erdgas [TWh/h]",
        legend=legend_dict,
    )

    st.plotly_chart(fig, use_container_width=True)


hash_val = hash(
    (
        total_import,
        total_production,
        total_import_russia,
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
        russ_share,
        lng_add_capacity,
        use_soc_slack,
    )
)
default_hash = -5024794703248336817 # 3073516694676277863
# st.write(hash_val)

st.markdown("## Optimierungsergebnisse")
start_opti = False
if hash_val != default_hash:
    # st.info("Starten Sie die Optimierung druch Klicken auf 'Optimierung ausführen' im Menüband unten auf der linken Seite")
    start_opti = st.button("Optimierung ausführen")


if start_opti:
    with st.spinner(
        text="Starte Optimierung. Rechnezeit kann 3-5 Minuten in Anspruch nehmen ☕ ..."
    ):
        try:
            df = opti.run_scenario(
                total_import=total_import,
                total_production=total_production,
                total_import_russia=total_import_russia,
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
                russ_share=russ_share,
                lng_add_capacity=lng_add_capacity,
                use_soc_slack=use_soc_slack,
            )
            plot_optimization_results(df)
        except Exception as e:
            st.write(e)

if hash_val == default_hash:
    if not start_opti:
        with st.spinner(text="Lade Ergebnisse..."):
            scenario_name = "default_scenario"
            df = gdta.get_optiRes(scenario_name)
            plot_optimization_results(df)
            tmp_download_link = download_link(df, f"Optimization_Results_{int(hash_val)}.csv", "Ergebnisse herunterladen")
            st.markdown(tmp_download_link, unsafe_allow_html=True)