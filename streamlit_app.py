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


def download_df(
    object_to_download, download_filename, download_link_text, streamlit_obj=None
):

    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv()
    b64 = base64.b64encode(object_to_download.encode()).decode()

    link = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    if streamlit_obj is None:
        st.markdown(link, unsafe_allow_html=True)
    else:
        streamlit_obj.markdown(link, unsafe_allow_html=True)


def download_pdf(object_to_download, download_filename, download_link_text):
    with open(object_to_download, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    link = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    st.markdown(link, unsafe_allow_html=True)


FZJcolor = ut.get_fzjColor()
legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)

font_dict = dict(size=16)

write_image = False  # True False
scale = 2
width = 3000 / scale
height = 1000 / scale


### Streamlit App
st.set_page_config(
    page_title="Energy Independence",
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

st.text("")
st.markdown("# Reduktion Russischer Erdgas-Importe")
st.markdown("## Auswirkungen auf die Versorgungssicherheit in Europa")

st.text("")
# st.markdown("Dashboard:")


def displayPDF(file, width=700, height=1000):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def render_svg(figDir):
    f = open(figDir, "r")
    lines = f.readlines()
    svg = "".join(lines)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    return html


with st.sidebar:
    cols = st.columns([2, 6])
    svg_image = render_svg("Input/FJZ IEK-3.svg")
    cols[0].write(svg_image, unsafe_allow_html=True)
    st.text("")

    st.markdown("### Einstellungen")
    with st.expander("Importstopp", expanded=False):
        import_stop_date = st.date_input(
            "Beginn der Importreduktion",
            value=datetime.date(2022, 4, 16),
            min_value=datetime.date(2022, 3, 15),
            max_value=datetime.date(2023, 12, 31),
        )
        import_stop_date = datetime.datetime.fromordinal(import_stop_date.toordinal())

        total_import = st.number_input(
            "Erdgas-Import gesamt¹ [TWh/a]", min_value=0, max_value=None, value=4190
        )

        total_production = st.number_input(
            "Inländische Erdgasproduktion¹ [TWh/a]",
            min_value=0,
            max_value=None,
            value=608,
        )

        total_import_russia = st.number_input(
            "Erdgas-Import aus Russland¹ [TWh/a]",
            min_value=0,
            max_value=None,
            value=1752,
        )

        pl_reduction = (
            st.slider(
                "Reduktion der russischen Erdgas-Importe um [%]",
                min_value=100,
                max_value=0,
                value=100,
                step=1,
            )
            / 100
        )
        russ_share = 1 - pl_reduction

        st.markdown(
            "¹ Voreingestellte Werte: Erdgas-Import/-Produktion EU27, 2019 (Quelle: [Eurostat Energy Balance](https://ec.europa.eu/eurostat/databrowser/view/NRG_TI_GAS__custom_2316821/default/table?lang=en), 2022)"
        )

    with st.expander("Nachfrageredutkion", expanded=False):
        demand_reduction_date = st.date_input(
            "Beginn der Nachfragereduktion",
            value=datetime.date(2022, 3, 16),
            min_value=datetime.date(2022, 3, 15),
            max_value=datetime.date(2023, 12, 31),
        )
        demand_reduction_date = datetime.datetime.fromordinal(
            demand_reduction_date.toordinal()
        )

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

        st.markdown(
            "² Voreingestellte Werte: Erdgas-Bedarf EU27, 2019 (Quelle: [Eurostat Databrowser](https://ec.europa.eu/eurostat/cache/sankey/energy/sankey.html?geos=EU27_2020&year=2019&unit=GWh&fuels=TOTAL&highlight=_2_&nodeDisagg=1111111111111&flowDisagg=true&translateX=15.480270462412136&translateY=135.54626885696325&scale=0.6597539553864471&language=EN), 2022)"
        )

    with st.expander("LNG Kapazitäten", expanded=False):
        lng_increase_date = st.date_input(
            "Beginn der LNG Kapazität-Erhöhung",
            value=datetime.date(2022, 5, 1),
            min_value=datetime.date(2022, 1, 1),
            max_value=datetime.date(2023, 12, 30),
        )
        lng_increase_date = datetime.datetime.fromordinal(lng_increase_date.toordinal())

        lng_base_import = 875
        lng_total_import = st.slider(
            "Genutzte LNG Import Kapazität² [TWh/a]",
            min_value=0,
            max_value=2025,
            value=965 + lng_base_import,
        )
        lng_add_import = lng_total_import - lng_base_import

        st.markdown(
            "³ Genutzte LNG-Kapazitäten EU27 (2021): 875 TWh/a (43% Auslastung) (Quelle: [GIE](https://www.gie.eu/transparency/databases/lng-database/), 2022)"
        )

    st.text("")
    st.markdown(
        "⛲ [Quellcode der Optimierung](https://github.com/FZJ-IEK3-VSA/NoStream/blob/master/optimization.py)"
    )  # 💻

    st.markdown(
        "🔎 [Weitere Informationen](https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html)"
    )  # 📜

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
        marker=dict(color=FZJcolor.get("green")),
    )
)

yvals[ypos] = total_ghd_demand
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Bedarfe",
        name="GHD",
        marker=dict(color=FZJcolor.get("purple2")),
    )
)

yvals[ypos] = total_industry_demand
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Bedarfe",
        name="Industrie",
        marker=dict(color=FZJcolor.get("grey2")),
    )
)

yvals[ypos] = total_electricity_demand
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Bedarfe",
        name="Energie",
        marker=dict(color=FZJcolor.get("blue")),
    )
)

yvals[ypos] = total_exports_and_other
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Bedarfe",
        name="Export etc.",
        marker=dict(color=FZJcolor.get("blue2")),
    )
)

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
        marker=dict(color=FZJcolor.get("orange")),
    )
)

yvals[ypos] = total_production
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Import & Produktion",
        name="Produktion",
        marker=dict(color=FZJcolor.get("green2")),
    )
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
        marker=dict(color=FZJcolor.get("red")),
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
        marker=dict(color=FZJcolor.get("green")),
    )
)

yvals[ypos] = total_ghd_demand * red_ghd_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="GHD (Nachfragereduktion)",
        marker=dict(color=FZJcolor.get("purple2")),
    )
)

yvals[ypos] = total_industry_demand * red_ind_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="Industrie (Nachfragereduktion)",
        marker=dict(color=FZJcolor.get("grey2")),
    )
)

yvals[ypos] = total_electricity_demand * red_elec_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="Energie (Nachfragereduktion)",
        marker=dict(color=FZJcolor.get("blue")),
    )
)

yvals[ypos] = total_exports_and_other * red_exp_dem
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="Export etc. (Nachfragereduktion)",
        marker=dict(color=FZJcolor.get("blue2")),
    )
)

yvals[ypos] = lng_add_import
fig.add_trace(
    go.Bar(
        x=xval,
        y=yvals,
        legendgroup="Kompensation",
        name="LNG Kapazitätserhöhung",
        marker=dict(color=FZJcolor.get("yellow3")),
    )
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
    total_demand = df.domDem + df.elecDem + df.indDem + df.ghdDem + df.exp_n_oth
    total_demand_served = (
        df.domDem_served
        + df.elecDem_served
        + df.indDem_served
        + df.ghdDem_served
        + df.exp_n_oth_served
    )
    unserved_demand = total_demand - total_demand_served

    fig = go.Figure()
    xvals = df.time

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.domDem_served,
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
            y=df.ghdDem_served,
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
            y=df.elecDem_served,
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
            y=df.indDem_served,
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
            y=df.lngImp_served,
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
            y=df.pipeImp_served,
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
            name="Speicherkapazität",
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
            y=df.pipeImp_served,
            stackgroup="one",
            name="Pipeline Import",
            mode="none",
            fillcolor=FZJcolor.get("orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.lngImp_served,
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
    storage_operation = df.lngImp_served + df.pipeImp_served - total_demand_served
    storage_discharge = [min(0, x) for x in storage_operation]
    storage_charge = np.array([max(0, x) for x in storage_operation])

    storage_operation_pl = df.pipeImp_served - total_demand_served
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


def hash_from_tuple(t):
    # m = hashlib.md5()
    res = ""
    for s in t:
        s = str(s)
        res += s
        # m.update(s.encode())
    return res  # m.hexdigest()


hash_val = hash_from_tuple(
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
        lng_add_import,
        use_soc_slack,
    )
)
default_hash = "41906081752926421151511109880.130.20.080.080.02022-04-16 00:00:002022-03-16 00:00:002022-05-01 00:00:000.0965False"  # 3073516694676277863
# st.write(hash_val)

st.markdown("## Optimierungsergebnisse")
start_opti = False

if hash_val != default_hash:
    start_opti = st.button("Optimierung ausführen")


if start_opti:
    with st.spinner(
        text="Starte Optimierung. Rechenzeit kann 3-5 Minuten in Anspruch nehmen ☕ ..."
    ):
        try:
            df, input_data = opti.run_scenario(
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
                lng_add_import=lng_add_import,
                use_soc_slack=use_soc_slack,
            )
            plot_optimization_results(df)
        except Exception as e:
            st.write(e)

if hash_val == default_hash:
    if not start_opti:
        with st.spinner(text="Lade Ergebnisse..."):
            df = pd.read_excel("Results_Optimization/default_results.xlsx", index_col=0)
            plot_optimization_results(df)
            input_data = pd.read_excel(
                "Results_Optimization/default_inputs.xlsx", index_col=0
            )

if start_opti or hash_val == default_hash:
    short_hash = int(abs(hash(hash_val)))
    download_df(
        df,
        f"Optimierungsergebnisse_{short_hash}.csv",
        "Optimierungsergebnisse herunterladen",
    )
    download_df(
        input_data, f"Input_Daten_{short_hash}.csv", "Input-Daten herunterladen",
    )


st.markdown("## Analyse: Energieversorgung ohne russisches Erdgas")
download_pdf(
    "Input/Analyse.pdf",
    "Analyse_energySupplyWithoutRussianGasAnalysis.pdf",
    "Analyse herunterladen",
)

displayPDF("Input/Analyse.pdf", width=900, height=635)


st.text("")

st.markdown("## Pressemitteilung")
download_pdf(
    "Input/Pressemitteilung.pdf",
    "Pressemitteilung_energySupplyWithoutRussianGasAnalysis.pdf",
    "Pressemitteilung herunterladen",
)
displayPDF("Input/Pressemitteilung.pdf", width=900, height=635)
