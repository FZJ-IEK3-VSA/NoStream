import streamlit as st
import numpy as np
import plotly.graph_objects as go
import datetime
import utils as ut
import optimization as opti
import requests

try:
    from streamlit_lottie import st_lottie_spinner
except:
    print("Failed to load streamlit_lottie")

FZJcolor = ut.get_fzjColor()
legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
font_dict = dict(size=16)

# Energy demands
total_lng_import = 914
total_industry_demand = 1110
total_exports_and_other = 988
total_domestic_demand = 926
total_ghd_demand = 421
total_electricity_demand = 1515
total_ng_import = 4190
total_ng_import_russia = 1752
total_lng_import_russia = 160
total_pl_import_russia = total_ng_import_russia - total_lng_import_russia
total_ng_production = 608

# Storage Reserve
reserve_dates_default = [
    datetime.datetime(2022, 8, 1, 0, 0),
    datetime.datetime(2022, 9, 1, 0, 0),
    datetime.datetime(2022, 10, 1, 0, 0),
    datetime.datetime(2022, 11, 1, 0, 0),
    datetime.datetime(2023, 2, 1, 0, 0),
    datetime.datetime(2023, 5, 1, 0, 0),
    datetime.datetime(2023, 7, 1, 0, 0),
]
reserve_soc_val = [0.63, 0.68, 0.74, 0.80, 0.43, 0.33, 0.52]
reserve_soc_val_percent = [int(x * 100) for x in reserve_soc_val]
storage_cap = 1100
reserve_soc_val_default = [x * storage_cap for x in reserve_soc_val]

reserve_dict = dict(zip(reserve_dates_default, reserve_soc_val_percent))

# Dates
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date(2023, 7, 1)

# Formats
format_date = "DD.MM.YYYY"
format_percent = "%g %%"
format_ng = "%g TWh/a"


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def centered_fzj_logo():
    svg_image = (
        r'<center><a href="https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html">'
        + ut.render_svg("./static/FJZ_IEK-3.svg")
        + r"</a> </center>"
    )
    st.write(svg_image, unsafe_allow_html=True)


def sidebar_further_info():
    st.markdown(
        "⛲ [Quellcode & Dokumentation](https://github.com/FZJ-IEK3-VSA/NoStream)"
    )

    st.markdown(
        "📖 [Analyse & Hintergrundinformationen](https://www.fz-juelich.de/iek/iek-3/DE/_Documents/Downloads/energySupplyWithoutRussianGasAnalysis.pdf?__blob=publicationFile)"
    )

    st.markdown(
        "🌎 [Zur Institutsseite (IEK-3)](https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html)"
    )

    st.markdown(
        "📜 [Impressum](https://www.fz-juelich.de/portal/DE/Service/Impressum/impressum_node.html)"
    )

    st.markdown(
        "💡 [Verbesserungsvorschläge](https://github.com/FZJ-IEK3-VSA/NoStream/issues)"
    )

    st.markdown("`NoStream 0.3`")


def start_optimization(
    add_lng_import,
    add_pl_import,
    red_ind_dem,
    red_elec_dem,
    red_ghd_dem,
    red_dom_dem,
    red_exp_dem,
    reduction_import_russia,
    consider_gas_reserve,
):
    # lottie_download = "https://assets7.lottiefiles.com/packages/lf20_mdgiw1k2.json"
    # with st_lottie_spinner(
    #     load_lottieurl(lottie_download), width="30%", quality="high"
    # ):
    with st.spinner(
        text="Starte Optimierung. Rechenzeit kann einige Minuten in Anspruch nehmen ☕ ..."
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
                import_stop_date=st.session_state.import_stop_date,
                demand_reduction_date=st.session_state.demand_reduction_date,
                lng_increase_date=st.session_state.lng_increase_date,
                reduction_import_russia=reduction_import_russia,
                add_lng_import=add_lng_import,
                add_pl_import=add_pl_import,
                consider_gas_reserve=consider_gas_reserve,
                reserve_dates=st.session_state.reserve_dates,
                reserve_soc_val=st.session_state.reserve_soc_val,
            )
            return df, input_data
        except Exception as e:
            # pass
            st.write(e)


@st.experimental_memo(show_spinner=False)
def getFig_import_gap(
    reduction_import_russia,
    red_exp_dem,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    add_lng_import,
    add_pl_import,
    compact=False,
):
    fig = go.Figure()
    xval = ["Embargo", "Kompensation"]
    yempty = [0, 0]

    ## Import gap
    ypos = 0
    yvals = yempty.copy()
    yvals[ypos] = total_lng_import_russia * reduction_import_russia
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Embargo",
            legendgrouptitle_text="Embargo",
            name="LNG Import RU",
            marker=dict(color=FZJcolor.get("red2")),
        )
    )

    yvals[ypos] = total_pl_import_russia * reduction_import_russia
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Embargo",
            name="Pipeline Import RU",
            marker=dict(color=FZJcolor.get("red")),
        )
    )

    ## Compensation
    ypos = 1
    yvals = yempty.copy()

    yvals[ypos] = add_lng_import
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Kompensation",
            legendgrouptitle_text="Kompensation",
            name="LNG Import (zus.)",
            marker=dict(color=FZJcolor.get("orange2")),
        )
    )

    if add_pl_import > 0:
        yvals[ypos] = add_pl_import
        fig.add_trace(
            go.Bar(
                x=xval,
                y=yvals,
                legendgroup="Kompensation",
                name="Pipeline Import (zus.)",
                marker=dict(color=FZJcolor.get("orange")),
            )
        )

    if red_exp_dem > 0:
        yvals[ypos] = total_exports_and_other * red_exp_dem
        fig.add_trace(
            go.Bar(
                x=xval,
                y=yvals,
                legendgroup="Kompensation",
                name="Export etc.",
                marker=dict(color=FZJcolor.get("blue2")),
            )
        )

    yvals[ypos] = total_domestic_demand * red_dom_dem
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Kompensation",
            legendgrouptitle_text="Nachfragereduktion",
            name="Haushalte",
            marker=dict(color=FZJcolor.get("green")),
        )
    )

    yvals[ypos] = total_ghd_demand * red_ghd_dem
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Kompensation",
            name="Handel/Dienstleistung",
            marker=dict(color=FZJcolor.get("purple2")),
        )
    )

    yvals[ypos] = total_electricity_demand * red_elec_dem
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Kompensation",
            name="Kraft- und Heizwerke",
            marker=dict(color=FZJcolor.get("blue")),
        )
    )

    yvals[ypos] = total_industry_demand * red_ind_dem
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Kompensation",
            name="Industrie",
            marker=dict(color=FZJcolor.get("grey2")),
        )
    )

    if not compact:
        fig.update_layout(title="Embargo und Kompensation")
    fig.update_layout(
        yaxis_title="Erdgas [TWh/a]",
        barmode="stack",
        font=font_dict,
        # legend=legend_dict,
    )

    return fig


def plot_import_gap(
    reduction_import_russia,
    red_exp_dem,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    add_lng_import,
    add_pl_import,
    streamlit_object=st,
    compact=False,
):
    fig = getFig_import_gap(
        reduction_import_russia,
        red_exp_dem,
        red_dom_dem,
        red_elec_dem,
        red_ghd_dem,
        red_ind_dem,
        add_lng_import,
        add_pl_import,
        compact=compact,
    )
    streamlit_object.plotly_chart(fig, use_container_width=True)


@st.experimental_memo(show_spinner=False)
def getFig_status_quo():
    fig = go.Figure()
    xval = ["Versorgung", "Bedarfe"]
    yempty = [0, 0]
    marker_pattern_shape = "/"

    ## Versorgung
    ypos = 0
    yvals = yempty.copy()

    yvals[ypos] = total_ng_production
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Versorgung",
            legendgrouptitle_text="Versorgung",
            name="Produktion Inland",
            marker=dict(color=FZJcolor.get("green2")),
        )
    )

    yvals[ypos] = total_lng_import
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Versorgung",
            name="LNG Import ges.",
            marker=dict(color=FZJcolor.get("orange2")),
        )
    )

    # yvals[ypos] = total_lng_import - total_lng_import_russia
    # fig.add_trace(
    #     go.Bar(
    #         x=xval,
    #         y=yvals,
    #         legendgroup="Versorgung",
    #         name="LNG Import Rest",
    #         marker=dict(color=FZJcolor.get("orange2")),
    #     )
    # )

    # yvals[ypos] = total_lng_import_russia
    # fig.add_trace(
    #     go.Bar(
    #         x=xval,
    #         y=yvals,
    #         legendgroup="Versorgung",
    #         name="LNG Import RU",
    #         marker=dict(color=FZJcolor.get("red2")),
    #         # marker_pattern_shape=marker_pattern_shape,
    #     )
    # )

    yvals[ypos] = total_ng_import - total_lng_import
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Versorgung",
            name="Pipeline Import ges.",
            marker=dict(color=FZJcolor.get("orange")),
        )
    )

    # yvals[ypos] = total_ng_import - total_pl_import_russia - total_lng_import
    # fig.add_trace(
    #     go.Bar(
    #         x=xval,
    #         y=yvals,
    #         legendgroup="Versorgung",
    #         name="Pipeline Import Rest",
    #         marker=dict(color=FZJcolor.get("orange")),
    #     )
    # )

    # yvals[ypos] = total_pl_import_russia
    # fig.add_trace(
    #     go.Bar(
    #         x=xval,
    #         y=yvals,
    #         legendgroup="Versorgung",
    #         legendgrouptitle_text="Versorgung",
    #         name="Pipeline Import RU",
    #         marker=dict(color=FZJcolor.get("red")),
    #         # marker_pattern_shape=marker_pattern_shape,
    #     )
    # )

    ## Bedarfe
    ypos = 1
    yvals = yempty.copy()

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
            name="Handel/Dienstleistung",
            marker=dict(color=FZJcolor.get("purple2")),
        )
    )

    yvals[ypos] = total_electricity_demand
    fig.add_trace(
        go.Bar(
            x=xval,
            y=yvals,
            legendgroup="Bedarfe",
            name="Kraft- und Heizwerke",
            marker=dict(color=FZJcolor.get("blue")),
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

    fig.update_layout(
        title=f"Status Quo",
        yaxis_title="Erdgas [TWh/a]",
        barmode="stack",
        font=font_dict,
        # legend=legend_dict,
    )
    # fig.update_layout(showlegend=False)

    return fig


def plot_status_quo(streamlit_object=st):
    fig = getFig_status_quo()
    streamlit_object.plotly_chart(fig, use_container_width=True)


def add_dates(fig):
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=[st.session_state.lng_increase_date],
            y=[0],
            legendgroup="Dates",
            name="Importerhöhung",
            marker=dict(size=12, color=FZJcolor.get("green"), symbol="arrow-down"),
        )
    )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=[st.session_state.import_stop_date],
            y=[0],
            legendgroup="Dates",
            name="Embargo",
            marker=dict(size=12, color=FZJcolor.get("red"), symbol="arrow-down"),
        )
    )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=[st.session_state.demand_reduction_date],
            y=[0],
            legendgroup="Dates",
            legendgrouptitle_text="Startdaten",
            name="Nachfragereduktion",
            marker=dict(size=12, color=FZJcolor.get("grey"), symbol="arrow-down"),
        )
    )

    return fig


@st.experimental_memo(show_spinner=False)
def getFig_optimization_results(df):
    # df_og = df.copy()
    # Prevent flickering at the beginning
    df.loc[0:1080, "lngImp_served"] = df.loc[0:1080, "lngImp"]
    df.loc[0:1080, "plImp_served"] = df.loc[0:1080, "plImp"]
    df.loc[0:1080, "domProd_served"] = df.loc[0:1080, "domProd"]

    # Prevent last values from being zero
    df.loc[len(df) - 3 : len(df), "lngImp_served"] = df.loc[
        len(df) - 6 : len(df) - 4, "lngImp_served"
    ]
    df.loc[len(df) - 3 : len(df), "plImp_served"] = df.loc[
        len(df) - 6 : len(df) - 4, "plImp_served"
    ]

    # Demand
    total_demand = df.domDem + df.elecDem + df.indDem + df.ghdDem + df.exp_n_oth
    total_demand_served = (
        df.domDem_served
        + df.elecDem_served
        + df.indDem_served
        + df.ghdDem_served
        + df.exp_n_oth_served
    )

    threshold = 0.001
    unserved_demand = total_demand - total_demand_served
    unserved_demand = [x if x > threshold else 0 for x in unserved_demand]

    fig_flow = go.Figure()
    xvals = df.time

    fig_flow.add_trace(
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

    fig_flow.add_trace(
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

    fig_flow.add_trace(
        go.Scatter(
            x=xvals,
            y=df.ghdDem_served,
            stackgroup="one",
            legendgroup="bedarf",
            name="Handel/Dienstleistung",
            mode="none",
            fillcolor=FZJcolor.get("purple2"),
        )
    )

    fig_flow.add_trace(
        go.Scatter(
            x=xvals,
            y=df.elecDem_served,
            stackgroup="one",
            legendgroup="bedarf",
            name="Kraft- und Heizwerke",
            mode="none",
            fillcolor=FZJcolor.get("blue"),
        )
    )

    fig_flow.add_trace(
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

    if sum(unserved_demand) > threshold:
        fig_flow.add_trace(
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

    fig_flow.add_trace(
        go.Scatter(
            x=xvals,
            y=df.domProd_served,
            # stackgroup="two",
            line=dict(color=FZJcolor.get("green2"), width=3.5),
            legendgroup="Erdgasversorgung",
            legendgrouptitle_text="Erdgasversorgung",
            name="Produktion Inland",
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    fig_flow.add_trace(
        go.Scatter(
            x=xvals,
            y=df.lngImp_served,
            # stackgroup="two",
            line=dict(color=FZJcolor.get("orange2"), width=3.5),
            legendgroup="Erdgasversorgung",
            name="LNG",
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    fig_flow.add_trace(
        go.Scatter(
            x=xvals,
            y=df.plImp_served,
            # stackgroup="two",
            line=dict(color=FZJcolor.get("orange"), width=3.5),
            legendgroup="Erdgasversorgung",
            name="Pipeline",
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    fig_flow.add_trace(
        go.Scatter(
            x=xvals,
            y=df.plImp_served + df.lngImp_served + df.domProd_served,
            # stackgroup="two",
            line=dict(color=FZJcolor.get("black1"), width=3.5),
            legendgroup="Erdgasversorgung",
            name="Gesamt",
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    # Dates
    add_dates(fig_flow)

    fig_flow.update_layout(
        title=f"Erdgasbedarfe und Import",
        font=font_dict,
        yaxis_title="Erdgas [TWh/h]",
        yaxis_range=[0, 1],
        xaxis_range=[start_date, end_date],
    )

    ## SOC
    fig_soc = go.Figure()

    fig_soc.add_trace(
        go.Scatter(
            x=xvals,
            y=df.soc,
            stackgroup="one",
            name="Füllstand",
            legendgroup="Kenndaten",
            legendgrouptitle_text="Kenndaten",
            mode="none",
            fillcolor=FZJcolor.get("orange"),
        )
    )

    fig_soc.add_trace(
        go.Scatter(
            x=xvals,
            y=np.ones(len(xvals)) * 1100,
            name="Speicherkapazität",
            legendgroup="Kenndaten",
            line=dict(color=FZJcolor.get("black"), width=2),
            fillcolor="rgba(0, 0, 0, 0)",
        )
    )

    # Füllstandvorgabe
    fig_soc.add_trace(
        go.Scatter(
            mode="markers",
            x=st.session_state.reserve_dates,
            y=st.session_state.reserve_soc_val,
            name="Füllstandvorgabe",
            legendgroup="Kenndaten",
            marker=dict(size=8, color=FZJcolor.get("blue")),
        )
    )

    # Dates
    add_dates(fig_soc)

    fig_soc.update_layout(
        title=f"Speicherfüllstand",
        font=font_dict,
        yaxis_title="Erdgas [TWh]",
        yaxis_range=[0, 1200],
        xaxis_range=[start_date, end_date],
    )

    # st.plotly_chart(fig, use_container_width=True)

    # ##  Pipeline Import
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Scatter(
    #         x=xvals,
    #         y=df.plImp_served,
    #         stackgroup="one",
    #         name="Pipeline Import",
    #         mode="none",
    #         fillcolor=FZJcolor.get("orange"),
    #     )
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=xvals,
    #         y=df.lngImp_served,
    #         stackgroup="one",
    #         name="LNG Import",
    #         mode="none",
    #         fillcolor=FZJcolor.get("orange2"),
    #     )
    # )

    # fig.update_layout(
    #     title=f"Erdgasimporte", yaxis_title="Erdgas [TWh/h]", font=font_dict,
    # )
    # st.plotly_chart(fig, use_container_width=True)

    # ## Storage Charge and discharge
    # df = df_og.copy()
    # storage_operation = (
    #     df.lngImp_served + df.plImp_served + df.domProd_served - total_demand_served
    # )
    # storage_discharge = [min(0, x) for x in storage_operation]
    # storage_charge = np.array([max(0, x) for x in storage_operation])

    # storage_operation_prod = df.domProd_served - total_demand_served
    # storage_charge_prod = np.array([max(0, x) for x in storage_operation_prod])
    # rem_demand_prod = np.array([abs(min(0, x)) for x in storage_operation_prod])

    # storage_operation_pl = df.plImp_served - rem_demand_prod
    # storage_charge_pl = np.array([max(0, x) for x in storage_operation_pl])
    # rem_demand_pl = np.array([abs(min(0, x)) for x in storage_operation_pl])

    # storage_operation_lng = df.lngImp_served - rem_demand_pl
    # storage_charge_lng = np.array([max(0, x) for x in storage_operation_lng])
    # rem_demand_lng = np.array([abs(min(0, x)) for x in storage_operation_lng])

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         x=xvals,
    #         y=storage_discharge,
    #         stackgroup="two",
    #         legendgroup="Ausspeicherung",
    #         legendgrouptitle_text="Ausspeicherung",
    #         name="Ausspeicherung",
    #         mode="none",
    #         fillcolor=FZJcolor.get("red"),
    #     )
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=xvals,
    #         y=storage_charge_prod,
    #         stackgroup="one",
    #         legendgroup="Einspeicherung",
    #         legendgrouptitle_text="Einspeicherung",
    #         name="Produktion Inland",
    #         mode="none",
    #         fillcolor=FZJcolor.get("green2"),
    #     )
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=xvals,
    #         y=storage_charge_pl,
    #         stackgroup="one",
    #         legendgroup="Einspeicherung",
    #         name="Pipeline Import",
    #         mode="none",
    #         fillcolor=FZJcolor.get("orange"),
    #     )
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=xvals,
    #         y=storage_charge_lng,
    #         stackgroup="one",
    #         legendgroup="Einspeicherung",
    #         name="LNG Import",
    #         mode="none",
    #         fillcolor=FZJcolor.get("orange2"),
    #     )
    # )

    # fig.update_layout(
    #     title=f"Ein- und Ausspeicherung Gasspeicher",
    #     yaxis_title="Erdgas [TWh/h]",
    #     font=font_dict,
    # )

    # st.plotly_chart(fig, use_container_width=True)

    return fig_flow, fig_soc


def plot_optimization_results(df, streamlit_object=st):
    fig_flow, fig_soc = getFig_optimization_results(df)
    streamlit_object.plotly_chart(fig_flow, use_container_width=True)
    streamlit_object.plotly_chart(fig_soc, use_container_width=True)


def setting_compensation(streamlit_object=st, expanded=False, compact=False):
    with streamlit_object.expander("Kompensation", expanded=expanded):
        streamlit_object.markdown("### Nachfragereduktion")
        cols = streamlit_object.columns(2)

        so = cols[0]
        red_ind_dem = (
            so.slider(
                "Industrie",
                key="red_ind_dem",
                min_value=0,
                max_value=100,
                value=8,
                step=1,
                format=format_percent,
            )
            / 100
        )

        so = cols[1]
        red_elec_dem = (
            so.slider(
                "Kraft- und Heizwerke",
                key="red_elec_dem",
                min_value=0,
                max_value=100,
                value=20,
                step=1,
                format=format_percent,
            )
            / 100
        )

        cols = st.columns(2)
        # so = cols[0] if compact else st
        so = cols[0]
        red_ghd_dem = (
            so.slider(
                "Handel/Dienstleistung",
                key="red_ghd_dem",
                min_value=0,
                max_value=100,
                value=8,
                step=1,
                format=format_percent,
            )
            / 100
        )

        so = cols[1]
        red_dom_dem = (
            so.slider(
                "Haushalte",
                key="red_dom_dem",
                min_value=0,
                max_value=100,
                value=13,
                step=1,
                format=format_percent,
            )
            / 100
        )

        red_exp_dem = 0
        if not compact:
            # Reduction in exports equals the average reduction in the other sectors
            red_exp_dem = (
                red_dom_dem * total_domestic_demand
                + red_ghd_dem * total_ghd_demand
                + red_elec_dem * total_electricity_demand
                + red_ind_dem * total_industry_demand
            ) / (
                total_domestic_demand
                + total_ghd_demand
                + total_electricity_demand
                + total_industry_demand
            )
            red_exp_dem = int(round(100 * red_exp_dem, 0))

            cols = st.columns(2)
            so = cols[0]
            red_exp_dem = so.slider(
                "Exporte etc.",
                key="red_exp_dem",
                min_value=0,
                max_value=100,
                value=red_exp_dem,
                step=1,
                format=format_percent,
            )
            red_exp_dem /= 100

            # Date for start of the reduction
            date_input_red = streamlit_object.empty()
            start_now_red = streamlit_object.button("Ab sofort", key="start_now_red")
            if start_now_red:
                st.session_state.demand_reduction_date = datetime.date.today()
            st.session_state.demand_reduction_date = date_input_red.date_input(
                "Startdatum:",
                key="demand_reduction_date_key",
                value=st.session_state.demand_reduction_date,
                min_value=start_date,
                max_value=end_date,
            )
            st.session_state.demand_reduction_date = datetime.datetime.fromordinal(
                st.session_state.demand_reduction_date.toordinal()
            )
            streamlit_object.markdown("---")

        # Importerhöhung
        streamlit_object.markdown("### Importerhöhung")

        # Additional lng imports
        cols = streamlit_object.columns(2)
        so = cols[0] if not compact else st
        add_lng_import = so.slider(
            "LNG [TWh/a]¹",
            min_value=0,
            max_value=2025 - total_lng_import,
            value=int(0.9 * 2025 - total_lng_import),
        )
        streamlit_object.markdown(
            f"¹ Genutzte LNG-Kapazitäten EU27, 2021: {total_lng_import} TWh/a. Maximale Auslastung: 2025 TWh/a ➜ Freie Kapazität: {2025-total_lng_import} TWh/a (Quelle: [GIE](https://www.gie.eu/transparency/databases/lng-database/), 2022) - innereuropäische Pipeline-Engpässe sind hier nicht berücksichtigt."
        )

        # Additional pipeline imports
        so = cols[1]
        add_pl_import = 0
        if not compact:
            add_pl_import = so.slider(
                "Pipeline [TWh/a]²", min_value=0, max_value=300, value=add_pl_import,
            )
            streamlit_object.markdown(
                f"² Im Vergleich zum Höchstwert von Erdgaslieferungen (im Jahr 2017) können theoretisch weitere 26,1 Mrd. m³ Erdgas (~285 TWh/a) aus Norwegen, Algerien und UK importiert werden."
            )

        if not compact:
            # Date for the start of increasing the imports
            date_input_incr = streamlit_object.empty()
            start_now_incr = streamlit_object.button("Ab sofort", key="start_now_incr")
            if start_now_incr:
                st.session_state.lng_increase_date = datetime.date.today()
            st.session_state.lng_increase_date = date_input_incr.date_input(
                "Startdatum:",
                key="lng_increase_date_key",
                value=st.session_state.lng_increase_date,
                min_value=start_date,
                max_value=end_date,
            )
            st.session_state.lng_increase_date = datetime.datetime.fromordinal(
                st.session_state.lng_increase_date.toordinal()
            )

        return (
            red_ind_dem,
            red_elec_dem,
            red_ghd_dem,
            red_dom_dem,
            red_exp_dem,
            add_lng_import,
            add_pl_import,
        )


def setting_embargo(streamlit_object=st, expanded=False, compact=False):
    with streamlit_object.expander("Embargo", expanded=expanded):

        reduction_import_russia = 100
        if not compact:
            reduction_import_russia = st.slider(
                "Reduktion russischer Erdgasimporte um",
                min_value=100,
                max_value=0,
                value=reduction_import_russia,
                format=format_percent,
                step=1,
            )
        reduction_import_russia /= 100

        if not compact:
            date_input_embargo = st.empty()
            start_now_embargo = st.button("Ab sofort", key="start_now_embargo")
            if start_now_embargo:
                st.session_state.import_stop_date = datetime.date.today()
            st.session_state.import_stop_date = date_input_embargo.date_input(
                "Startdatum:",
                key="import_stop_date_key",
                value=st.session_state.import_stop_date,
                min_value=datetime.datetime(2022, 3, 15, 0, 0),
                max_value=end_date,
            )
            st.session_state.import_stop_date = datetime.datetime.fromordinal(
                st.session_state.import_stop_date.toordinal()
            )

        return reduction_import_russia


def setting_storage(
    custom_option=False, streamlit_object=st, expanded=False, compact=False,
):
    with streamlit_object.expander(
        "Füllstandvorgabe Erdgasspeicher", expanded=expanded
    ):
        # Füllstandvorgaben
        consider_gas_reserve = False
        if not compact:
            cols = streamlit_object.columns(2)
            consider_gas_reserve = streamlit_object.checkbox(
                "Füllstandvorgabe berücksichtigen³", value=False
            )
            # consider_gas_reserve = cols[0].checkbox("Füllstand vorgeben³", value=False)

            streamlit_object.markdown(
                "³ Entsprechend der EU-Verordnung (laufendes Verfahren) [COM(2022) 135](https://eur-lex.europa.eu/legal-content/DE/TXT/?uri=COM%3A2022%3A135%3AFIN&qid=1648043128482)"
            )

            custom_values = False
            if consider_gas_reserve and custom_option:
                custom_values = streamlit_object.checkbox(
                    "Benutzerdefinierte Werte", value=False
                )

            if custom_values and consider_gas_reserve:
                streamlit_object.info(
                    "Referenzwerte (COM(2022) 135):  \n 2022:  \n August 63%, September 68%, Oktober 74%, November 80%  \n 2023:  \n Febraur 43%, Mai 33%, Juli 52%"
                )
                # num_points = streamlit_object.slider(
                #     "Anzahl Stützstellen", value=3, min_value=0, max_value=15
                # )
                num_points = 15
                dates = []
                red_rates = []
                for num in range(num_points):
                    cols = streamlit_object.columns(2)

                    month = 5 + num
                    year = 2022
                    if month > 12:
                        month = month - 12
                        year = 2023
                    date = cols[0].date_input(
                        "Datum:",
                        key=f"date_{num}",
                        value=datetime.datetime(year, month, 1, 0, 0),
                        min_value=start_date,
                        max_value=end_date,
                    )
                    date = datetime.datetime.fromordinal(date.toordinal())
                    dates.append(date)
                    red_rate = cols[1].slider(
                        "Mindestfüllstand:",
                        key=f"reduction_{num}",
                        value=reserve_dict.get(date, 0),
                        min_value=0,
                        max_value=100,
                        format=format_percent,
                    )
                    red_rates.append(red_rate / 100)
                red_rates = [x * storage_cap for x in red_rates]
                st.session_state.reserve_dates = dates
                st.session_state.reserve_soc_val = red_rates
            else:
                st.session_state.reserve_dates = reserve_dates_default
                st.session_state.reserve_soc_val = reserve_soc_val_default
        return consider_gas_reserve


def setting_statusQuo_supply(
    streamlit_object=st, expanded=False, compact=False,
):
    with streamlit_object.expander("Versorgung", expanded=expanded):
        st.metric("Erdgasimport gesamt (inkl. LNG)⁴", f"{total_ng_import} TWh/a")
        st.metric(
            "Erdgasimport aus Russland (inkl. LNG)⁵", f"{total_ng_import_russia} TWh/a"
        )
        st.metric("LNG Import gesamt⁵", f"{total_lng_import} TWh/a")
        st.metric("LNG Import aus Russland⁵", f"{total_lng_import_russia} TWh/a")
        st.metric("Inländische Erdgasproduktion⁵", f"{total_ng_production} TWh/a")

        st.text("")

        st.markdown(
            "⁴ Erdgas-Bedarf EU27, 2019 (Quelle: [Eurostat Databrowser](https://ec.europa.eu/eurostat/cache/sankey/energy/sankey.html?geos=EU27_2020&year=2019&unit=GWh&fuels=TOTAL&highlight=_2_&nodeDisagg=1111111111111&flowDisagg=true&translateX=15.480270462412136&translateY=135.54626885696325&scale=0.6597539553864471&language=EN), 2022)"
        )
        st.markdown(
            "⁵ Erdgasimport/-produktion EU27, 2019 (Quelle: [Eurostat Energy Balance](https://ec.europa.eu/eurostat/databrowser/view/NRG_TI_GAS__custom_2316821/default/table?lang=en), 2022)"
        )


def setting_statusQuo_demand(
    streamlit_object=st, expanded=False, compact=False,
):
    with streamlit_object.expander("Bedarfe", expanded=expanded):
        st.metric("Nachfrage Industrie⁴", f"{total_industry_demand} TWh/a")
        st.metric(
            "Nachfrage Kraft- und Heizwerke⁴", f"{total_electricity_demand} TWh/a"
        )
        st.metric("Nachfrage Handel/Dienstleistung⁴", f"{total_ghd_demand} TWh/a")
        st.metric("Nachfrage Haushalte⁴", f"{total_domestic_demand} TWh/a")
        st.metric("Export und sonstige Nachfragen⁴", f"{total_exports_and_other} TWh/a")

        st.text("")

        st.markdown(
            "⁴ Erdgas-Bedarf EU27, 2019 (Quelle: [Eurostat Databrowser](https://ec.europa.eu/eurostat/cache/sankey/energy/sankey.html?geos=EU27_2020&year=2019&unit=GWh&fuels=TOTAL&highlight=_2_&nodeDisagg=1111111111111&flowDisagg=true&translateX=15.480270462412136&translateY=135.54626885696325&scale=0.6597539553864471&language=EN), 2022)"
        )


def message_embargo_compensation(
    add_lng_import,
    add_pl_import,
    reduction_import_russia,
    red_exp_dem,
    red_elec_dem,
    red_ind_dem,
    red_ghd_dem,
    red_dom_dem,
):
    compensation = (
        add_lng_import
        + add_pl_import
        + total_exports_and_other * red_exp_dem
        + total_electricity_demand * red_elec_dem
        + total_industry_demand * red_ind_dem
        + total_ghd_demand * red_ghd_dem
        + total_domestic_demand * red_dom_dem
    )
    compensation = int(round(compensation, 0))

    omitted = int(
        round(
            (total_pl_import_russia + total_lng_import_russia)
            * reduction_import_russia,
            0,
        )
    )
    delta = omitted - compensation

    if delta > 0:
        rel_str = "**größer** als die"
        likely = ""
    elif delta < 0:
        rel_str = "**kleiner** als die"
        likely = "un"
    else:
        rel_str = "**gleich** der"
        likely = "un"

    message = f"Der Wegfall russischer Erdgasimporte (**{omitted}** TWh/a) ist {rel_str} Kompensation durch zusätzliche LNG-Kapazitäten und Nachfragereduktionen (**{compensation}** TWh/a). Erzwungene **Abregelungen** von Erdgasbedarfen sind **{likely}wahrscheinlich**."

    if delta > 0:
        st.info(message)
    else:
        st.success(message)
