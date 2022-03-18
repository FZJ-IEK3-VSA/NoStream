#%%
import pandas as pd
import streamlit as st
import numpy as np

import plotly.graph_objects as go
import datetime
import get_data as gdta
import storage_sim as opti
# from PIL import Image

import os

#%%
# Get Data
FZJcolor = gdta.get_fzjColor()

legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
# scale = 2
font_dict = dict(size=24)

write_image = False  # True False
scale = 2
width = 3000 / scale
height = 1000 / scale


def annual_mean(df, scalefac):
    annual_mean_val = df.mean() * 365 / scalefac
    annual_mean_val = int(round(annual_mean_val, 0))
    return annual_mean_val


def get_color(key, default_col="blue"):
    return {"RU": FZJcolor.get(default_col)}.get(key, FZJcolor.get("grey1"))



font_size = 18

### Streamlit App
st.set_page_config(
    page_title="Energy Independence", page_icon="🇪🇺", layout="wide"  # layout="wide" 🚢
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.text("")
st.markdown("# Reduktion Russischer Gas-Importe")
st.markdown("## Auswirkungen auf die Versorgungssicherheit in Europa")

with st.expander("Importstopp", expanded=False):
    cols = st.columns(2)
    pl_reduction = cols[1].slider("Reduktion der russischer Erdgas-Importe um [%]", min_value=100, max_value=0, value=100, step=1)
    pl_reduction = 1 - pl_reduction/100

    import_stop_date = cols[0].date_input("Beginn der Importreduktion", value=datetime.date(2022, 3, 15), min_value=datetime.date(2022, 3, 15), max_value=datetime.date(2023, 12, 31))
    st.markdown("Erdgasimporte 2019 in die EU27: 4190 TWh/a, Produktion 2019 innerhalb der EU27: 608 TWh/a, Import aus Russland 2019: 1752 TWh/a (Quelle: Eurostat 2022)")

with st.expander("Nachfrageredutkion", expanded=False):
    cols = st.columns(2)
    # cols[0].markdown("Gesamt Erdgasbedarf (EU27, 2019): 3972 TWh/a")
    demand_reduction_date = cols[0].date_input("Beginn der Nachfragereduktion", value=datetime.date(2022, 3, 15), min_value=datetime.date(2022, 3, 15), max_value=datetime.date(2023, 12, 31))

    cols = st.columns(2)
    total_domestic_demand = cols[0].number_input("Nachfrage Haushalte¹ [TWh/a]", min_value=0, max_value=None, value=926)
    red_dom_dem = cols[1].slider("Reduktion der Nachfrage um [%]", key="red_dom_dem", min_value=0, max_value=100, value=13, step=1)

    cols = st.columns(2)
    total_ghd_demand = cols[0].number_input("Nachfrage GHD¹ [TWh/a]", min_value=0, max_value=None, value=421) #420.5
    red_ghd_dem = cols[1].slider("Reduktion der Nachfrage um [%]", key="red_ghd_dem", min_value=0, max_value=100, value=8, step=1)

    cols = st.columns(2)
    total_electricity_demand = cols[0].number_input("Nachfrage Energie-Sektor¹ [TWh/a]", min_value=0, max_value=None, value=1515)
    red_elec_dem = cols[1].slider("Reduktion der Nachfrage um [%]", key="red_elec_dem", min_value=0, max_value=100, value=20, step=1)

    cols = st.columns(2)
    total_industry_demand = cols[0].number_input("Nachfrage Industrie¹ [TWh/a]", min_value=0, max_value=None, value=1110)
    red_ind_dem = cols[1].slider("Reduktion der Nachfrage um [%]", key="red_ind_dem", min_value=0, max_value=100, value=8, step=1)
    st.markdown("¹ Standardwerte entsprechen den EU27 Erdgasbedarfen von 2019, zuzüglich 988 TWh/a Export und sonstige Bedarfe. Gesamterdgasbedarf 2019: 4961 TWh/a (Quelle: Eurostat 2022)")

with st.expander("LNG Kapazitäten", expanded=False):
    cols = st.columns(2)
    lng_capacity = cols[1].number_input("Zusätzliche LNG Import Kapazität² [TWh/a]", min_value=0, max_value=1150, value=965)
    lng_increase_date = cols[0].date_input("Beginn der LNG Kapazität-Erhöhung", value=datetime.date(2022, 3, 15), min_value=datetime.date(2022, 3, 15), max_value=datetime.date(2023, 12, 31))
    st.markdown("² Aktuell genutzte LNG Kapazität: 875 TWh/a, Maximal nutzbare LNG Kapazitäten: 2025 TWh/a, Maximal zusätzlich nurzbare LNG-Kapazitäten: 1150 TWh/a (Quelle: GIE 2022)")


cols = st.columns(3)
pl_reduction = cols[0].selectbox("Anteil russischer Gas-Importe [%]", [0])


reduced_demand = cols[1].selectbox("Nachfrageredutkion", ["False", "True"], 1)

lng_capacity = cols[2].selectbox(
    "Zusätzliche LNG Import Kapazität [TWh/Tag]", [0, 2.6], 1
)  # [2.4, 4.0, 5.6]

# soc_slack = cols[3].selectbox("SOC Slack", ["False", "True"], 0)
soc_slack = False

start_opti = st.button("Start optimization")

def plot_optimization_results(df):
    # Demand
    total_demand = df.dom_Dem + df.elec_Dem + df.ind_Dem + df.ghd_Dem + df.exp_n_oth
    total_demand_served = (
        df.dom_served + df.elec_served + df.ind_served + df.ghd_served + df.exp_n_oth_served
    )
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
            fillcolor=FZJcolor.get("purple2")
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
            fillcolor=FZJcolor.get("blue")
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
            fillcolor=FZJcolor.get("blue2")
        )
    )

    unserved_demand = total_demand - total_demand_served
    st.markdown(f"Abgeregelte Menge Erdgas: {int(sum(unserved_demand))} TWh")

    if sum(unserved_demand) > 0.001:
        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=total_demand - total_demand_served,
                stackgroup="one",
                legendgroup="bedarf",
                name="Ungedeckter Bedarf",
                mode="none",
                fillcolor=FZJcolor.get("red"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.lngServed,  # lng_val / 24,
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
        # font=font_dict,
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


    #%%
    # SOC
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.soc,
            stackgroup="one",
            name="Füllstand",
            mode="none",
            fillcolor=FZJcolor.get("black3")
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
        # font=font_dict,
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


    #%%
    # Pipeline Import
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.pipeServed,
            stackgroup="one",
            name="Pipeline Import",
            mode="none",
            fillcolor=FZJcolor.get("orange")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=df.lngServed,  # lng_val / 24,
            stackgroup="one",
            name="LNG Import",
            mode="none",
            fillcolor=FZJcolor.get("yellow3")
        )
    )

    fig.update_layout(
        title=f"Erdgasimporte",
        # font=dict(size=16),
        yaxis_title="Erdgas [TWh/h]",
        legend=legend_dict,
    )
    st.plotly_chart(fig, use_container_width=True)


    #%%
    # Storage Charge and discharge
    storage_operation = df.lngServed + df.pipeServed - total_demand_served
    storage_discharge = [min(0, x) for x in storage_operation]
    storage_charge = np.array([max(0, x) for x in storage_operation])

    storage_operation_pl = df.pipeServed - total_demand_served
    storage_charge_pl = np.array([max(0, x) for x in storage_operation_pl])

    storage_operation_lng = storage_charge - storage_charge_pl
    # storage_operation_lng = df.pipeServed - total_demand_served
    storage_charge_lng = np.array([max(0, x) for x in storage_operation_lng])


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=storage_discharge,
            stackgroup="two",
            name="Ausspeicherung",
            mode="none",
            fillcolor=FZJcolor.get("red")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=storage_charge_pl,
            stackgroup="one",
            name="Speicherung (Pipeline)",
            mode="none",
            fillcolor=FZJcolor.get("orange")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=storage_charge_lng,
            stackgroup="one",
            name="Speicherung (LNG)",
            mode="none",
            fillcolor=FZJcolor.get("yellow3")
        )
    )

    fig.update_layout(
        title=f"Ein- und Ausspeicherung Gasspeicher",
        # font=dict(size=16),
        yaxis_title="Erdgas [TWh/h]",
        legend=legend_dict,
    )

    st.plotly_chart(fig, use_container_width=True)

with st.spinner(text="Running optimization, this might take several minutes..."):
    if start_opti:
        scenario_name = gdta.get_scenario_name(pl_reduction, lng_capacity, reduced_demand, soc_slack)
        try:
            opti.run_scenario(russ_share=pl_reduction, lng_val=lng_capacity, demand_reduct=bool(reduced_demand), use_soc_slack=soc_slack)
            df = gdta.get_optiRes(scenario_name)
            plot_optimization_results(df)
        except Exception as e:
            st.write(e)
    else:
        scenario_name = "default_scenario"
        df = gdta.get_optiRes(scenario_name)
        plot_optimization_results(df)

        