#%%
import pandas as pd
import streamlit as st

# import matplotlib.pyplot as plt
import numpy as np

# import plotly.express as px
import plotly.graph_objects as go

# import plotly.figure_factory as ff
from get_data import *

# import rusngstorage.storage_sim as opti
import storage_sim as opti
# from PIL import Image

#%%
# Get Data
FZJcolor = get_fzjColor()
# lng_df = get_lng_storage()
# gng_df = get_ng_storage()

legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
# scale = 2
font_dict = dict(size=28)

write_image = True  # True False
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
st.markdown("# Energy imports from Russia and possible alternatives")


st.markdown("## Scenario calculation: Reduction of Russian gas imports")

cols = st.columns(3)
# pl_reduction = cols[0].selectbox("Anteil russischer Gas-Importe [%]", [0])
pl_reduction = cols[0].slider("Anteil Russisches Gas an Importen [%]", min_value=0, max_value=100, value=0, step=5)

reduced_demand = cols[1].selectbox("Nachfrageredutkion", ["False", "True"], 1)

lng_capacity = cols[2].selectbox(
    "Zusätzliche LNG Import Kapazität [TWh/Tag]", [0, 2.6], 1
)  # [2.4, 4.0, 5.6]

# soc_slack = cols[3].selectbox("SOC Slack", ["False", "True"], 0)

start_opti = st.button("Start optimization")

if start_opti:
    with st.spinner(text="Running optimization..."):
        opti.run_scenario(russ_share=int(pl_reduction), lng_val=float(lng_capacity), demand_reduct=bool(reduced_demand))



    with st.spinner(text="Fetching results..."):
        df = get_optiRes(pl_reduction, lng_capacity, reduced_demand, soc_slack)

        # cols = st.columns(2)

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
                # legendgroup="bedarf",
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
                # legendgroup="bedarf",
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
                # legendgroup="bedarf",
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
                # legendgroup="bedarf",
                # legendgrouptitle_text="Erdgasbedarfe",
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
                # legendgroup="bedarf",
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
                    # legendgroup="bedarf",
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
                # legendgroup="import",
                # legendgrouptitle_text="Erdgasimport",
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
                # legendgroup="import",
                name="Pipeline Import",
                fillcolor="rgba(0, 0, 0, 0)",
            )
        )


        fig.update_layout(
            title=f"Erdgasbedarfe und Import",
            font=font_dict,
            yaxis_title="Erdgas [TWh/h]",
            legend=legend_dict,
        )
        fig.update_layout(showlegend=False)

        if write_image:
            fig.write_image(
                f"Output/Optimierung_Erdgasbedarf_{pl_reduction}_{lng_capacity}_{reduced_demand}_{soc_slack}.png",
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
                fillcolor=FZJcolor.get("orange")
            )
        )

        fig.add_trace(
            go.Line(
                x=xvals,
                y=np.ones(len(xvals)) * 1100,
                name="Maximale Kapazität",
                line_color=FZJcolor.get("black")
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
                f"Output/Optimierung_Speicher_{pl_reduction}_{lng_capacity}_{reduced_demand}_{soc_slack}.png",
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
                fillcolor=FZJcolor.get("blue3")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=df.lngServed,  # lng_val / 24,
                stackgroup="one",
                name="LNG Import",
                mode="none",
                fillcolor=FZJcolor.get("yellow")
            )
        )

        fig.update_layout(
            title=f"Erdgasimporte",
            font=dict(size=16),
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
                fillcolor=FZJcolor.get("yellow")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=storage_charge_lng,
                stackgroup="one",
                name="Speicherung (LNG)",
                mode="none",
                fillcolor=FZJcolor.get("blue3")
            )
        )

        fig.update_layout(
            title=f"Ein- und Ausspeicherung Gasspeicher",
            font=dict(size=16),
            yaxis_title="Erdgas [TWh/h]",
            legend=legend_dict,
        )

        st.plotly_chart(fig, use_container_width=True)