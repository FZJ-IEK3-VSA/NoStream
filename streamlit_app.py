#%%
import pandas as pd
import streamlit as st

# import matplotlib.pyplot as plt
import numpy as np

# import plotly.express as px
import plotly.graph_objects as go

# import plotly.figure_factory as ff
from get_data import *

# from PIL import Image

#%%
# Get Data
FZJcolor = get_fzjColor()
lng_df = get_lng_storage()
gng_df = get_ng_storage()

legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
# scale = 2
font_dict = dict(size=28)

write_image = True  # True False
scale = 2
width = 3000 / scale
height = 1000 / scale

# Pipelines
pl_import = get_pipeline_data()

# ng_share = get_ng_share()
# solid_fuel_share = get_solid_fuel_share()
# crude_oil_share = get_crude_oil_share()

year = 2019
xval = lng_df["gasDayStartedOn"]

### Functions


def annual_mean(df, scalefac):
    annual_mean_val = df.mean() * 365 / scalefac
    annual_mean_val = int(round(annual_mean_val, 0))
    return annual_mean_val


def get_color(key, default_col="blue"):
    return {"RU": FZJcolor.get(default_col)}.get(key, FZJcolor.get("grey1"))


def eurostat_plots(commodity, mode, df_all, region, df_single, streamlit_obj):
    unit_dict = {
        "Natural gas": "TWh",
        "LNG": "TWh",
        "Solid fuels": "Mio. t",
        "Crude oil": "Mio. t",
        "Oil products": "Mio. t",
    }
    translation_dict = {
        "Natural gas": "Erdgas",
        "Solid fuels": "Steinkohle",
        "Crude oil": "Rohöl",
        "Oil products": "Mineralölprodukte",
    }
    trans_mode_dict = {
        "import": "Import",  # "",  # "Import",
        "export": "Export",  # "",  # "Export",
        "production": "Produktion",  # "",  # "Produktion"
    }

    unit = unit_dict.get(commodity)
    fig = go.Figure()
    years = df_all.columns

    for _, row in df_all.iterrows():
        if "import" in mode.lower():
            marker_dict = dict(color=get_color(row.name))
        else:
            marker_dict = None
        fig.add_trace(
            go.Scatter(
                x=years,
                y=row.values,
                stackgroup="one",
                name=row.name,
                marker=marker_dict,
            )
        )
    # if unit == "TWh":
    fig.add_trace(
        go.Line(
            x=years,
            y=df_all.sum(axis=0),
            # stackgroup="one",
            name="Total",
            line_color=FZJcolor.get("black3"),
            # secondary_y=True,
        )
    )
    #     fig.update_yaxes(title_text="[TWh]", secondary_y=False)
    #     fig.update_yaxes(title_text="[Mrd. m3]", secondary_y=True)

    fig.update_layout(
        title=f"{translation_dict.get(commodity, commodity)} {trans_mode_dict.get(mode, mode)} [{unit}]",
        font=dict(size=16),
    )

    streamlit_obj.plotly_chart(fig, use_container_width=True)
    # scale = 3
    if write_image:
        fig.write_image(f"Output/{region}_{commodity}_{mode}_{unit}.png", scale=scale)
    # streamlit_obj.caption("Source: Eurostat, 2022")

    # Pie Chart
    try:
        if "import" in mode.lower():
            colors = [get_color(x) for x in df_single.index]
            marker_dict = dict(colors=colors)
        else:
            marker_dict = None

        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels=df_single.index,
                values=df_single.values,
                hole=0.3,
                marker=marker_dict,
            )
        )
        fig.update_layout(
            title=f"{translation_dict.get(commodity, commodity)} {trans_mode_dict.get(mode, mode)} {year} ({int(sum(df_single.values))} {unit}) [%]",
            font=dict(size=16),
        )
        streamlit_obj.plotly_chart(fig, use_container_width=True)
        if write_image:
            fig.write_image(
                f"Output/{region}_{commodity}_{mode}_{year}.png", scale=scale
            )
        # streamlit_obj.caption("Source: Eurostat, 2022")
    except:
        streamlit_obj.markdown("No data  available")


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
st.markdown(
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
)


st.markdown("## Scenario calculation: Reduction of Russian gas imports")
st.markdown(
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
)

cols = st.columns(4)

# cols[0].markdown("Reduktion Russischer Gas-Importe: 100 %")
pl_reduction = cols[0].selectbox("Anteil russischer Gas-Importe [%]", [0])

# pl_reduction = 100
# pl_reduction = cols[0].slider(
#     "Reduktion Russischer Gas-Importe [%]",
#     min_value=0,
#     max_value=100,
#     value=100,
#     step=10,
# )

reduced_demand = cols[1].selectbox("Nachfrageredutkion", ["False", "True"], 1)

lng_capacity = cols[2].selectbox(
    "Zusätzliche LNG Import Kapazität [TWh/Tag]", [0, 2.6], 1
)  # [2.4, 4.0, 5.6]

soc_slack = cols[3].selectbox("SOC Slack", ["False", "True"], 0)

df = get_optiRes(pl_reduction, lng_capacity, reduced_demand, soc_slack)

cols = st.columns(2)
# cols[0].markdown("### Supply and demand")
# Demand
total_demand = df.dom_Dem + df.elec_Dem + df.ind_Dem + df.ghd_Dem + df.exp_n_oth
total_demand_served = (
    df.dom_served + df.elec_served + df.ind_served + df.ghd_served + df.exp_n_oth_served
)
fig = go.Figure()
xvals = df.time

# fig.add_trace(
#     go.Line(
#         x=xvals,
#         y=total_demand,
#         name="Demand",
#         # mode="none",
#         line=dict(width=0.1, color=FZJcolor.get("black")),
#         # marker=marker_dict,
#     )
# )


fig.add_trace(
    go.Scatter(
        x=xvals,
        y=df.dom_served,
        stackgroup="one",
        # legendgroup="bedarf",
        name="Haushalte",
        mode="none",
        fillcolor=FZJcolor.get("green")  # green
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
        fillcolor=FZJcolor.get("purple2")  # purple
        # marker=marker_dict,
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
        # marker=marker_dict,
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
        fillcolor=FZJcolor.get("grey2"),  # lblue
        # marker=marker_dict,
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
        fillcolor=FZJcolor.get("blue2")  # blue2
        # marker=marker_dict,
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
            # marker=marker_dict,
        )
    )

fig.add_trace(
    go.Scatter(
        x=xvals,
        y=df.lngServed,  # lng_val / 24,
        stackgroup="two",
        line=dict(color=FZJcolor.get("yellow4"), width=3.5),
        # legendgroup="import",
        name="LNG Import",
        fillcolor="rgba(0, 0, 0, 0)",
        # line_color=FZJcolor.get("yellow"),
    )
)

fig.add_trace(
    go.Scatter(
        x=xvals,
        y=df.pipeServed,
        stackgroup="two",
        line=dict(color=FZJcolor.get("orange"), width=3.5),
        # legendgroup="import",
        # legendgrouptitle_text="Erdgasimport",
        name="Pipeline Import",
        fillcolor="rgba(0, 0, 0, 0)",
        # line_color=FZJcolor.get("orange"),
    )
)


fig.update_layout(
    title=f"Gedeckte Erdgasbedarfe",  # [TWh/h]
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
        # marker=marker_dict,
    )
)

fig.add_trace(
    go.Line(
        x=xvals,
        y=np.ones(len(xvals)) * 1100,
        # stackgroup="one",
        name="Maximale Kapazität",
        # mode="none",
        line_color=FZJcolor.get("black")
        # marker=marker_dict,
    )
)

fig.update_layout(
    title=f"Speicherfüllstand",
    font=font_dict,
    yaxis_title="Erdgas [TWh]",
    # legend=legend_dict,
)
fig.update_layout(showlegend=False)

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
        # marker=marker_dict,
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
        # marker=marker_dict,
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
        # marker=marker_dict,
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
        # marker=marker_dict,
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
        # marker=marker_dict,
    )
)

fig.update_layout(
    title=f"Ein- und Ausspeicherung Gasspeicher",
    font=dict(size=16),
    yaxis_title="Erdgas [TWh/h]",
    legend=legend_dict,
)

st.plotly_chart(fig, use_container_width=True)


st.text("")

st.markdown("## Energy imports, production and export by country")
st.markdown(
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
)


cols = st.columns(3)
region_list = get_eu27()
region_list = ["EU27"] + region_list
region = cols[0].selectbox("Region", region_list, 0)
balance = cols[1].multiselect("Bilanz", ["Import", "Production", "Export"], ["Import"])
year = cols[2].selectbox("Jahr", [2017, 2018, 2019, 2020], 2)
with st.spinner(text="Connecting to Eurostat database..."):
    # Import
    if "Import" in balance:
        # with st.expander("Import", expanded=True):
        st.markdown("## Import")
        ng_imports, ng_import_pie = get_eurostat_data(
            "ng", "import", region, 7, year=year
        )
        lng_imports, lng_import_pie = get_eurostat_data(
            "lng", "import", region, 7, year=year
        )
        oilCrude_imports, oilCrude_imports_pie = get_eurostat_data(
            "oilCrude", "import", region, 12, year=year
        )
        oilProducts_imports, oilProducts_imports_pie = get_eurostat_data(
            "oilProducts", "import", region, 12, year=year
        )
        sff_imports, sff_import_pie = get_eurostat_data(
            "sff", "import", region, 7, year=year
        )

        cols = st.columns(5)
        eurostat_plots(
            "Natural gas", "import", ng_imports, region, ng_import_pie, cols[0]
        )

        eurostat_plots("LNG", "import", lng_imports, region, lng_import_pie, cols[1])

        eurostat_plots(
            "Solid fuels", "import", sff_imports, region, sff_import_pie, cols[2]
        )

        eurostat_plots(
            "Crude oil",
            "import",
            oilCrude_imports,
            region,
            oilCrude_imports_pie,
            cols[3],
        )

        eurostat_plots(
            "Oil products",
            "import",
            oilProducts_imports,
            region,
            oilProducts_imports_pie,
            cols[4],
        )
        st.caption("Source: Eurostat, 2022")

    # Production
    if "Production" in balance:
        # with st.expander("Production", expanded=True):
        st.markdown("## Production")
        ng_production, ng_production_pie = get_eurostat_data(
            "ng", "production", region, 7, year=year
        )
        lng_production, lng_production_pie = get_eurostat_data(
            "lng", "production", region, 7, year=year
        )
        oilCrude_production, oilCrude_production_pie = get_eurostat_data(
            "oilCrude", "production", region, 7, year=year
        )
        oilProducts_production, oilProducts_production_pie = get_eurostat_data(
            "oilProducts", "production", region, 7, year=year
        )
        sff_production, sff_production_pie = get_eurostat_data(
            "sff", "production", region, 7, year=year
        )

        cols = st.columns(5)
        eurostat_plots(
            "Natural gas",
            "production",
            ng_production,
            region,
            ng_production_pie,
            cols[0],
        )
        eurostat_plots(
            "LNG", "production", lng_production, region, lng_production_pie, cols[1]
        )
        eurostat_plots(
            "Solid fuels",
            "production",
            sff_production,
            region,
            sff_production_pie,
            cols[2],
        )
        eurostat_plots(
            "Crude oil",
            "production",
            oilCrude_production,
            region,
            oilCrude_production_pie,
            cols[3],
        )

        eurostat_plots(
            "Oil Products",
            "production",
            oilProducts_production,
            region,
            oilProducts_production_pie,
            cols[4],
        )
    # Export
    if "Export" in balance:
        # with st.expander("Export", expanded=False):
        st.markdown("## Export")
        ng_exports, ng_export_pie = get_eurostat_data(
            "ng", "export", region, 7, year=year
        )
        lng_exports, lng_export_pie = get_eurostat_data(
            "lng", "export", region, 7, year=year
        )
        oilCrude_exports, oilCrude_exports_pie = get_eurostat_data(
            "oilCrude", "export", region, 7, year=year
        )
        oilProducts_exports, oilProducts_exports_pie = get_eurostat_data(
            "oilProducts", "export", region, 7, year=year
        )
        sff_exports, sff_export_pie = get_eurostat_data(
            "sff", "export", region, 7, year=year
        )

        cols = st.columns(5)
        eurostat_plots(
            "Natural gas", "export", ng_exports, region, ng_export_pie, cols[0]
        )
        eurostat_plots("LNG", "export", lng_exports, region, lng_export_pie, cols[1])
        eurostat_plots(
            "Solid fuels", "export", sff_exports, region, sff_export_pie, cols[2]
        )
        eurostat_plots(
            "Crude oil",
            "export",
            oilCrude_exports,
            region,
            oilCrude_exports_pie,
            cols[3],
        )

        eurostat_plots(
            "Oil Products",
            "export",
            oilProducts_exports,
            region,
            oilProducts_exports_pie,
            cols[4],
        )


# Pipeline Flow
st.markdown("## Physical pipeline flow of natural gas")

st.markdown(
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
)

fig = go.Figure()
date = pl_import.columns
for _, row in pl_import.iterrows():
    country = row.name[-3:-1]
    marker_dict = dict(color=get_countryColor(country, FZJcolor))
    fig.add_trace(
        go.Scatter(
            x=date, y=row.values, stackgroup="one", name=row.name, marker=marker_dict,
        )
    )

fig.update_layout(
    title="Pipeline flow from Russia to EU",
    yaxis_title="NG [GWh/d]",
    yaxis=dict(range=[0, 7000]),
    font=font_dict,
    legend=legend_dict,
    barmode="stack",
)
fig.update_layout(hovermode="x unified")


st.plotly_chart(fig, use_container_width=True)
st.caption("Source: ENTSOG, 2022")


st.markdown("## Storages")

st.markdown(
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
)


col1, col2 = st.columns(2)
col1.markdown("### Liquefied Natural Gas (LNG)")
col2.markdown("### Natural Gas (NG)")

############
###  LNG
############


# Plot inventory LNG
fig = go.Figure()
fig.add_trace(
    go.Line(
        x=xval,
        y=lng_df["dtmi"],  # dtmi_median
        name="Max capacity",
        marker=dict(color=FZJcolor.get("black3")),
    )
)


fig.add_trace(
    go.Scatter(
        x=xval,
        y=lng_df["lngInventory"],
        name="State of charge",
        marker=dict(color=FZJcolor.get("blue")),
        # mode="lines",
        # line=dict(width=0),
        fill="tozeroy",
    )
)


# fig.add_trace(
#     go.Scatter(
#         x=xval,
#         y=lng_df["dtmi_median"],
#         name="Free capacity",
#         marker=dict(color=FZJcolor.get("green")),
#         mode="lines",
#         line=dict(width=0),
#         fill="tonexty",
#     )
# )

fig.update_layout(
    title="Storage level of LNG facilities in the EU",
    yaxis_title="LNG [TWh]",
    yaxis=dict(range=[0, 60]),
    font=font_dict,
    legend=legend_dict,
)
col1.plotly_chart(fig, use_container_width=True)
col1.caption("Source: GIE, 2022")

# # Plot free capacity LNG
# fig = go.Figure()
# fig.add_trace(
#     go.Line(
#         x=xval,
#         y=lng_df["dtmi_median"],
#         name="Max capacity",
#         marker=dict(color=FZJcolor.get("black")),
#     )
# )
# fig.add_trace(
#     go.Scatter(
#         x=xval,
#         y=lng_df["free_inventory"],
#         name="Free capacity",
#         marker=dict(color=FZJcolor.get("green")),
#         fill="tozeroy",
#     )
# )

# fig.update_layout(
#     title="Spare LNG storage capacity (Max capacity - State of charge)",
#     yaxis_title="LNG [TWh]",
#     yaxis=dict(range=[0, 60]),
#     font=font_dict,
#     legend=legend_dict,
# )
# col1.plotly_chart(fig, use_container_width=True)
# col1.caption("Source: GIE, 2022")


# Send Out
fig = go.Figure()
fig.add_trace(
    go.Line(
        x=xval,
        y=lng_df["dtrs"],  # dtrs_median
        name=f"Max send out (Ø {int(lng_df['dtrs_median'].mean()*365/10**3)} TWh/a)",
        marker=dict(color=FZJcolor.get("black")),
    )
)
fig.add_trace(
    go.Scatter(
        x=xval,
        y=lng_df["sendOut"],
        name=f"Send out rate (Ø {int(lng_df['sendOut'].mean()*365/10**3)} TWh/a)",
        marker=dict(color=FZJcolor.get("blue")),
    )
)


fig.update_layout(
    title="Send out of LNG",
    yaxis_title="LNG [GWh/d]",
    yaxis=dict(range=[0, 7000]),
    font=font_dict,
    legend=legend_dict,
)
col1.plotly_chart(fig, use_container_width=True)
col1.caption("Source: GIE, 2022")

############
###  NG
############

# Plot NG
fig = go.Figure()
xval_gng = lng_df["gasDayStartedOn"]
fig.add_trace(
    go.Line(
        x=xval,
        y=gng_df["workingGasVolume"],  # workingGasVolume_median
        name="Max capacity",
        marker=dict(color=FZJcolor.get("black")),
    )
)
fig.add_trace(
    go.Scatter(
        x=xval_gng,
        y=gng_df["gasInStorage"],
        name="State of charge",
        marker=dict(color=FZJcolor.get("orange")),
        fill="tozeroy",
    )
)

fig.update_layout(
    title="Storage level NG in the EU",
    yaxis_title="NG [TWh]",
    yaxis=dict(range=[0, 1200]),
    font=font_dict,
    legend=legend_dict,
)

col2.plotly_chart(fig, use_container_width=True)
col2.caption("Source: GIE, 2022")

# # Plot NG free
# fig = go.Figure()
# xval_gng = lng_df["gasDayStartedOn"]
# fig.add_trace(
#     go.Line(
#         x=xval,
#         y=gng_df["workingGasVolume_median"],
#         name="Max capacity",
#         marker=dict(color=FZJcolor.get("black")),
#     )
# )
# # fig.add_trace(go.Bar(x=xval_gng, y=gng_df["gasInStorage"], name="State of charge", marker=dict(color= rgb_to_hex(FZJcolor.orange))))
# fig.add_trace(
#     go.Scatter(
#         x=xval_gng,
#         y=gng_df["free_cap"],
#         name="Free capacity",
#         marker=dict(color=FZJcolor.get("green")),
#         fill="tozeroy",
#     )
# )

# fig.update_layout(
#     title="Spare NG storage capacity (Max capacity - State of charge)",
#     yaxis_title="NG [TWh]",
#     yaxis=dict(range=[0, 1200]),
#     font=font_dict,
#     legend=legend_dict,
# )
# col2.plotly_chart(fig, use_container_width=True)
# col2.caption("Source: GIE, 2022")

# Withdrawal
fig = go.Figure()
fig.add_trace(
    go.Line(
        x=xval,
        y=gng_df["withdrawalCapacity"],  # withdrawalCapacity_median
        name=f"Max withdrawl (Ø {int(gng_df['withdrawalCapacity_median'].mean()*365/10**3)} TWh/a)",
        marker=dict(color=FZJcolor.get("black")),
    )
)
fig.add_trace(
    go.Scatter(
        x=xval,
        y=gng_df["withdrawal"],
        name=f"Withdrawl rate (Ø {int(gng_df['withdrawal'].mean()*365/10**3)} TWh/a)",
        marker=dict(color=FZJcolor.get("orange")),
    )
)

fig.update_layout(
    title="Withdrawal of NG",
    yaxis_title="NG [GWh/d]",
    yaxis=dict(range=[0, 22000]),
    font=font_dict,
    legend=legend_dict,
)
col2.plotly_chart(fig, use_container_width=True)
col2.caption("Source: GIE, 2022")
