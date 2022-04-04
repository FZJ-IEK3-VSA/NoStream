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


FZJcolor = ut.get_fzjColor()
legend_dict = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)

font_dict = dict(size=16)

compact = True  # Use compact version of the app


### Streamlit App
st.set_page_config(
    page_title="No Stream",
    page_icon="🇪🇺",
    layout="centered",  # wide centered
    initial_sidebar_state="collapsed",  # expanded
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("# No Stream: Erdgas Energy Dashboard")


with st.sidebar:
    cols = st.columns([2, 6])
    svg_image = (
        r'<a href="https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html">'
        + ut.render_svg("static/FJZ IEK-3.svg")
        + r"</a>"
    )
    cols[0].write(svg_image, unsafe_allow_html=True)
    st.text("")

    st.text("")
    st.markdown(
        "⛲ [Quellcode der Optimierung](https://github.com/FZJ-IEK3-VSA/NoStream/blob/develop/streamlit/optimization.py)"
    )

    st.markdown(
        "🌎 [Zur Institutsseite (IEK-3)](https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html)"
    )

    st.markdown(
        "📜 [Impressum](https://www.fz-juelich.de/portal/DE/Service/Impressum/impressum_node.html)"
    )


# Energiebilanz

# Embarge und Kompensation
import_stop_date, reduction_import_russia = se.setting_embargo(
    compact=compact, expanded=True,
)

# Compensation
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
) = se.setting_compensation(compact=compact, expanded=True)


## Importlücke Plot
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
    # streamlit_object=cols[0],
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
