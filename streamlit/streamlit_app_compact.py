# %%
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

compact = True  # Use compact version of the app

### Streamlit App
st.set_page_config(
    page_title="NoStream",
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

st.markdown("# NoStream: Erdgas Energy Dashboard")


with st.sidebar:
    # Logo
    se.centered_fzj_logo_top()

    st.text("")

    # Further links and information
    st.text("")
    se.sidebar_further_info()


# Energiebilanz
slot_figure = st.empty()
slot_settings = st.empty()

# Embarge und Kompensation
reduction_import_russia = se.setting_embargo(compact=compact, expanded=True)

# Compensation
(
    red_ind_dem,
    red_elec_dem,
    red_ghd_dem,
    red_dom_dem,
    red_exp_dem,
    add_lng_import,
    add_pl_import,
) = se.setting_compensation(
    streamlit_object=slot_settings, compact=compact, expanded=True
)


# Importlücke Plot
se.plot_import_gap(
    reduction_import_russia,
    red_exp_dem,
    red_dom_dem,
    red_elec_dem,
    red_ghd_dem,
    red_ind_dem,
    add_lng_import,
    add_pl_import,
    streamlit_object=slot_figure,
    compact=compact,
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

st.text("")
se.centered_fzj_logo()
st.text("")

st.markdown(
    "<center> 👉  <a href='https://no-stream.fz-juelich.de/'> Zum vollständigen Tool </a> </center>",
    unsafe_allow_html=True,
)
st.text("")
