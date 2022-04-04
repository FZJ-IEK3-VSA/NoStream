import pandas as pd
import os
import base64
import streamlit as st

APP_DIR = os.path.dirname(os.path.realpath(__file__))


def get_scenario_name(pl_reduction, lng_add_capacity, reduced_demand, soc_slack):
    return (
        f"{int(pl_reduction*100)}_{int(lng_add_capacity)}_{reduced_demand}_{soc_slack}"
    )


def get_result_dir(scenario_name):
    if "default" in scenario_name:
        fileName = f"default_scenario.xlsx"
    else:
        fileName = f"results_aGasFlowScen_{scenario_name}.xlsx"
    fileDir = os.path.join("static/results", fileName)
    return fileDir


def results_exists(scenario_name):
    fileDir = get_result_dir(scenario_name)
    return os.path.isfile(fileDir)


def get_optiRes(scenario_name):
    fileDir = get_result_dir(scenario_name)
    df = pd.read_excel(fileDir, index_col=0)
    df.fillna(0, inplace=True)
    df.time = pd.to_datetime(df.time)
    return df


def get_fzjColor():
    FZJcolor = pd.read_csv(os.path.join("static", "FZJcolor.csv"))

    def rgb_to_hex(reg_vals):
        def clamp(x):
            return int(max(0, min(x * 255, 255)))

        return "#{0:02x}{1:02x}{2:02x}".format(
            clamp(reg_vals[0]), clamp(reg_vals[1]), clamp(reg_vals[2])
        )

    col_names = FZJcolor.columns
    hex_vals = [rgb_to_hex(FZJcolor.loc[:, col]) for col in col_names]

    FZJcolor_dict = dict(zip(col_names, hex_vals))
    return FZJcolor_dict


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