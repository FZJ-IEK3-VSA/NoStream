from requests import Session
import pandas as pd
import numpy as np
import datetime
import streamlit as st


@st.experimental_memo(show_spinner=False)
def api_call(spacial_scope, start_day, end_day):
    with open("static/private_keys/gie_api_key.txt") as f:
        lines = f.readlines()
    api_key = lines[0]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Host": "agsi.gie.eu",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "TE": "trailers",
        "Upgrade-Insecure-Requests": "1",
        "x-key": api_key,
    }

    session = Session()
    url_meta = f"https://agsi.gie.eu/api/data/{spacial_scope}"

    # Create dataframe with data from first day
    payload = {}
    payload["from"] = start_day
    payload["till"] = end_day
    payload["limit"] = "9999"

    r2 = session.get(url_meta, params=payload, headers=headers)
    data_json = r2.json()
    data_df = pd.json_normalize(data_json["data"])
    return data_df


@st.experimental_memo(show_spinner=False)
def get_storage_capacity(spacial_scope, today):

    start_day = datetime.date(today.year, 1, 1)
    end_day = today.date() - datetime.timedelta(days=2)

    data_df = api_call(spacial_scope, start_day, end_day)

    soc_fix_day = data_df.gasInStorage.astype(float)

    # reverse order oldest on top
    soc_fix_day = soc_fix_day.iloc[::-1]

    # For every hour the same value
    # TODO interpolation instead of same value for every hour
    soc_fix_hour = np.array(24 * [soc_fix_day])

    # reorder the matrix to a vector
    soc_fix_hour = np.ravel(soc_fix_hour, order="F")

    # convert to list
    soc_fix_hour = soc_fix_hour.tolist()

    return soc_fix_hour


# get_storage_capacity("DE", datetime.datetime.today())


@st.experimental_memo(show_spinner=False)
def get_max_storage_capacity(spacial_scope):
    start_day = datetime.date(2021, 1, 1)
    end_day = datetime.date(2021, 12, 31)

    data_df = api_call(spacial_scope, start_day, end_day)

    wgv = data_df.workingGasVolume.astype(float)

    max_storage_capacity = wgv.mean()

    return max_storage_capacity


# get_storage_capacity("DE", datetime.datetime.today())
