from requests import Session
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import json

# GCV_to_LCV = 1.1107

def fetch(session, url, payload, headers):
    try:
        result = session.get(url, params=payload, headers=headers)
        return result.json()
    except Exception:
        return {}


@st.experimental_memo(show_spinner=False)
def api_call(spacial_scope, start_day, end_day):
    with open("static/private_keys/gie_api_key.txt") as f:
        lines = f.readlines()
    api_key = lines[0]

    # convert datetime to string
    start_day_str = (start_day.strftime("%Y-%m-%d")).replace("-0", "-")
    end_day_str = (end_day.strftime("%Y-%m-%d")).replace("-0", "-")

    # days between start and end day
    # timedelta = end_day - start_day
    # timedelta = (timedelta.days + 1)*2
    timedelta = 730

    session = Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
        "x-key": api_key,
    }

    payload = {}
    payload["from"] = start_day
    payload["till"] = end_day
    payload["limit"] = str(timedelta)

    if "eu" in spacial_scope.lower():
        # url = f"https://agsi.gie.eu/api?type={spacial_scope}&from={start_day_str}&to={end_day_str}&size={timedelta}"  # &size={timedelta}"
        url = f"https://agsi.gie.eu/api/data/{spacial_scope}"
    else:
        # TODO check if size argument is breaking api
        url = f"https://agsi.gie.eu/api?country={spacial_scope}&from={start_day_str}&to={end_day_str}&size={timedelta}"

    try:
        request = session.get(url, params=payload, headers=headers)
        data_json = request.json()
        data_df = pd.json_normalize(data_json["data"])
    except:
        # TODO reload data if api call fails
        data_df = pd.DataFrame()
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

    return soc_fix_hour # unit TWh GCV


@st.experimental_memo(show_spinner=False)
def get_max_storage_capacity(spacial_scope):
    # period for which the average gas storage capacity should be derived from
    start_day = datetime.date(2021, 1, 1)
    end_day = datetime.date(2021, 12, 31)

    data_df = api_call(spacial_scope, start_day, end_day)

    wgv = data_df.workingGasVolume.astype(float)

    max_storage_capacity = wgv.mean()

    return max_storage_capacity # unit TWh GCV
