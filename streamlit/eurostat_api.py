# %%
import eurostat
import pandas as pd
import streamlit as st

# nrg: natural Gas
# bal: balance
# sd: Sankey Diagram
# SCIE codes: https://dd.eionet.europa.eu/vocabulary/eurostat/siec/view
# Eurostat api: https://pypi.org/project/eurostat/

# searching for tables
# toc_df = eurostat.get_toc_df()
# f = eurostat.subset_toc_df(toc_df, 'fleet')

KTOE_to_TWH = 0.011630000  # TWh/KTOE


@st.experimental_memo(show_spinner=False)
def get_eurostat_data(year, spacial_scope, nrg_bal, siec="G3000", unit="KTOE"):
    if spacial_scope == "EU":
        spacial_scope = "EU27_2020"

    # G3000_C0350
    KTOE_to_TWH = 0.011630000  # TWh/KTOE

    if isinstance(nrg_bal, list):
        nrg_bal_list = nrg_bal
    elif (nrg_bal, str):
        nrg_bal_list = [nrg_bal]
    assert isinstance(nrg_bal_list, list), "nrg_bal must be a list or a string"

    filter_pars = {
        "GEO": [spacial_scope],
        "SIEC": [siec],
        "UNIT": [unit],
        "NRG_BAL": nrg_bal_list,
    }

    table = "nrg_bal_sd"  # Sankey diagram natural gas
    data = eurostat.get_sdmx_data_df(
        table, year, year, filter_pars, flags=False, verbose=False
    )

    # convert str to float
    data.loc[:, year] = data.loc[:, year].astype(float)

    value = data.loc[:, year].sum()

    if unit == "KTOE":
        value = value * KTOE_to_TWH

    return value


@st.experimental_memo(show_spinner=False)
def get_sector_data(spacial_scope, sector, year=2019):
    # "IMP"                 # import
    # "PPRD"                # primary production
    # "FC_IND_E",           # final consumpttion industry
    # "TI_NRG_FC_IND_NE",   # final non-energy consumption industry
    # "FC_OTH_HH_E",        # final consumption households
    # "FC_OTH_CP_E",        # final consumption services
    # "TI_EHG_E",           # transformation input electricity and heat generation
    # "NRG_E",

    balance_dict = {
        "import": ["IMP", "STATDIFF_INFLOW"],
        "production": ["PPRD"],
        "export": ["EXP"],
        "industry": ["FC_IND_E", "TI_NRG_FC_IND_NE"],
        "hh": ["FC_OTH_HH_E"],
        "ghd": ["FC_OTH_CP_E"],
        "energy": ["TI_EHG_E", "NRG_E"],
        "sources": ["IMP", "PPRD", "STATDIFF_INFLOW", "STK_DRW"],
        "covered_sinks": [
            "FC_IND_E",  # final consumpttion industry
            "FC_OTH_HH_E",  # final consumption households
            "FC_OTH_CP_E",  # final consumption services
            "TI_NRG_FC_IND_NE",  # final non-energy consumption industry
            "TI_EHG_E",
            "NRG_E",
        ],
        "remaining_sinks": [
            "EXP",
            "STATDIFF_OUTFLOW",
            "STK_BLD",
            "FC_TRA_E",
            "FC_OTH_NSP_E",  # final consumption transport
            "FC_OTH_AF_E",
            "DL",
            "TI_GW_E",
            "TI_CO_E",
            "TI_OTH",
        ],
        # "export_and_others": [
        #     "FC_TRA_E",
        #     "FC_TRA_E",
        #     "STK_BLD",
        # ],  # final consumption transport
    }

    if isinstance(sector, list):
        value = get_eurostat_data(year, spacial_scope, sector)
    elif sector == "export_and_others":
        # sources = get_eurostat_data(year, spacial_scope, balance_dict.get("sources"))
        # sinks = get_eurostat_data(year, spacial_scope, balance_dict.get("sinks"))
        # value = sources - sinks
        value = get_eurostat_data(
            year, spacial_scope, balance_dict.get("remaining_sinks")
        )
    else:
        value = get_eurostat_data(year, spacial_scope, balance_dict.get(sector, sector))

    if value == float("nan"):
        print(spacial_scope, sector, year)
        value = 0
    return value


# 'IMP', 'EXP', 'PPRD', 'PRD',
# 'NTI',            # transformation input
# 'AAS'             # available from all sources
# 'IDCO','FDCO'     # direct carry-over
# 'AAT'             # Available after transformation
# 'DL',             # distribution losses

### Transformation Input
# 'TI_E',
#   'TI_EHG_E',         # Electricity and heat generation input
#       'TI_EHG_CHP_E', 'TI_EHG_EONL_E', 'TI_EHG_HONL_E',
#   'TI_GW_E', 'TI_CO_E',   # Gas works, coke oven
#   'TI_NRG_FC_IND_NE',     # final non-energy consumption Industry
#   'TI_OTH',       # other transformation

### final consumption
# 'FC',
#   'FC_E',         # final energy consumption
#       'FC_TRA_E'  # transportation
#           'FC_TRA_PIPE_E', 'FC_TRA_ROAD_E',
#       'FC_OTH_E'  # other
#           'FC_OTH_HH_E', 'FC_OTH_NSP_E', 'FC_OTH_AF_E', 'FC_OTH_CP_E',
#       'FC_IND_E', # industry
#           'FC_IND_WP_E', 'FC_IND_MQ_E', 'FC_IND_CON_E', 'FC_IND_TL_E', 'FC_IND_IS_E', 'FC_IND_PPP_E', 'FC_IND_MAC_E', 'FC_IND_FBT_E', 'FC_IND_NMM_E', 'FC_IND_CPC_E', 'FC_IND_TE_E', 'FC_IND_NFM_E', 'FC_IND_NSP_E',
#   'FC_NE',

### consumption of the energy branch
# 'NRG_E',
#   'NRG_PR_E', 'NRG_NSP_E', 'NRG_GW_E', 'NRG_CM_E', 'NRG_CO_E', 'NRG_OIL_NG_E',

### Stocks
# 'STK_DRW', 'STK_BLD',

# Statistical difference
# 'STATDIFF_OUTFLOW', 'STATDIFF_INFLOW',


def natural_gas_import(spacial_scope, commodity, partner, unit="TJ_GCV", year=2019):
    if isinstance(partner, list):
        partner_list = partner
    elif (partner, str):
        partner_list = [partner]
    assert isinstance(partner_list, list), "nrg_bal must be a list or a string"

    siec_dict = {"ng": "G3000", "lng": "G3200"}
    if spacial_scope == "EU":
        spacial_scope = "EU27_2020"

    TJ_to_TWH = 1 / 3600  # TWh/TJ

    filter_pars = {
        "GEO": [spacial_scope],
        "SIEC": [siec_dict.get(commodity)],
        "UNIT": [unit],
        "PARTNER": partner_list,
    }

    table = "nrg_ti_gas"
    data = eurostat.get_sdmx_data_df(
        table, year, year, filter_pars, flags=False, verbose=False
    )

    data.loc[:, year] = data.loc[:, year].astype(float)

    value = data[year].sum()

    if "TJ" in unit:
        value = value * TJ_to_TWH

        if "GCV" in unit:
            # Brennwert -> Heizwert
            value = value / 1.11111

    return value
