# %%
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
import math
import pyomo.environ as pyomo
import pyomo.opt as opt
import time
import warnings
import copy
import streamlit as st

# import matplotlib
# plt.rc('font', size=16)
# plt.rc('xtick', labelsize=16)
# plt.rc('ytick', labelsize=16)
# plt.rc('axes', titlesize=16)
# plt.rc('axes', titlesize=16)
# plt.rc('legend', fontsize=16)
# plt.rc('font', size=16)

# matplotlib.rcParams['font.family'] = 'arial'
# %%
# %matplotlib inline


periods_per_year = 8760



# Sensitivity analysis
# import share of russion gas [-]
russian_gas_share = [0.0]
# Average European LNG import [TWh/d]
base_lng_import = 2.4
lng_values = [0.0, 2.64] # 90% load # [0.0, 1.6, 3.2]
# Demand reduction
demand_reduction = [True, False]

# Relative demand reduction of the different sectors [-]
red_dom_dem = 0.13  # 13%
red_elec_dem = 0.20  # 59% -
red_ghd_dem = 0.08  # 8%
red_ind_dem = 0.08  # 8%


# Discounting factor [-/h]
fac = (1/1.06)**(1/8760)


# Energy balance
# EUROSTAT 2019 (sankey)
# https://ec.europa.eu/eurostat/cache/sankey/energy/sankey.html?geos=EU27_2020&year=2019&unit=GWh&fuels=TOTAL&highlight=_2_&nodeDisagg=1111111111111&flowDisagg=true&translateX=15.480270462412136&translateY=135.54626885696325&scale=0.6597539553864471&language=EN
# Different total demands [TWh/a]
total_domestic_demand = 926
electricity_demand_volatile = 1515.83*0.3
electricity_demand_const = 1515.83*0.7
total_ghd_demand = 420.5
industry_demand_volatile = 1110.88*0.3
industry_demand_const = 1110.88*0.7
exports_and_other = 988  # from which country?
balance_delta = 163
exports_and_other += balance_delta

# Imports [TWh/a]
russian_import = 1752
non_russian_pipeline_import_and_domestic_production = 3046-2.4*365


## Time and dates
# [h/a]
periods_per_year = 8760

# Start date of the observation period
start_date = "2022-01-01"
number_periods = periods_per_year*1.5

# derive time index for the observation periods
time_index = pd.date_range(start_date, periods=number_periods, freq="H")

# last stop import
time_index_import_normal = pd.date_range(
            start='2022-01-01 00:00:00', end='2022-04-16 00:00:00', freq="H")

time_index_import_reduced = pd.date_range(start='2022-04-16 01:00:00', end='2023-07-02 11:00:00', freq="H")

# derive time for the reduced demand
time_index_demand_reduced = pd.date_range(
    start='2022-03-16 01:00:00', end='2023-07-02 11:00:00', freq="H")

# time for increased lng
time_index_lng_increased = pd.date_range(
    start='2022-05-01 01:00:00', end='2023-07-02 11:00:00', freq="H")


# Storage
# Allow negative State of charge
use_soc_slack = False

# Maximum storage capacity [TWh]
storCap = 1100

# Read daily state of charge data for the beginning of the year (source: GIE)
df_storage = pd.read_excel("Input/Optimization/storage_data_5a.xlsx", index_col=0)
year = 2022
bool_year = [str(year) in str(x) for x in df_storage.gasDayStartedOn]
df_storage = df_storage.loc[bool_year, :]
df_storage.sort_values("gasDayStartedOn", ignore_index=True, inplace=True)

# Fix the state of charge values from January-March; otherwise soc_max = capacity_max [TWh]
soc_max_day = df_storage.gasInStorage

# Convert daily state of charge to hourly state of charge (hourly values=daily values/24) [TWh]
soc_max_hour = []
for value in soc_max_day:
    hour_val = [value]
    soc_max_hour = soc_max_hour + 24 * hour_val



def run_scenario(russ_share=0, lng_val=2.64, demand_reduct=True):
    scenario(lng_val, russ_share, demand_reduct, total_domestic_demand, electricity_demand_const,
            electricity_demand_volatile, industry_demand_const, industry_demand_volatile, total_ghd_demand)

def scenario(lng_val: float, russ_share: float, demand_reduct: bool, total_domestic_demand: float,
             electricity_demand_const: float, electricity_demand_volatile: float, industry_demand_const: float,
             industry_demand_volatile: float, total_ghd_demand: float):
    """Solves a MILP storage model given imports,exports, demands, and production.

    Parameters
    ----------
    lng_val : float
        increased daily LNG flow [TWh/d]
    russ_share : float
        share of Russian natural gas [0 - 1]
    demand_reduct : bool
        indicator whether you want to consider demand reduction or not
    total_domestic_demand : float
        total natural gas demand for domestic purposes [TWh/a]
    electricity_demand_const : float
        base (non-volatile) demand of natural gas for electricity production [TWh/a]
    electricity_demand_volatile : float
        volatile demand of natural gas for electricity production [TWh/a]
    industry_demand_const : float
        base (non-volatile) demand of natural gas for the industry sector [TWh/a]
    industry_demand_volatile : float
        volatile demand of natural gas for the industry sector [TWh/a]
    total_ghd_demand : float
        total demand for the cts sector
    """

    def red_func(demand, red):
        """returns the reduced demand"""
        return demand * (1-red)

    # read timeseries for volatility modeling
    ts = (pd.read_csv("Input/Optimization/ts_normalized.csv")["Private Haushalte"]).values

    # split and recombine to extend to 1.5 years timeframe
    h1, h2 = np.split(ts, [4380])
    new_ts = np.concatenate((ts, h1))

    # setup initial demand timeseries
    domDem = pd.Series(new_ts*total_domestic_demand, index=time_index)

    elecDem_vol = pd.Series(new_ts*electricity_demand_volatile, index=time_index)

    elecDem_const = pd.Series(electricity_demand_const/periods_per_year, index=time_index)

    ghdDem = pd.Series(total_ghd_demand/periods_per_year, index=time_index)

    exp_n_oth = pd.Series(exports_and_other/8760, index=time_index)

    indDem_vol = pd.Series(new_ts*industry_demand_volatile,
                           index=time_index)

    indDem_const = pd.Series(industry_demand_const/periods_per_year, index=time_index)

    # if demand reduction is 'True' reduce individual sector timeseries from
    # the begining of 'time_index_demand_reduced'
    if demand_reduct is True:

        domDem_reduced = pd.Series(new_ts * red_func(total_domestic_demand, red_dom_dem), index=time_index)

        domDem[time_index_demand_reduced] = domDem_reduced[time_index_demand_reduced]

        elecDem_vol_reduced = pd.Series(new_ts*red_func(electricity_demand_volatile, red_elec_dem), index=time_index)

        elecDem_vol[time_index_demand_reduced] = elecDem_vol_reduced[time_index_demand_reduced]

        elecDem_const_reduced = pd.Series(
            red_func(electricity_demand_const, red_elec_dem)/periods_per_year, index=time_index)

        elecDem_const[time_index_demand_reduced] = elecDem_const_reduced[time_index_demand_reduced]

        ghdDem_reduced = pd.Series(red_func(total_ghd_demand, red_ghd_dem)/periods_per_year, index=time_index)
        ghdDem[time_index_demand_reduced] = ghdDem_reduced[time_index_demand_reduced]

        indDem_vol_reduced = pd.Series(new_ts*red_func(industry_demand_volatile, red_ind_dem), index=time_index)

        indDem_vol[time_index_demand_reduced] = indDem_vol_reduced[time_index_demand_reduced]

        indDem_const_reduced = pd.Series(
            red_func(industry_demand_const, red_ind_dem)/periods_per_year, index=time_index)

        indDem_const[time_index_demand_reduced] = indDem_const_reduced[time_index_demand_reduced].values

    # combine volatile and constant parts of the volatile sectors
    elecDem = elecDem_vol + elecDem_const
    indDem = indDem_vol + indDem_const

    # setup initial pipeline supply (before embargo)
    pipe_normal = pd.Series(
        (russian_import+non_russian_pipeline_import_and_domestic_production)/periods_per_year, index=time_index_import_normal)
    pipe_reduced = pd.Series(
        (russ_share*russian_import + non_russian_pipeline_import_and_domestic_production) / periods_per_year,
        index=time_index_import_reduced)
    pipeImp = pd.concat([pipe_normal, pipe_reduced])

    # setup LNG timeseries
    lngImp = pd.Series((lng_val)/24, index=time_index)
    lngImp[time_index_lng_increased] = pd.Series((lng_val+base_lng_import) / 24,
                              index=time_index_lng_increased)

    # create a PYOMO optimzation model
    pyM = pyomo.ConcreteModel()

    # define timesteps
    timeSteps = np.arange(len(domDem)+1)

    def initTimeSet(pyM):
        return (t for t in timeSteps)

    pyM.TimeSet = pyomo.Set(dimen=1, initialize=initTimeSet)

    # state of charge and state of charge slack introduction
    pyM.Soc = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.Soc_slack = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)

    # define flow variables
    pyM.expAndOtherServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.domDemServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.elecDemServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.indDemServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.ghdDemServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.lngServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.pipeServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)

    pyM.NegOffset = pyomo.Var(domain=pyomo.Binary)

    # indicator variables indicating if demand is left unserved
    pyM.expAndOtherIsUnserved = pyomo.Var(domain=pyomo.Binary)
    pyM.domDemIsUnserved = pyomo.Var(domain=pyomo.Binary)
    pyM.elecDemIsUnserved = pyomo.Var(domain=pyomo.Binary)
    pyM.indDemIsUnserved = pyomo.Var(domain=pyomo.Binary)
    pyM.ghdDemIsUnserved = pyomo.Var(domain=pyomo.Binary)

    print(80*"=")
    print("Variables created.")
    print(80*"=")

    # actual hourly LNG flow must be less than the maximum given
    def Constr_lng_ub_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.lngServed[t] <= lngImp.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_lng_ub = pyomo.Constraint(pyM.TimeSet, rule=Constr_lng_ub_rule)

    # actual hourly natural gas pipeline flow must be less than the maximum given
    def Constr_pipe_ub_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.pipeServed[t] <= pipeImp.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_pipe_ub = pyomo.Constraint(pyM.TimeSet, rule=Constr_pipe_ub_rule)

    print(80*"=")
    print("pipe and lng constraint created.")
    print(80*"=")

    # define the objective function (to be minimized) penalizes unserved demands discounted
    # by factor to inscentivize a late occurance
    def Objective_rule(pyM):
        return (- 0.5/len(domDem) * sum(pyM.Soc[t] for t in pyM.TimeSet)/storCap
                + 0*pyM.NegOffset
                + 1*sum(fac**t * pyM.Soc_slack[t] for t in timeSteps[:-1])
                + 1.0*(0*pyM.expAndOtherIsUnserved + sum(fac**t * (exp_n_oth.iloc[t]-pyM.expAndOtherServed[t]) for t in timeSteps[:-1]))
                + 2.5*(0*pyM.domDemIsUnserved + sum(fac**t * (domDem.iloc[t]-pyM.domDemServed[t]) for t in timeSteps[:-1]))
                + 2.5*(0*pyM.ghdDemIsUnserved + sum(fac**t * (ghdDem.iloc[t]-pyM.ghdDemServed[t]) for t in timeSteps[:-1]))
                + 2*(0*pyM.elecDemIsUnserved + sum(fac**t * (elecDem.iloc[t]-pyM.elecDemServed[t]) for t in timeSteps[:-1]))
                + 1.5*(0*pyM.indDemIsUnserved + sum(fac**t * (indDem.iloc[t]-pyM.indDemServed[t]) for t in timeSteps[:-1])))

    pyM.OBJ = pyomo.Objective(rule=Objective_rule, sense=1)

    print(80*"=")
    print("Objective created.")
    print(80*"=")

    # state of charge balance
    def Constr_Soc_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.Soc[t+1]-pyM.Soc_slack[t+1] == pyM.Soc[t]-pyM.domDemServed[t] - pyM.elecDemServed[t]-pyM.indDemServed[t] - pyM.ghdDemServed[t]+pyM.pipeServed[t]+pyM.lngServed[t] - pyM.expAndOtherServed[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Soc = pyomo.Constraint(pyM.TimeSet, rule=Constr_Soc_rule)

    print(80*"=")
    print("SoC constraint created.")
    print(80*"=")

    # maximum storage capacity
    def Constr_Cap_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.Soc[t] <= storCap
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Cap = pyomo.Constraint(pyM.TimeSet, rule=Constr_Cap_rule)

    print(80*"=")
    print("max storage capacity constraint created.")
    print(80*"=")

    # served/unserved demands must not exceed their limits
    def Constr_ExpAndOtherServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.expAndOtherServed[t] <= exp_n_oth.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_ExpAndOtherServed = pyomo.Constraint(pyM.TimeSet, rule=Constr_ExpAndOtherServed_rule)

    def Constr_ExpAndOtherIsUnserved_rule(pyM):
        return sum(exp_n_oth.iloc[t] - pyM.expAndOtherServed[t] for t in timeSteps[: -1]) <= sum(exp_n_oth.iloc[t]
                                                                                                 for t in timeSteps[: -1]) * pyM.expAndOtherIsUnserved

    pyM.Constr_ExpAndOtherIsUnserved = pyomo.Constraint(rule=Constr_ExpAndOtherIsUnserved_rule)

    def Constr_DomDemIsUnserved_rule(pyM):
        return sum(domDem.iloc[t] - pyM.domDemServed[t] for t in timeSteps[: -1]) <= sum(domDem.iloc[t]
                                                                                         for t in timeSteps[: -1]) * pyM.domDemIsUnserved

    pyM.Constr_DomDemIsUnserved = pyomo.Constraint(rule=Constr_DomDemIsUnserved_rule)

    def Constr_DomDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.domDemServed[t] <= domDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_DomDemServed = pyomo.Constraint(pyM.TimeSet, rule=Constr_DomDemServed_rule)

    def Constr_GhdDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.ghdDemServed[t] <= ghdDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_GhdDemServed = pyomo.Constraint(pyM.TimeSet, rule=Constr_GhdDemServed_rule)

    def Constr_GhdDemIsUnserved_rule(pyM):
        return sum(ghdDem.iloc[t] - pyM.ghdDemServed[t] for t in timeSteps[: -1]) <= sum(ghdDem.iloc[t]
                                                                                         for t in timeSteps[: -1]) * pyM.ghdDemIsUnserved

    pyM.Constr_GhdDemIsUnserved = pyomo.Constraint(rule=Constr_GhdDemIsUnserved_rule)

    def Constr_ElecDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.elecDemServed[t] <= elecDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_ElecDemServed = pyomo.Constraint(pyM.TimeSet, rule=Constr_ElecDemServed_rule)

    def Constr_ElecDemIsUnserved_rule(pyM):
        return sum(elecDem.iloc[t] - pyM.elecDemServed[t] for t in timeSteps[: -1]) <= sum(elecDem.iloc[t]
                                                                                           for t in timeSteps[: -1]) * pyM.elecDemIsUnserved

    pyM.Constr_ElecDemIsUnserved = pyomo.Constraint(rule=Constr_ElecDemIsUnserved_rule)

    def Constr_IndDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.indDemServed[t] <= indDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_IndDemServed = pyomo.Constraint(pyM.TimeSet, rule=Constr_IndDemServed_rule)

    def Constr_IndDemIsUnserved_rule(pyM):
        return sum(indDem.iloc[t] - pyM.indDemServed[t] for t in timeSteps[: -1]) <= sum(indDem.iloc[t]
                                                                                         for t in timeSteps[: -1]) * pyM.indDemIsUnserved

    pyM.Constr_IndDemIsUnserved = pyomo.Constraint(rule=Constr_IndDemIsUnserved_rule)

    # fix the initial (past) state of charge to historic value (slightly relaxed with buffer +/-10 TWh)
    def Constr_soc_start_ub_rule(pyM, t):
        if t < len(soc_max_hour):
            return pyM.Soc[t] <= soc_max_hour[t] + 10
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Soc_start_ub = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_soc_start_ub_rule)

    def Constr_soc_start_lb_rule(pyM, t):
        if t < len(soc_max_hour):
            return pyM.Soc[t] >= soc_max_hour[t]-10
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Soc_start_lb = pyomo.Constraint(pyM.TimeSet, rule=Constr_soc_start_lb_rule)

    # fix state of charge slack to zero if not wanted
    if use_soc_slack is False:
        for i in timeSteps:
            pyM.Soc_slack[i].fix(0)

    print(80*"=")
    print("Starting solve...")
    print(80*"=")

    # set solver details
    solver = 'glpk'  # gurobi
    optimizer = opt.SolverFactory(solver)
    solver_info = optimizer.solve(pyM, tee=True)

    print(solver_info['Problem'][0])

    print(80*"=")
    print("Retrieving solution...")
    print(80*"=")

    # retrieve solution values and collect in a pandas dataframe
    socList = [pyM.Soc[t].value for t in timeSteps]
    socSlackList = [pyM.Soc_slack[t].value for t in timeSteps]
    domDemList = [pyM.domDemServed[t].value for t in timeSteps]
    elecDemList = [pyM.elecDemServed[t].value for t in timeSteps]
    indDemList = [pyM.indDemServed[t].value for t in timeSteps]
    ghdDemList = [pyM.ghdDemServed[t].value for t in timeSteps]
    pipeServedList = [pyM.pipeServed[t].value for t in timeSteps]
    lngServedList = [pyM.lngServed[t].value for t in timeSteps]
    expAndOtherServedList = [pyM.expAndOtherServed[t].value for t in timeSteps]

    print("building DataFrame...")
    df = pd.DataFrame(
        {"pipeImp": pipeImp.values, "lngImp": lngImp.values, "lngServed": lngServedList[: -1],
         "pipeServed": pipeServedList[: -1],
         "soc": socList[: -1],
         "soc_slack": socSlackList[: -1],
         "dom_served": domDemList[: -1],
         "elec_served": elecDemList[: -1],
         "ind_served": indDemList[: -1],
         "ghd_served": ghdDemList[: -1],
         "exp_n_oth_served": expAndOtherServedList[: -1]})
    df["time"] = pipeImp.index
    print("initial df created.")
    df = df.assign(dom_Dem=domDem.values)
    df = df.assign(elec_Dem=elecDem.values)
    df = df.assign(ind_Dem=indDem.values)
    df = df.assign(ghd_Dem=ghdDem.values)
    df = df.assign(exp_n_oth=exp_n_oth.values)
    print("columns assigned, adding scalar values...")
    df["russ_share"] = russ_share
    df["lng_val"] = lng_val
    df["storCap"] = storCap
    df["stor_Start"] = storCap
    df["timeSteps"] = timeSteps[:-1]

    df["neg_offset"] = pyM.NegOffset.value
    df["dom_unserved"] = pyM.domDemIsUnserved.value
    df["elec_unserved"] = pyM.elecDemIsUnserved.value
    df["ind_unserved"] = pyM.indDemIsUnserved.value
    df["ghd_unserved"] = pyM.ghdDemIsUnserved.value
    df["exp_n_oth_unserved"] = pyM.expAndOtherIsUnserved.value

    print("positive side of balance: ", df.soc_slack.sum() +
          df.pipeServed.sum()+df.lngServed.sum())
    print("storage_delta: ", df.soc.iloc[0]-df.soc.iloc[-1])
    print("negative side of balance: ", df.dom_served.sum(
    )+df.elec_served.sum()+df.ind_served.sum()+df.ghd_served.sum() + df.exp_n_oth_served.sum())

    print("soc slack sum: ", df.soc_slack.sum())

    df['balance'] = df.soc_slack.sum()+df.pipeServed.sum()+df.lngServed.sum()+df.soc.iloc[0]-df.soc.iloc[-1] - \
        (df.dom_served.sum()+df.elec_served.sum()+df.ind_served.sum()+df.ghd_served.sum() + df.exp_n_oth_served.sum())
    print(df['balance'])
    print("saving...")

    df.to_excel(f'Input/Optimization/results_aGasFlowScen{int(russ_share*100)}_{int(lng_val*10)}_{demand_reduct}_{use_soc_slack}.xlsx')
    print("Done!")

    # fig, ax = plt.subplots(figsize=(20, 4))
    # ax.stackplot(timeSteps, socList)
    # ax.set_xticks(np.linspace(0, number_periods, 13+6))
    # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt',
    #                     'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'], rotation=90)
    # ax.set_xlim(0, number_periods)
    # ax.set_ylim(0, 1200)
    # ax.set_xlabel('Time of the Year', fontsize=20)
    # ax.set_ylabel('Storage Level [TWh]', fontsize=20)
    # ax.set_title(
    #     f'Natural Gas Storage Level ({russ_share*100} % Share of Russian gas, LNG val. {lng_val},demand reduction: {demand_reduct})',
    #     fontsize=24, pad=10)
    # # plt.show()

    # plt.savefig(f'results_aGasSocScen{int(russ_share*100)}_{int(lng_val*10)}_{demand_reduct}_{use_soc_slack}.png')
    # fig, ax = plt.subplots(figsize=(20, 4))
    # ax.stackplot(
    #     timeSteps, [i for i in pipeServedList[:-1]+[0]],
    #     [i for i in lngServedList[:-1]+[0]],
    #     labels=['pipeImp', 'lngImp'],
    #     zorder=1)
    # ax.stackplot(
    #     timeSteps, domDemList[: -1] + [0],
    #     elecDemList[: -1] + [0],
    #     indDemList[: -1] + [0],
    #     ghdDemList[: -1] + [0],
    #     expAndOtherServedList[: -1] + [0],
    #     labels=['domestic sector', 'electricity sector', 'industry sector', 'cts sector', 'export and other'])
    # ax.set_xticks(np.linspace(0, 8760*1.5, 13+6))
    # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt',
    #                     'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'], rotation=90)
    # ax.set_xlim(0, 8760*1.5)
    # ax.set_ylim(0, 1.1)
    # ax.set_xlabel('Time of the Year', fontsize=20)
    # ax.set_ylabel('served demands [TWh/h]', fontsize=20)
    # plt.legend()
    # # plt.show()
    # plt.savefig(f'results_aGasFlowScen{int(russ_share*100)}_{int(lng_val*10)}_{demand_reduct}_{use_soc_slack}.png')

    # for i in [pyM.domDemIsUnserved, pyM.elecDemIsUnserved, pyM.ghdDemIsUnserved, pyM.indDemIsUnserved]:
    #     print(i.value)
    # plt.close()


# # %%
# # loop over all scenario variations
# for russ_share in russian_gas_share:
#     for lng_val in lng_values:
#         for demand_reduct in demand_reduction:
#             scenario(lng_val, russ_share, demand_reduct, total_domestic_demand, electricity_demand_const,
#                      electricity_demand_volatile, industry_demand_const, industry_demand_volatile, total_ghd_demand)

# # %%
