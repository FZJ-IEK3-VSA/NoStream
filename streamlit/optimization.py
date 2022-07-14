# %%
import pandas as pd
import numpy as np
import pyomo.environ as pyomo
import pyomo.opt as opt
import streamlit as st
import utils as ut
import datetime
import os
import gie_api


@st.experimental_memo(show_spinner=False)
def run_scenario(
    total_ng_import=4190,
    total_pl_import_russia=1752,
    total_ng_production=608,
    total_lng_import=914,
    total_lng_import_russia=160,
    total_domestic_demand=926,
    total_ghd_demand=420.5,
    total_electricity_demand=1515.83,
    total_industry_demand=1110.88,
    total_exports_and_other=988,
    red_dom_dem=0.13,
    red_elec_dem=0.20,
    red_ghd_dem=0.08,
    red_ind_dem=0.08,
    red_exp_dem=0.0,
    import_stop_date=datetime.datetime(2022, 4, 16, 0, 0),
    demand_reduction_date=datetime.datetime(2022, 3, 16, 0, 0),
    lng_increase_date=datetime.datetime(2022, 5, 1, 0, 0),
    add_lng_import=965,
    add_pl_import=0,
    reduction_import_russia=1,
    consider_gas_reserve=False,
    reserve_dates=None,
    reserve_soc_val=None,
    storage_capacity=1100,
    spacial_scope="EU",
):
    """Solves a MILP storage model given imports,exports, demands, and production.

    Parameters
    ----------
    add_lng_import : float
        increased daily LNG flow [TWh/d]
    reduction_import_russia : float
        reduction rate of Russian natural gas/LNG imports [0 - 1]
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
    total_exports_and_other : float
        total demand for the cts sectorexports and other demands
    """
    ###############################################################################
    ############            Preprocessing/Input generation             ############
    ###############################################################################

    # Storage
    print(80 * "=")
    print("Obtaining storage levels...")
    print(80 * "=")

    today = datetime.datetime.today()
    soc_max_hour = gie_api.get_storage_capacity(
        spacial_scope, today
    )  # get_storage_capacity()

    if red_dom_dem + red_elec_dem + red_ghd_dem + red_ind_dem + red_exp_dem > 0:
        demand_reduct = True
    else:
        demand_reduct = False

    # Start date of the observation period
    start_date = "2022-01-01"
    datetime_start = datetime.datetime(2022, 1, 1, 0, 0)
    periods_per_year = 8760  # [h/a]
    number_periods = periods_per_year * 1.5

    # Time index defualt
    time_index = pd.date_range(start_date, periods=number_periods, freq="H")

    # Time index till today, no optimization (fix)
    end_date = datetime.datetime.today()
    timedelta = end_date - datetime_start
    number_periods_fix = (timedelta.days + 1) * 24
    time_index_fix = pd.date_range(start_date, periods=number_periods_fix, freq="H")

    # Time index import stop
    time_index_pl_red = pd.date_range(
        start=import_stop_date + datetime.timedelta(hours=1),
        end="2023-07-02 11:00:00",
        freq="H",
    )
    time_index_lng_red = time_index_pl_red.copy()

    # Time index slack stop
    # TODO dynamically adapt to storage data
    time_index_slack = pd.date_range(
        start_date, periods=len(soc_max_hour) - 3, freq="H"
    )

    # Time index reduced demand
    time_index_demand_red = pd.date_range(
        start=demand_reduction_date + datetime.timedelta(hours=1),
        end="2023-07-02 11:00:00",
        freq="H",
    )

    # Time index uncurtailed demand
    time_index_uncurtailed_demand = pd.date_range(
        start="2022-01-01 00:00:00", end=datetime.datetime.now(), freq="H",
    )

    # Time index increased lng
    time_index_lng_increased = pd.date_range(
        start=lng_increase_date + datetime.timedelta(hours=1),
        end="2023-07-02 11:00:00",
        freq="H",
    )
    time_index_pl_increased = time_index_lng_increased.copy()

    # Normalized volatile timeseries
    ts_vol = (
        pd.read_csv("static/Optimization/ts_normalized.csv")["Private Haushalte"]
    ).values

    # split and recombine to extend to 1.5 years timeframe
    h1, h2 = np.split(ts_vol, [int(0.5 * periods_per_year)])
    ts_vol = np.concatenate((ts_vol, h1))
    ts_const = np.ones_like(ts_vol) * 1 / periods_per_year

    # Setup initial demand timeseries
    # Energy balance, EUROSTAT 2019 (sankey)
    # https://ec.europa.eu/eurostat/cache/sankey/energy/sankey.html?geos=EU27_2020&year=2019&unit=GWh&fuels=TOTAL&highlight=_2_&nodeDisagg=1111111111111&flowDisagg=true&translateX=15.480270462412136&translateY=135.54626885696325&scale=0.6597539553864471&language=EN
    electricity_demand_volatile = total_electricity_demand * 0.3
    electricity_demand_const = total_electricity_demand * 0.7

    industry_demand_volatile = total_industry_demand * 0.3
    industry_demand_const = total_industry_demand * 0.7

    domDem = pd.Series(ts_vol * total_domestic_demand, index=time_index)
    ghdDem = pd.Series(ts_const * total_ghd_demand, index=time_index)
    exp_n_oth = pd.Series(ts_const * total_exports_and_other, index=time_index)

    elecDem_vol = pd.Series(ts_vol * electricity_demand_volatile, index=time_index)
    elecDem_const = pd.Series(ts_const * electricity_demand_const, index=time_index)
    elecDem = elecDem_vol + elecDem_const

    indDem_vol = pd.Series(ts_vol * industry_demand_volatile, index=time_index)
    indDem_const = pd.Series(ts_const * industry_demand_const, index=time_index)
    indDem = indDem_vol + indDem_const

    # Demand reduction
    def red_func(demand, red):
        """returns the reduced demand"""
        return demand * (1 - red)

    domDem_red = red_func(domDem, red_dom_dem)
    domDem[time_index_demand_red] = domDem_red[time_index_demand_red]

    ghdDem_red = red_func(ghdDem, red_ghd_dem)
    ghdDem[time_index_demand_red] = ghdDem_red[time_index_demand_red]

    exp_n_oth_red = red_func(exp_n_oth, red_exp_dem)
    exp_n_oth[time_index_demand_red] = exp_n_oth_red[time_index_demand_red]

    elecDem_red = red_func(elecDem, red_elec_dem)
    elecDem[time_index_demand_red] = elecDem_red[time_index_demand_red]

    indDem_red = red_func(indDem, red_ind_dem)
    indDem[time_index_demand_red] = indDem_red[time_index_demand_red]

    # Minimum served demand for timesteps in the past
    totalDem_uncurtailed = pd.Series(ts_const * 0, index=time_index)
    totalDem = domDem + ghdDem + exp_n_oth + elecDem + indDem
    totalDem_uncurtailed[time_index_uncurtailed_demand] = totalDem[
        time_index_uncurtailed_demand
    ]

    # Pipeline Supply
    total_pl_import = total_ng_import - total_lng_import

    plImp = pd.Series(ts_const * (total_pl_import), index=time_index,)

    plImp_red = pd.Series(
        ts_const * (total_pl_import - reduction_import_russia * total_pl_import_russia),
        index=time_index,
    )

    plImp_increased = pd.Series(
        ts_const
        * (
            total_pl_import
            - reduction_import_russia * total_pl_import_russia
            + add_pl_import
        ),
        index=time_index,
    )

    plImp[time_index_pl_red] = plImp_red[time_index_pl_red]
    plImp[time_index_pl_increased] = plImp_increased[time_index_pl_increased]

    # LNG supply
    lngImp = pd.Series(ts_const * total_lng_import, index=time_index)

    lngImp_red = pd.Series(
        ts_const
        * (total_lng_import - reduction_import_russia * total_lng_import_russia),
        index=time_index,
    )

    lngImp_increased = pd.Series(
        ts_const
        * (
            total_lng_import
            - reduction_import_russia * total_lng_import_russia
            + add_lng_import
        ),
        index=time_index,
    )

    lngImp[time_index_lng_red] = lngImp_red[time_index_lng_red]
    lngImp[time_index_lng_increased] = lngImp_increased[time_index_lng_increased]

    # Domestiv production
    domProd = pd.Series(ts_const * total_ng_production, index=time_index)

    # Slack supply
    max_slack = float("inf")
    slackImp = pd.Series(ts_const * 0, index=time_index)
    slackImp_high = pd.Series(ts_const * max_slack, index=time_index)  # float("inf")

    slackImp[time_index_slack] = slackImp_high[time_index_slack]

    slackImp_neg = pd.Series(ts_const * 0, index=time_index)
    slackImp_neg_high = pd.Series(
        ts_const * -max_slack, index=time_index
    )  # float("inf")

    slackImp_neg[time_index_slack] = slackImp_neg_high[time_index_slack]

    ###############################################################################
    ############                      Optimization                     ############
    ###############################################################################

    # create a PYOMO optimzation model
    pyM = pyomo.ConcreteModel()

    # define timesteps
    timeSteps = np.arange(len(domDem) + 1)

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
    pyM.plServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.prodServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.slackServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonNegativeReals)
    pyM.slackNegServed = pyomo.Var(pyM.TimeSet, domain=pyomo.NonPositiveReals)

    print(80 * "=")
    print("Variables created.")
    print(80 * "=")

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
            return pyM.plServed[t] <= plImp.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_pipe_ub = pyomo.Constraint(pyM.TimeSet, rule=Constr_pipe_ub_rule)

    # actual hourly natural gas production flow must be less than the maximum given
    def Constr_prod_ub_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.prodServed[t] <= domProd.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_prod_ub = pyomo.Constraint(pyM.TimeSet, rule=Constr_prod_ub_rule)

    # actual hourly slack flow must be less than the maximum given
    def Constr_slack_ub_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.slackServed[t] <= slackImp.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_slack_ub = pyomo.Constraint(pyM.TimeSet, rule=Constr_slack_ub_rule)

    # slack neg
    def Constr_slack_neg_lb_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.slackNegServed[t] >= slackImp_neg.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_slack_neg_lb = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_slack_neg_lb_rule
    )

    print(80 * "=")
    print("pipe and lng constraint created.")
    print(80 * "=")

    # define the objective function (to be minimized) penalizes unserved demands discounted
    # by factor to inscentivize a late occurance

    # Discounting factor [-/h]
    fac = (1 / 1.06) ** (1 / 8760)

    def Objective_rule(pyM):
        return (
            -0.5 / len(domDem) * sum(pyM.Soc[t] for t in pyM.TimeSet) / storage_capacity
            + 1 * sum(fac ** t * pyM.Soc_slack[t] for t in timeSteps[:-1])
            + 1 * sum(fac ** t * pyM.slackServed[t] for t in timeSteps[:-1])
            - 1 * sum(fac ** t * pyM.slackNegServed[t] for t in timeSteps[:-1])
            + 3.0
            * sum(
                fac ** t * (exp_n_oth.iloc[t] - pyM.expAndOtherServed[t])
                for t in timeSteps[:-1]
            )
            + 2.5
            * sum(
                fac ** t * (domDem.iloc[t] - pyM.domDemServed[t])
                for t in timeSteps[:-1]
            )
            + 2.5
            * sum(
                fac ** t * (ghdDem.iloc[t] - pyM.ghdDemServed[t])
                for t in timeSteps[:-1]
            )
            + 2
            * sum(
                fac ** t * (elecDem.iloc[t] - pyM.elecDemServed[t])
                for t in timeSteps[:-1]
            )
            + 1.5
            * sum(
                fac ** t * (indDem.iloc[t] - pyM.indDemServed[t])
                for t in timeSteps[:-1]
            )
        )

    pyM.OBJ = pyomo.Objective(rule=Objective_rule, sense=1)

    print(80 * "=")
    print("Objective created.")
    print(80 * "=")

    # state of charge balance
    def Constr_Soc_rule(pyM, t):
        if t < timeSteps[-1]:
            return (
                pyM.Soc[t + 1] - pyM.Soc_slack[t + 1]
                == pyM.Soc[t]
                - pyM.domDemServed[t]
                - pyM.elecDemServed[t]
                - pyM.indDemServed[t]
                - pyM.ghdDemServed[t]
                - pyM.expAndOtherServed[t]
                + pyM.plServed[t]
                + pyM.lngServed[t]
                + pyM.prodServed[t]
                + pyM.slackServed[t]
                + pyM.slackNegServed[t]
            )
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Soc = pyomo.Constraint(pyM.TimeSet, rule=Constr_Soc_rule)

    print(80 * "=")
    print("SoC constraint created.")
    print(80 * "=")

    # maximum storage capacity
    def Constr_Cap_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.Soc[t] <= storage_capacity
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Cap = pyomo.Constraint(pyM.TimeSet, rule=Constr_Cap_rule)

    print(80 * "=")
    print("max storage capacity constraint created.")
    print(80 * "=")

    # Gas reserve
    if consider_gas_reserve:
        # reserve_soc_val = reserve_dates
        reserve_dates = [x - datetime_start for x in reserve_dates]
        reserve_dates = [x.days for x in reserve_dates]
        reserve_dates = [timeSteps[(x) * 24] for x in reserve_dates]
        reserve_soc_dict = dict(zip(reserve_dates, reserve_soc_val))

        def Constr_Reserve_rule(pyM, t):
            if t in reserve_dates:
                return pyM.Soc[t] >= reserve_soc_dict.get(t)
            else:
                return pyomo.Constraint.Skip

        pyM.Constr_Reserve = pyomo.Constraint(pyM.TimeSet, rule=Constr_Reserve_rule)
        pass
        print(80 * "=")
        print("gas reserve constraint created.")
        print(80 * "=")

    # served/unserved demands must not exceed their limits
    def Constr_ExpAndOtherServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.expAndOtherServed[t] <= exp_n_oth.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_ExpAndOtherServed = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_ExpAndOtherServed_rule
    )

    def Constr_DomDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.domDemServed[t] <= domDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_DomDemServed = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_DomDemServed_rule
    )

    def Constr_GhdDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.ghdDemServed[t] <= ghdDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_GhdDemServed = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_GhdDemServed_rule
    )

    def Constr_ElecDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.elecDemServed[t] <= elecDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_ElecDemServed = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_ElecDemServed_rule
    )

    def Constr_IndDemServed_rule(pyM, t):
        if t < timeSteps[-1]:
            return pyM.indDemServed[t] <= indDem.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_IndDemServed = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_IndDemServed_rule
    )

    # no curtailment for timesteps in the past
    def Constr_UncurtailedDemand_rule(pyM, t):
        if t < timeSteps[-1]:
            return (
                pyM.expAndOtherServed[t]
                + pyM.domDemServed[t]
                + pyM.ghdDemServed[t]
                + pyM.elecDemServed[t]
                + pyM.indDemServed[t]
            ) >= totalDem_uncurtailed.iloc[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_UncurtailedDemand = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_UncurtailedDemand_rule
    )

    # slack variable
    def Constr_soc_start_ub_rule(pyM, t):
        if t < len(soc_max_hour):
            return pyM.Soc[t] <= soc_max_hour[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Soc_start_ub = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_soc_start_ub_rule
    )

    def Constr_soc_start_lb_rule(pyM, t):
        if t < len(soc_max_hour):
            return pyM.Soc[t] >= soc_max_hour[t]
        else:
            return pyomo.Constraint.Skip

    pyM.Constr_Soc_start_lb = pyomo.Constraint(
        pyM.TimeSet, rule=Constr_soc_start_lb_rule
    )

    # fix state of charge slack to zero if not wanted
    use_soc_slack = False
    if use_soc_slack is False:
        for i in timeSteps:
            pyM.Soc_slack[i].fix(0)

    print(80 * "=")
    print("Starting solve...")
    print(80 * "=")

    # set solver details
    # Check which solvers are available and choose default solver if no solver is specified explicitely
    # Order of possible solvers in solverList defines the priority of chosen default solver.
    solverList = ["gurobi", "cbc", "glpk"]
    # solver = "cbc"

    for solver_option in solverList:
        if opt.SolverFactory(solver_option).available():
            solver = solver_option
            break

    # if opt.SolverFactory("gurobi").available():
    #     solver = "gurobi"
    # elif opt.SolverFactory("cbc").available():
    #     solver = "cbc"
    # else:
    #     solver = "glpk"

    optimizer = opt.SolverFactory(solver)
    solver_info = optimizer.solve(pyM, tee=True)

    print(solver_info["Problem"][0])

    print(80 * "=")
    print("Retrieving solution...")
    print(80 * "=")

    # retrieve solution values and collect in a pandas dataframe
    slackImpServedList = pd.Series([pyM.slackServed[t].value for t in timeSteps[:-1]])
    plServedList = pd.Series([pyM.plServed[t].value for t in timeSteps[:-1]])
    lngServedList = pd.Series([pyM.lngServed[t].value for t in timeSteps[:-1]])
    prodServedList = pd.Series([pyM.prodServed[t].value for t in timeSteps[:-1]])
    socList = pd.Series([pyM.Soc[t].value for t in timeSteps[:-1]])
    socSlackList = pd.Series([pyM.Soc_slack[t].value for t in timeSteps[:-1]])
    domDemServedList = pd.Series([pyM.domDemServed[t].value for t in timeSteps[:-1]])
    elecDemServedList = pd.Series([pyM.elecDemServed[t].value for t in timeSteps[:-1]])
    indDemServedList = pd.Series([pyM.indDemServed[t].value for t in timeSteps[:-1]])
    ghdDemServedList = pd.Series([pyM.ghdDemServed[t].value for t in timeSteps[:-1]])
    expAndOtherServedList = pd.Series(
        [pyM.expAndOtherServed[t].value for t in timeSteps[:-1]]
    )

    plServedList[:number_periods_fix] = plImp[:number_periods_fix]
    lngServedList[:number_periods_fix] = lngImp[:number_periods_fix]
    prodServedList[:number_periods_fix] = domProd[:number_periods_fix]

    print("building DataFrame...")
    df = pd.DataFrame()
    df = df.assign(
        time=plImp.index,
        plImp=plImp.values,
        plImp_served=plServedList,
        lngImp=lngImp.values,
        lngImp_served=lngServedList,
        domProd=domProd.values,
        domProd_served=prodServedList,
        domDem=domDem.values,
        domDem_served=domDemServedList,
        elecDem=elecDem.values,
        elecDem_served=elecDemServedList,
        indDem=indDem.values,
        indDem_served=indDemServedList,
        ghdDem=ghdDem.values,
        ghdDem_served=ghdDemServedList,
        exp_n_oth=exp_n_oth.values,
        exp_n_oth_served=expAndOtherServedList,
        soc=socList.values,
        soc_slack=socSlackList,
        slackImp=slackImp.values,
        slackImp_served=slackImpServedList,
    )
    df.fillna(0, inplace=True)

    print("saving...")
    # df.to_csv(f"default_results.csv")

    value_col = "value"
    input_data = pd.DataFrame(columns=["value"])
    input_data.loc["total_pl_import", value_col] = total_pl_import
    input_data.loc["total_ng_production", value_col] = total_ng_production
    input_data.loc["total_pl_import_russia", value_col] = total_pl_import_russia
    input_data.loc["total_domestic_demand", value_col] = total_domestic_demand
    input_data.loc["total_ghd_demand", value_col] = total_ghd_demand
    input_data.loc["total_electricity_demand", value_col] = total_electricity_demand
    input_data.loc["total_industry_demand", value_col] = total_industry_demand
    input_data.loc["total_exports_and_other", value_col] = total_exports_and_other
    input_data.loc["red_dom_dem", value_col] = red_dom_dem
    input_data.loc["red_elec_dem", value_col] = red_elec_dem
    input_data.loc["red_ghd_dem", value_col] = red_ghd_dem
    input_data.loc["red_ind_dem", value_col] = red_ind_dem
    input_data.loc["red_exp_dem", value_col] = red_exp_dem
    input_data.loc["import_stop_date", value_col] = import_stop_date
    input_data.loc["demand_reduction_date", value_col] = demand_reduction_date
    input_data.loc["lng_increase_date", value_col] = lng_increase_date
    input_data.loc["total_lng_import", value_col] = total_lng_import
    input_data.loc["add_lng_import", value_col] = add_lng_import
    input_data.loc["add_pl_import", value_col] = add_pl_import
    input_data.loc["reduction_import_russia", value_col] = reduction_import_russia
    input_data.loc["storage_capacity", value_col] = storage_capacity
    print("saving...")
    # input_data.to_csv(f"default_inputs.csv")

    print("Done!")
    return df, input_data


# %%
if __name__ == "__main__":
    # Sensitivity analysis

    # reduction of russion gas/LNG [-]
    russian_gas_reduction = [1]  # [0, 1]

    # Average European LNG import [TWh/d]
    lng_add_capacities = [965]  # [0.0, 965]  # 90% load

    # loop over all scenario variations
    for russ_red in russian_gas_reduction:
        for add_lng_import in lng_add_capacities:
            df, input_data = run_scenario(
                reduction_import_russia=russ_red, add_lng_import=add_lng_import
            )
    print("Done")
