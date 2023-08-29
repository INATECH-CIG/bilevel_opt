"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

# %%
# Imports
from linopy import Model
import pandas as pd


# %%
def uc_problem(gens_df, demand_df, k_values_df, u_fixed=None):
    model = Model()

    time = pd.Index(range(len(demand_df)), name="time")
    gens = pd.Index(range(len(gens_df)), name="generators")

    # primary problem variables
    g = model.add_variables(
        lower=0, coords=[gens, time], name="generation"
    )  # generation in MW
    d = model.add_variables(lower=0, coords=[time], name="demand")  # demand in MW

    c_up = model.add_variables(
        lower=0, coords=[gens, time], name="start-up"
    )  # start-up costs in EUR/h
    c_down = model.add_variables(
        lower=0, coords=[gens, time], name="shut-down"
    )  # shut-down costs in EUR/h

    if u_fixed is None:
        u = model.add_variables(
            binary=True, coords=[gens, time], name="on/off"
        )  # on/off status of generators
    else:
        u = model.add_variables(coords=[gens, time], name="on/off")
        model.add_constraints(u == u_fixed, coords=[gens, time], name="u_fixed")

    # primary problem objective
    def primary_objective_rule(model):
        expr = sum(
            (k_values_df[gen] * gens_df.at[gen, "mc"]).to_list() * g.loc[gen, :]
            + gens_df.at[gen, "f"] * u.loc[gen, :]
            + c_up.loc[gen, :]
            + c_down.loc[gen, :]
            for gen in gens
        )
        expr -= demand_df.loc[:, "price"].values * d
        return expr.sum()

    objective = model.add_objective(expr=primary_objective_rule(model))

    # energy balance constraint
    def balance_rule(model, t):
        return d[t] - sum(g[i, t] for i in gens) == 0

    balance_constr = model.add_constraints(balance_rule, coords=[time], name="balance")

    # max generation constraint
    def g_max_rule(model, i, t):
        return g[i, t] - gens_df.at[i, "g_max"] * u[i, t] <= 0

    g_max_constr = model.add_constraints(g_max_rule, coords=[gens, time], name="g_max")

    # min generation constraint
    def g_min_rule(model, i, t):
        return g[i, t] - gens_df.at[i, "g_min"] * u[i, t] >= 0

    g_min_constr = model.add_constraints(g_min_rule, coords=[gens, time], name="g_min")

    # max demand constraint
    demand_constr = model.add_constraints(
        d <= demand_df.loc[:, "volume"].values, name="d_max"
    )

    # ramp up constraint
    def ramp_up_rule(model, i, t):
        if t == 0:
            return g[i, t] <= gens_df.at[i, "g_0"] + gens_df.at[i, "r_up"]
        else:
            return g[i, t] - g[i, t - 1] <= gens_df.at[i, "r_up"]

    ramp_up_constr = model.add_constraints(
        ramp_up_rule, coords=[gens, time], name="ramp_up"
    )

    # ramp down constraint
    def ramp_down_rule(model, i, t):
        if t == 0:
            return -g[i, t] <= max(
                gens_df.at[i, "g_0"] - gens_df.at[i, "r_down"], gens_df.at[i, "g_min"]
            )
        else:
            return g[i, t - 1] - g[i, t] <= gens_df.at[i, "r_down"]

    ramp_down_constr = model.add_constraints(
        ramp_down_rule, coords=[gens, time], name="ramp_down"
    )

    # start up cost constraint
    def start_up_rule(model, i, t):
        if t == 0:
            return (
                c_up[i, t] - u[i, t] * gens_df.at[i, "k_up"]
                >= -gens_df.at[i, "u_0"] * gens_df.at[i, "k_up"]
            )
        else:
            return c_up[i, t] - (u[i, t] - u[i, t - 1]) * gens_df.at[i, "k_up"] >= 0

    start_up_constr = model.add_constraints(
        start_up_rule, coords=[gens, time], name="start_up"
    )

    # shut down cost constraint
    def shut_down_rule(model, i, t):
        if t == 0:
            return (
                c_down[i, t] + u[i, t] * gens_df.at[i, "k_down"]
                >= gens_df.at[i, "u_0"] * gens_df.at[i, "k_down"]
            )
        else:
            return c_down[i, t] - (u[i, t - 1] - u[i, t]) * gens_df.at[i, "k_down"] >= 0

    shut_down_constr = model.add_constraints(
        shut_down_rule, coords=[gens, time], name="shut_down"
    )

    model.solve(solver_name="gurobi", LogToConsole=False)

    if u_fixed is not None:
        prices = -balance_constr.dual
        generation = pd.DataFrame(
            columns=[f"gen_{gen}" for gen in gens], index=time, data=g.solution.T
        ).round(3)
        demand = pd.DataFrame(columns=["demand"], index=time, data=d.solution).round(3)
        mcp = pd.DataFrame(columns=["price"], index=time, data=prices.values).round(3)
        main_df = pd.concat([generation, demand, mcp], axis=1)
        supp_df = model.solution.to_dataframe()

        return main_df, supp_df

    else:
        return u.solution


def solve_and_get_prices(gens_df, demand_df, k_values_df):
    # solve once to get status u
    u_init = uc_problem(
        gens_df=gens_df,
        demand_df=demand_df,
        k_values_df=k_values_df,
    )
    # solve again with status u
    main_df, supp_df = uc_problem(
        gens_df=gens_df,
        demand_df=demand_df,
        k_values_df=k_values_df,
        u_fixed=u_init.values,
    )

    return main_df, supp_df


# %%
if __name__ == "__main__":
    start = pd.to_datetime("2019-03-01 00:00")
    end = pd.to_datetime("2019-03-02 00:00")

    # generators
    gens_df = pd.read_csv("inputs/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv("inputs/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]

    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.read_csv("outputs/k_values_df.csv", index_col=0)
    k_values_df.columns = k_values_df.columns.astype(int)

    # %%
    main_df, supp_df = solve_and_get_prices(gens_df, demand_df, k_values_df)

# %%
