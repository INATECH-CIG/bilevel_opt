"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

# %%
# Imports
from linopy import Model
import pandas as pd


# %%
def find_optimal_k(gens_df, demand_df, k_max, opt_gen, big_w=10000):
    model = Model()

    time = pd.Index(range(len(demand_df)), name="time")
    gens = pd.Index(gens_df.index, name="generators")

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
    k = model.add_variables(
        lower=1, upper=k_max, name="strategy"
    )  # strategic bidding parameter
    lambda_ = model.add_variables(
        lower=-500, upper=3000, coords=[time], name="market price"
    )  # market price in EUR/MWh
    u = model.add_variables(
        binary=True, coords=[gens, time], name="on/off"
    )  # on/off status of generators

    # secondary problem variables
    mu_max = model.add_variables(
        lower=0, coords=[gens, time], name="mu_max"
    )  # dual of P_max for generation constraint
    nu_max = model.add_variables(
        lower=0, coords=[time], name="nu_max"
    )  # dual of P_max for demand constraint
    zeta_min = model.add_variables(
        lower=0, coords=[gens, time], name="zeta_min"
    )  # dual of P_min for generation constraint

    # pi_u = model.add_variables(
    #     lower=0, coords=[gens, time], name="pi_u"
    # )  # dual of ramping up constraint
    # pi_d = model.add_variables(
    #     lower=0, coords=[gens, time], name="pi_d"
    # )  # dual of ramping down constraint

    sigma_u = model.add_variables(
        lower=0, coords=[gens, time], name="sigma_u"
    )  # dual of start-up constraint
    sigma_d = model.add_variables(
        lower=0, coords=[gens, time], name="sigma_d"
    )  # dual of shut-down constraint
    psi_max = model.add_variables(
        lower=0, coords=[gens, time], name="psi_max"
    )  # dual of k_max constraint

    # primary problem objective
    def primary_objective_rule(model):
        expr = (
            lambda_ * g.loc[opt_gen, :]
            - gens_df.at[opt_gen, "mc"] * g.loc[opt_gen, :]
            - c_up.loc[opt_gen, :]
            - c_down.loc[opt_gen, :]
        )

        return expr.sum()

    # duality gap objective part I
    def duality_gap_part_1_rule(model):
        expr = sum(
            (
                k * gens_df.at[gen, "mc"] * g.loc[gen, :]
                + gens_df.at[gen, "f"] * u.loc[gen, :]
                + c_up.loc[gen, :]
                + c_down.loc[gen, :]
            )
            if gen == opt_gen
            else (
                gens_df.at[gen, "mc"] * g.loc[gen, :]
                + gens_df.at[gen, "f"] * u.loc[gen, :]
                + c_up.loc[gen, :]
                + c_down.loc[gen, :]
            )
            for gen in gens
        )
        expr -= demand_df.loc[:, "price"].values * d
        return expr.sum()

    # duality gap objective part II
    def duality_gap_part_2_rule(model):
        expr = (nu_max * demand_df.loc[:, "volume"].values).sum()

        # expr += sum(
        #     pi_u.loc[i, :] * gens_df.at[i, "r_up"]
        #     + pi_d.loc[i, :] * gens_df.at[i, "r_down"]
        #     for i in gens
        # )
        # expr += sum(
        #     pi_u.loc[i, 1:] * gens_df.at[i, "g_0"] - pi_d.loc[i, 1:] * gens_df.at[i, "g_0"]
        #     for i in gens
        # )

        expr += sum(
            sigma_u.loc[i, 1:] * gens_df.at[i, "k_up"] * gens_df.at[i, "u_0"]
            - sigma_d.loc[i, 1:] * gens_df.at[i, "k_down"] * gens_df.at[i, "u_0"]
            for i in gens
        ).sum()

        expr += psi_max.sum()

        return -expr

    objective = model.add_objective(
        expr=-(
            primary_objective_rule(model)
            - big_w * (duality_gap_part_1_rule(model) - duality_gap_part_2_rule(model))
        )
    )

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

    # ramp_up_constr = model.add_constraints(
    #     ramp_up_rule, coords=[gens, time], name="ramp_up"
    # )

    # ramp down constraint
    def ramp_down_rule(model, i, t):
        if t == 0:
            return -g[i, t] <= max(
                gens_df.at[i, "g_0"] - gens_df.at[i, "r_down"], gens_df.at[i, "g_min"]
            )
        else:
            return g[i, t - 1] - g[i, t] <= gens_df.at[i, "r_down"]

    # ramp_down_constr = model.add_constraints(
    #     ramp_down_rule, coords=[gens, time], name="ramp_down"
    # )

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

    # dual constraints
    def gen_dual_rule(model, i, t):
        if t != time[-1]:
            if i == opt_gen:
                return (
                    k[i] * gens_df.at[i, "mc"]
                    - lambda_[t]
                    + mu_max[i, t]
                    - zeta_min[i, t]
                    # + pi_u[i, t]
                    # - pi_u[i, t + 1]
                    # - pi_d[i, t]
                    # + pi_d[i, t + 1]
                    >= 0
                )
            else:
                return (
                    -lambda_[t] + mu_max[i, t] - zeta_min[i, t]
                    # + pi_u[i, t]
                    # - pi_u[i, t + 1]
                    # - pi_d[i, t]
                    # + pi_d[i, t + 1]
                    >= -gens_df.at[i, "mc"]
                )
        else:
            if i == opt_gen:
                return (
                    k[i] * gens_df.at[i, "mc"]
                    - lambda_[t]
                    + mu_max[i, t]
                    - zeta_min[i, t]
                    # + pi_u[i, t]
                    # - pi_d[i, t]
                    >= 0
                )
            else:
                return (
                    -lambda_[t]
                    + mu_max[i, t]
                    - zeta_min[i, t]  # + pi_u[i, t] - pi_d[i, t]
                    >= -gens_df.at[i, "mc"]
                )

    gen_dual_constr = model.add_constraints(
        gen_dual_rule, coords=[gens, time], name="gen_dual"
    )

    def status_dual_rule(model, i, t):
        if t != time[-1]:
            return (
                -mu_max[i, t] * gens_df.at[i, "g_max"]
                + zeta_min[i, t] * gens_df.at[i, "g_min"]
                + (sigma_u[i, t] - sigma_u[i, t + 1]) * gens_df.at[i, "k_up"]
                - (sigma_d[i, t] - sigma_d[i, t + 1]) * gens_df.at[i, "k_down"]
                + psi_max[i, t]
                >= -gens_df.at[i, "f"]
            )
        else:
            return (
                -mu_max[i, t] * gens_df.at[i, "g_max"]
                + zeta_min[i, t] * gens_df.at[i, "g_min"]
                + sigma_u[i, t] * gens_df.at[i, "k_up"]
                - sigma_d[i, t] * gens_df.at[i, "k_down"]
                + psi_max[i, t]
                >= -gens_df.at[i, "f"]
            )

    status_dual_constr = model.add_constraints(
        status_dual_rule, coords=[gens, time], name="status_dual"
    )

    start_up_dual_constr = model.add_constraints(sigma_u <= 1, name="start_up_dual")
    shut_down_dual_constr = model.add_constraints(sigma_d <= 1, name="shut_down_dual")

    def demand_dual_rule(model, t):
        return lambda_[t] + nu_max[t] >= demand_df.at[t, "price"]

    demand_dual_constr = model.add_constraints(
        demand_dual_rule, coords=[time], name="demand_dual"
    )
    model.solve(solver_name="gurobi", NonConvex=2)

    generation = pd.DataFrame(
        columns=[f"gen_{gen}" for gen in gens], index=time, data=g.solution.T
    ).round(1)
    demand = pd.DataFrame(columns=["demand"], index=time, data=d.solution).round(1)
    mcp = pd.DataFrame(columns=["price"], index=time, data=lambda_.solution).round(3)
    main_df = pd.concat([generation, demand, mcp], axis=1)
    supp_sol = model.solution.to_dataframe()

    return main_df, supp_sol, k.solution


# %%
if __name__ == "__main__":
    # generators
    gens_df = pd.read_csv("inputs/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv("inputs/demand.csv", index_col=0)

    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    opt_gen = 2  # generator that is allowed to bid strategically

    sol, k = find_optimal_k(
        gens_df=gens_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
    )
