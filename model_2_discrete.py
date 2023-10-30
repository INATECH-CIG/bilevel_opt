# %%
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_k_method_2(
    gens_df,
    k_values_df,
    demand_df,
    k_max,
    opt_gen,
    big_w=10000,
    time_limit=60,
    print_results=False,
    big_M=10e6,
    K=5,
):
    model = pyo.ConcreteModel()

    # sets
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)

    # primary variables
    model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.k = pyo.Var(model.time, bounds=(1, k_max))
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3000))
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # secondary variables
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.zeta_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))

    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # duals of LP relaxation
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3000))
    model.mu_max_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max_hat = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.nu_min_hat = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.zeta_min_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.pi_u_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.sigma_u_hat = pyo.Var(model.gens, model.time, bounds=(0, 1), within=pyo.Reals)
    model.sigma_d_hat = pyo.Var(model.gens, model.time, bounds=(0, 1), within=pyo.Reals)

    # objective rules
    def primary_objective_rule(model):
        return sum(
            model.lambda_hat[t] * model.g[opt_gen, t]
            - gens_df.at[opt_gen, "mc"] * model.g[opt_gen, t]
            - model.c_up[opt_gen, t]
            - model.c_down[opt_gen, t]
            for t in model.time
        )

    def duality_gap_part_1_rule(model):
        expr = sum(
            (
                model.k[t] * gens_df.at[gen, "mc"] * model.g[gen, t]
                + gens_df.at[gen, "f"] * model.u[gen, t]
                + model.c_up[gen, t]
                + model.c_down[gen, t]
            )
            if gen == opt_gen
            else (
                k_values_df.at[t, gen] * gens_df.at[gen, "mc"] * model.g[gen, t]
                + gens_df.at[gen, "f"] * model.u[gen, t]
                + model.c_up[gen, t]
                + model.c_down[gen, t]
            )
            for gen in model.gens
            for t in model.time
        )
        expr -= sum(demand_df.at[t, "price"] * model.d[t] for t in model.time)
        return expr

    def duality_gap_part_2_rule(model):
        expr = sum(model.nu_max[t] * demand_df.at[t, "volume"] for t in model.time)

        expr += sum(
            model.pi_u[i, t] * gens_df.at[i, "r_up"]
            for i in model.gens
            for t in model.time
        )

        expr += sum(
            model.pi_d[i, t] * gens_df.at[i, "r_down"]
            for i in model.gens
            for t in model.time
        )

        expr += sum(model.pi_u[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)

        expr -= sum(model.pi_d[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)

        for t in model.time:
            if t == 0:
                continue
            expr += sum(
                model.sigma_u[i, t] * gens_df.at[i, "k_up"] * gens_df.at[i, "u_0"]
                - model.sigma_d[i, t] * gens_df.at[i, "k_down"] * gens_df.at[i, "u_0"]
                for i in model.gens
            )

        expr += pyo.quicksum(
            model.psi_max[i, t] for i in model.gens for t in model.time
        )

        return -expr

    def final_objective_rule(model):
        return primary_objective_rule(model) - big_w * (
            duality_gap_part_1_rule(model) - duality_gap_part_2_rule(model)
        )

    model.objective = pyo.Objective(expr=final_objective_rule, sense=pyo.maximize)

    # constraints
    # energy balance constraint
    def balance_rule(model, t):
        return model.d[t] - sum(model.g[i, t] for i in model.gens) == 0

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        return model.g[i, t] <= gens_df.at[i, "g_max"] * model.u[i, t]

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # min generation constraint
    def g_min_rule(model, i, t):
        return model.g[i, t] >= gens_df.at[i, "g_min"] * model.u[i, t]

    model.g_min = pyo.Constraint(model.gens, model.time, rule=g_min_rule)

    # max demand constraint
    def d_max_rule(model, t):
        return model.d[t] <= demand_df.at[t, "volume"]

    model.d_max = pyo.Constraint(model.time, rule=d_max_rule)

    # max ramp up constraint
    def ru_max_rule(model, i, t):
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        else:
            return model.g[i, t] - model.g[i, t - 1] <= gens_df.at[i, "r_up"]

    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

    # max ramp down constraint
    def rd_max_rule(model, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] <= gens_df.at[i, "r_down"]
        else:
            return model.g[i, t - 1] - model.g[i, t] <= gens_df.at[i, "r_down"]

    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

    # start up cost constraint
    def start_up_cost_rule(model, i, t):
        if t == 0:
            return (
                model.c_up[i, t]
                >= (model.u[i, t] - gens_df.at[i, "u_0"]) * gens_df.at[i, "k_up"]
            )
        else:
            return (
                model.c_up[i, t]
                >= (model.u[i, t] - model.u[i, t - 1]) * gens_df.at[i, "k_up"]
            )

    model.start_up_cost = pyo.Constraint(
        model.gens, model.time, rule=start_up_cost_rule
    )

    # shut down cost constraint
    def shut_down_cost_rule(model, i, t):
        if t == 0:
            return (
                model.c_down[i, t]
                >= (gens_df.at[i, "u_0"] - model.u[i, t]) * gens_df.at[i, "k_down"]
            )
        else:
            return (
                model.c_down[i, t]
                >= (model.u[i, t - 1] - model.u[i, t]) * gens_df.at[i, "k_down"]
            )

    model.shut_down_cost = pyo.Constraint(
        model.gens, model.time, rule=shut_down_cost_rule
    )

    # dual constraints
    def gen_dual_rule(model, i, t):
        if t != model.time.at(-1):
            return (
                (
                    model.k[t] * gens_df.at[i, "mc"]
                    - model.lambda_[t]
                    + model.mu_max[i, t]
                    - model.zeta_min[i, t]
                    + model.pi_u[i, t]
                    - model.pi_u[i, t + 1]
                    - model.pi_d[i, t]
                    + model.pi_d[i, t + 1]
                    >= 0
                )
                if i == opt_gen
                else (
                    k_values_df.at[t, i] * gens_df.at[i, "mc"]
                    - model.lambda_[t]
                    + model.mu_max[i, t]
                    - model.zeta_min[i, t]
                    + model.pi_u[i, t]
                    - model.pi_u[i, t + 1]
                    - model.pi_d[i, t]
                    + model.pi_d[i, t + 1]
                    >= 0
                )
            )
        if i == opt_gen:
            return (
                model.k[t] * gens_df.at[i, "mc"]
                - model.lambda_[t]
                + model.mu_max[i, t]
                - model.zeta_min[i, t]
                + model.pi_u[i, t]
                - model.pi_d[i, t]
                >= 0
            )
        else:
            return (
                k_values_df.at[t, i] * gens_df.at[i, "mc"]
                - model.lambda_[t]
                + model.mu_max[i, t]
                - model.zeta_min[i, t]
                + model.pi_u[i, t]
                - model.pi_d[i, t]
                >= 0
            )

    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

    def status_dual_rule(model, i, t):
        if t != model.time.at(-1):
            return (
                -model.mu_max[i, t] * gens_df.at[i, "g_max"]
                + model.zeta_min[i, t] * gens_df.at[i, "g_min"]
                + (model.sigma_u[i, t] - model.sigma_u[i, t + 1])
                * gens_df.at[i, "k_up"]
                - (model.sigma_d[i, t] - model.sigma_d[i, t + 1])
                * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= -gens_df.at[i, "f"]
            )
        else:
            return (
                -model.mu_max[i, t] * gens_df.at[i, "g_max"]
                + model.zeta_min[i, t] * gens_df.at[i, "g_min"]
                + model.sigma_u[i, t] * gens_df.at[i, "k_up"]
                - model.sigma_d[i, t] * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= -gens_df.at[i, "f"]
            )

    model.status_dual = pyo.Constraint(model.gens, model.time, rule=status_dual_rule)

    def demand_dual_rule(model, t):
        return model.lambda_[t] + model.nu_max[t] >= demand_df.at[t, "price"]

    model.demand_dual = pyo.Constraint(model.time, rule=demand_dual_rule)

    # KKT condition from LP relaxation
    def kkt_gen_dual_rule(model, i, t):
        if t != model.time.at(-1):
            return (
                (
                    model.k[t] * gens_df.at[i, "mc"]
                    - model.lambda_hat[t]
                    + model.mu_max_hat[i, t]
                    - model.zeta_min_hat[i, t]
                    + model.pi_u_hat[i, t]
                    - model.pi_u_hat[i, t + 1]
                    - model.pi_d_hat[i, t]
                    + model.pi_d_hat[i, t + 1]
                    == 0
                )
                if i == opt_gen
                else (
                    k_values_df.at[t, i] * gens_df.at[i, "mc"]
                    - model.lambda_hat[t]
                    + model.mu_max_hat[i, t]
                    - model.zeta_min_hat[i, t]
                    + model.pi_u_hat[i, t]
                    - model.pi_u_hat[i, t + 1]
                    - model.pi_d_hat[i, t]
                    + model.pi_d_hat[i, t + 1]
                    == 0
                )
            )
        if i == opt_gen:
            return (
                model.k[t] * gens_df.at[i, "mc"]
                - model.lambda_hat[t]
                + model.mu_max_hat[i, t]
                - model.zeta_min_hat[i, t]
                + model.pi_u_hat[i, t]
                - model.pi_d_hat[i, t]
                == 0
            )
        else:
            return (
                k_values_df.at[t, i] * gens_df.at[i, "mc"]
                - model.lambda_hat[t]
                + model.mu_max_hat[i, t]
                - model.zeta_min_hat[i, t]
                + model.pi_u_hat[i, t]
                - model.pi_d_hat[i, t]
                == 0
            )

    model.gen_dual_kkt = pyo.Constraint(model.gens, model.time, rule=kkt_gen_dual_rule)

    def kkt_demand_rule(model, t):
        return (
            model.lambda_hat[t]
            + model.nu_max_hat[t]
            - model.nu_min_hat[t]
            - demand_df.at[t, "price"]
            == 0
        )

    model.demand_dual_kkt = pyo.Constraint(model.time, rule=kkt_demand_rule)

    # complementary slackness
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def mu_max_hat_binary_rule_1(model, i, t):
        return (gens_df.at[i, "g_max"] * model.u[i, t] - model.g[i, t]) <= max(
            gens_df["g_max"]
        ) * (1 - model.mu_max_hat_binary[i, t])

    def mu_max_hat_binary_rule_2(model, i, t):
        return model.mu_max_hat[i, t] <= big_M * model.mu_max_hat_binary[i, t]

    def mu_max_hat_binary_rule_3(model, i, t):
        return (gens_df.at[i, "g_max"] * model.u[i, t] - model.g[i, t]) >= 0

    model.mu_max_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_1
    )
    model.mu_max_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_2
    )
    model.mu_max_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_3
    )

    model.zeta_min_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def zeta_min_hat_binary_rule_1(model, i, t):
        return model.g[i, t] - gens_df.at[i, "g_min"] * model.u[i, t] <= max(
            gens_df["g_max"]
        ) * (1 - model.zeta_min_hat_binary[i, t])

    def zeta_min_hat_binary_rule_2(model, i, t):
        return model.zeta_min_hat[i, t] <= big_M * model.zeta_min_hat_binary[i, t]

    def zeta_min_hat_binary_rule_3(model, i, t):
        return model.g[i, t] - gens_df.at[i, "g_min"] * model.u[i, t] >= 0

    model.zeta_min_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=zeta_min_hat_binary_rule_1
    )
    model.zeta_min_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=zeta_min_hat_binary_rule_2
    )
    model.zeta_min_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=zeta_min_hat_binary_rule_3
    )

    model.nu_max_hat_binary = pyo.Var(model.time, within=pyo.Binary)

    def nu_max_hat_binary_rule_1(model, t):
        return demand_df.at[t, "volume"] - model.d[t] <= max(demand_df["volume"]) * (
            1 - model.nu_max_hat_binary[t]
        )

    def nu_max_hat_binary_rule_2(model, t):
        return model.nu_max_hat[t] <= big_M * model.nu_max_hat_binary[t]

    def nu_max_hat_binary_rule_3(model, t):
        return demand_df.at[t, "volume"] - model.d[t] >= 0

    model.nu_max_hat_binary_constr_1 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_1
    )
    model.nu_max_hat_binary_constr_2 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_2
    )
    model.nu_max_hat_binary_constr_3 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_3
    )

    model.nu_min_hat_binary = pyo.Var(model.time, within=pyo.Binary)

    def nu_min_hat_binary_rule_1(model, t):
        return model.d[t] <= max(demand_df["volume"]) * (1 - model.nu_min_hat_binary[t])

    def nu_min_hat_binary_rule_2(model, t):
        return model.nu_min_hat[t] <= big_M * model.nu_min_hat_binary[t]

    model.nu_min_hat_binary_constr_1 = pyo.Constraint(
        model.time, rule=nu_min_hat_binary_rule_1
    )
    model.nu_min_hat_binary_constr_2 = pyo.Constraint(
        model.time, rule=nu_min_hat_binary_rule_2
    )

    model.sigma_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def sigma_u_hat_binary_rule_1(model, i, t):
        if t == 0:
            return model.c_up[i, t] - (
                model.u[i, t] - gens_df.at[i, "u_0"]
            ) * gens_df.at[i, "k_up"] <= max(gens_df["k_up"]) * (
                1 - model.sigma_u_hat_binary[i, t]
            )
        else:
            return model.c_up[i, t] - (model.u[i, t] - model.u[i, t - 1]) * gens_df.at[
                i, "k_up"
            ] <= max(gens_df["k_up"]) * (1 - model.sigma_u_hat_binary[i, t])

    def sigma_u_hat_binary_rule_2(model, i, t):
        return model.sigma_u_hat[i, t] <= big_M * model.sigma_u_hat_binary[i, t]

    def sigma_u_hat_binary_rule_3(model, i, t):
        if t == 0:
            return (
                model.c_up[i, t]
                - (model.u[i, t] - gens_df.at[i, "u_0"]) * gens_df.at[i, "k_up"]
                >= 0
            )
        else:
            return (
                model.c_up[i, t]
                - (model.u[i, t] - model.u[i, t - 1]) * gens_df.at[i, "k_up"]
                >= 0
            )

    model.sigma_u_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=sigma_u_hat_binary_rule_1
    )
    model.sigma_u_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=sigma_u_hat_binary_rule_2
    )
    model.sigma_u_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=sigma_u_hat_binary_rule_3
    )

    model.sigma_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def sigma_d_hat_binary_rule_1(model, i, t):
        if t == 0:
            return model.c_down[i, t] - (
                gens_df.at[i, "u_0"] - model.u[i, t]
            ) * gens_df.at[i, "k_down"] <= max(gens_df["k_down"]) * (
                1 - model.sigma_d_hat_binary[i, t]
            )
        else:
            return model.c_down[i, t] - (
                model.u[i, t - 1] - model.u[i, t]
            ) * gens_df.at[i, "k_down"] <= max(gens_df["k_down"]) * (
                1 - model.sigma_d_hat_binary[i, t]
            )

    def sigma_d_hat_binary_rule_2(model, i, t):
        return model.sigma_d_hat[i, t] <= big_M * model.sigma_d_hat_binary[i, t]

    def sigma_d_hat_binary_rule_3(model, i, t):
        if t == 0:
            return (
                model.c_down[i, t]
                - (gens_df.at[i, "u_0"] - model.u[i, t]) * gens_df.at[i, "k_down"]
                >= 0
            )
        else:
            return (
                model.c_down[i, t]
                - (model.u[i, t - 1] - model.u[i, t]) * gens_df.at[i, "k_down"]
                >= 0
            )

    model.sigma_d_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=sigma_d_hat_binary_rule_1
    )
    model.sigma_d_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=sigma_d_hat_binary_rule_2
    )
    model.sigma_d_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=sigma_d_hat_binary_rule_3
    )

    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_u_hat_binary_rule_1(model, i, t):
        if t == 0:
            return gens_df.at[i, "r_up"] - (
                model.g[i, t] - gens_df.at[i, "g_0"]
            ) <= big_M * (1 - model.pi_u_hat_binary[i, t])
        else:
            return gens_df.at[i, "r_up"] - (model.g[i, t] - model.g[i, t - 1]) <= big_M * (1 - model.pi_u_hat_binary[i, t])

    def pi_u_hat_binary_rule_2(model, i, t):
        return model.pi_u_hat[i, t] <= big_M * model.pi_u_hat_binary[i, t]

    def pi_u_hat_binary_rule_3(model, i, t):
        if t == 0:
            return gens_df.at[i, "r_up"] - (model.g[i, t] - gens_df.at[i, "g_0"]) >= 0
        else:
            return gens_df.at[i, "r_up"] - (model.g[i, t] - model.g[i, t - 1]) >= 0

    model.pi_u_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_1
    )
    model.pi_u_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_2
    )
    model.pi_u_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_3
    )

    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_d_hat_binary_rule_1(model, i, t):
        if t == 0:
            return gens_df.at[i, "r_down"] - (
                gens_df.at[i, "g_0"] - model.g[i, t]
            ) <= big_M * (1 - model.pi_d_hat_binary[i, t])
        else:
            return gens_df.at[i, "r_down"] - (model.g[i, t - 1] - model.g[i, t]) <= big_M * (1 - model.pi_d_hat_binary[i, t])

    def pi_d_hat_binary_rule_2(model, i, t):
        return model.pi_d_hat[i, t] <= big_M * model.pi_d_hat_binary[i, t]

    def pi_d_hat_binary_rule_3(model, i, t):
        if t == 0:
            return gens_df.at[i, "r_down"] - (gens_df.at[i, "g_0"] - model.g[i, t]) >= 0
        else:
            return gens_df.at[i, "r_down"] - (model.g[i, t - 1] - model.g[i, t]) >= 0

    model.pi_d_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_1
    )
    model.pi_d_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_2
    )
    model.pi_d_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_3
    )

    # solve
    instance = model.create_instance()

    solver = SolverFactory("gurobi")
    options = {"NonConvex": 2, "LogToConsole": print_results, "TimeLimit": time_limit}

    results = solver.solve(instance, options=options, tee=print_results)

    # check if solver exited due to time limit
    if results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        print("Solver did not converge to an optimal solution")

    generation_df = pd.DataFrame(
        index=demand_df.index, columns=[f"gen_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            generation_df.at[t, f"gen_{gen}"] = instance.g[gen, t].value

    demand_df = pd.DataFrame(index=demand_df.index, columns=["demand"])
    for t in demand_df.index:
        demand_df.at[t, "demand"] = instance.d[t].value

    mcp = pd.DataFrame(index=demand_df.index, columns=["mcp"])
    for t in demand_df.index:
        mcp.at[t, "mcp"] = instance.lambda_[t].value

    mcp_hat = pd.DataFrame(index=demand_df.index, columns=["mcp_hat"])
    for t in demand_df.index:
        mcp_hat.at[t, "mcp_hat"] = instance.lambda_hat[t].value

    main_df = pd.concat([generation_df, demand_df, mcp, mcp_hat], axis=1)

    start_up_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"start_up_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            start_up_cost.at[t, f"start_up_{gen}"] = instance.c_up[gen, t].value

    shut_down_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"shut_down_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            shut_down_cost.at[t, f"shut_down_{gen}"] = instance.c_down[gen, t].value

    supp_df = pd.concat([start_up_cost, shut_down_cost], axis=1)

    k_values = pd.DataFrame(index=demand_df.index, columns=["k"])
    for t in demand_df.index:
        k_values.at[t, "k"] = instance.k[t].value

    return main_df, supp_df, k_values


# %%
if __name__ == "__main__":
    case = "Case_1"

    big_w = 1000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding

    start = pd.to_datetime("2019-03-02 00:00")
    end = pd.to_datetime("2019-03-03 00:00")

    # gens
    gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
    opt_gen = 2  # generator that is allowed to bid strategically

    main_df, supp_df, k_values = find_optimal_k_method_2(
        gens_df=gens_df,
        k_values_df=k_values_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
        print_results=True,
        K=5,
        big_M=10e6,
        time_limit=60,
    )

    print(main_df)
    print()
    print(k_values)
# %%
