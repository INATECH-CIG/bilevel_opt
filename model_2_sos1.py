# %%
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import write_iis


def find_optimal_k_method_2(
    gens_df,
    k_values_df,
    demand_df,
    k_max,
    opt_gen,
    big_w=10000,
    time_limit=60,
    print_results=False,
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
    model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals)
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # secondary variables
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(model.time, within=pyo.NonNegativeReals)

    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))

    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # duals of LP relaxation
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals)
    model.mu_max_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max_hat = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.nu_min_hat = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.pi_u_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # binary expansion variables
    model.g_binary = pyo.Var(model.time, range(K), within=pyo.Binary)
    model.z_lambda = pyo.Var(model.time, range(K), within=pyo.NonNegativeReals)
    model.z_k = pyo.Var(model.time, range(K), within=pyo.NonNegativeReals)

    model.M = pyo.Param(initialize=max(gens_df["mc"]) * k_max)
    delta = [gens_df.at[gen, "g_max"] / (pow(2, K) - 1) for gen in gens_df.index]

    # binary expansion constraints
    def g_binary_rule(model, t):
        return model.g[opt_gen, t] == delta[opt_gen] * sum(
            pow(2, k) * model.g_binary[t, k] for k in range(K)
        )

    model.g_binary_constr = pyo.Constraint(model.time, rule=g_binary_rule)

    def binary_expansion_1_constr_1_max_rule(model, t, n):
        return model.lambda_hat[t] - model.z_lambda[t, n] <= model.M * (
            1 - model.g_binary[t, n]
        )

    def binary_expansion_1_constr_1_min_rule(model, t, n):
        return model.lambda_hat[t] - model.z_lambda[t, n] >= 0

    def binary_expansion_1_constr_2_rule(model, t, n):
        return model.z_lambda[t, n] <= model.M * model.g_binary[t, n]

    model.binary_expansion_1_constr_1_max = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_1_constr_1_max_rule
    )
    model.binary_expansion_1_constr_1_min = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_1_constr_1_min_rule
    )
    model.binary_expansion_1_constr_2 = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_1_constr_2_rule
    )

    def binary_expansion_2_constr_1_max_rule(model, t, n):
        return model.k[t] - model.z_k[t, n] <= model.M * (1 - model.g_binary[t, n])

    def binary_expansion_2_constr_1_min_rule(model, t, n):
        return model.k[t] - model.z_k[t, n] >= 0

    def binary_expansion_2_constr_2_rule(model, t, n):
        return model.z_k[t, n] <= model.M * model.g_binary[t, n]

    model.binary_expansion_2_constr_1_max = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_2_constr_1_max_rule
    )
    model.binary_expansion_2_constr_1_min = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_2_constr_1_min_rule
    )
    model.binary_expansion_2_constr_2 = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_2_constr_2_rule
    )

    # objective rules
    def primary_objective_rule(model):
        return sum(
            delta[opt_gen] * sum(pow(2, n) * model.z_lambda[t, n] for n in range(K))
            - gens_df.at[opt_gen, "mc"] * model.g[opt_gen, t]
            - model.c_up[opt_gen, t]
            - model.c_down[opt_gen, t]
            for t in model.time
        )

    def duality_gap_part_1_rule(model):
        expr = sum(
            (
                gens_df.at[gen, "mc"]
                * delta[gen]
                * sum(pow(2, n) * model.z_k[t, n] for n in range(K))
                + model.c_up[gen, t]
                + model.c_down[gen, t]
            )
            if gen == opt_gen
            else (
                k_values_df.at[t, gen] * gens_df.at[gen, "mc"] * model.g[gen, t]
                + model.c_up[gen, t]
                + model.c_down[gen, t]
            )
            for gen in model.gens
            for t in model.time
        )
        expr -= sum(demand_df.at[t, "price"] * model.d[t] for t in model.time)
        return expr

    def duality_gap_part_2_rule(model):
        expr = -sum(model.nu_max[t] * demand_df.at[t, "volume"] for t in model.time)

        expr -= sum(
            model.pi_u[i, t] * gens_df.at[i, "r_up"]
            for i in model.gens
            for t in model.time
        )

        expr -= sum(
            model.pi_d[i, t] * gens_df.at[i, "r_down"]
            for i in model.gens
            for t in model.time
        )

        expr -= sum(model.pi_u[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)

        expr += sum(model.pi_d[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)

        expr -= sum(
            model.sigma_u[i, 0] * gens_df.at[i, "k_up"] * gens_df.at[i, "u_0"]
            for i in model.gens
        )

        expr += sum(
            model.sigma_d[i, 0] * gens_df.at[i, "k_down"] * gens_df.at[i, "u_0"]
            for i in model.gens
        )

        expr -= sum(model.psi_max[i, t] for i in model.gens for t in model.time)

        return expr

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
        # Conditional parts based on `i` and `t`
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_next_term = 0 if t == model.time.at(-1) else model.pi_u[i, t + 1]
        pi_d_next_term = 0 if t == model.time.at(-1) else model.pi_d[i, t + 1]

        # Combined expression
        return (
            k_term * gens_df.at[i, "mc"]
            - model.lambda_[t]
            + model.mu_max[i, t]
            - model.mu_min[i, t]
            + model.pi_u[i, t]
            - pi_u_next_term
            - model.pi_d[i, t]
            + pi_d_next_term
            == 0
        )

    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

    def status_dual_rule(model, i, t):
        if t != model.time.at(-1):
            return (
                -model.mu_max[i, t] * gens_df.at[i, "g_max"]
                + model.mu_min[i, t] * gens_df.at[i, "g_min"]
                + (model.sigma_u[i, t] - model.sigma_u[i, t + 1])
                * gens_df.at[i, "k_up"]
                - (model.sigma_d[i, t] - model.sigma_d[i, t + 1])
                * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= 0
            )
        else:
            return (
                -model.mu_max[i, t] * gens_df.at[i, "g_max"]
                + model.mu_min[i, t] * gens_df.at[i, "g_min"]
                + model.sigma_u[i, t] * gens_df.at[i, "k_up"]
                - model.sigma_d[i, t] * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= 0
            )

    model.status_dual = pyo.Constraint(model.gens, model.time, rule=status_dual_rule)

    def demand_dual_rule(model, t):
        return -demand_df.at[t, "price"] + model.lambda_[t] + model.nu_max[t] >= 0

    model.demand_dual = pyo.Constraint(model.time, rule=demand_dual_rule)

    # KKT conditions
    # Stationarity conditions
    def kkt_gen_rule(model, i, t):
        # Conditional parts based on `i` and `t`
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_hat_next_term = 0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        pi_d_hat_next_term = 0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]

        # Combined expression
        return (
            k_term * gens_df.at[i, "mc"]
            - model.lambda_hat[t]
            + model.mu_max_hat[i, t]
            - model.mu_min_hat[i, t]
            + model.pi_u_hat[i, t]
            - pi_u_hat_next_term
            - model.pi_d_hat[i, t]
            + pi_d_hat_next_term
            == 0
        )

    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

    def kkt_demand_rule(model, t):
        return (
            -demand_df.at[t, "price"]
            + model.lambda_hat[t]
            + model.nu_max_hat[t]
            - model.nu_min_hat[t]
            == 0
        )

    model.kkt_demand = pyo.Constraint(model.time, rule=kkt_demand_rule)

    # Complementary slackness conditions

    #############################
    model.mu_max_hat_sos = pyo.Var(model.gens, model.time, range(2))

    def mu_max_hat_sos1_rule(model, g, t):
        return ([model.mu_max_hat_sos[g, t, 0], model.mu_max_hat_sos[g, t, 1]], [1, 1])

    model.mu_max_hat_sos1 = pyo.SOSConstraint(
        model.gens, model.time, rule=mu_max_hat_sos1_rule, sos=1
    )

    def mu_max_hat_rule_1(model, i, t):
        return model.mu_max_hat_sos[i, t, 0] == (
            model.g[i, t] - gens_df.at[i, "g_max"] * model.u[i, t]
        )

    def mu_max_hat_rule_2(model, i, t):
        return model.mu_max_hat_sos[i, t, 1] == model.mu_max_hat[i, t]

    model.mu_max_hat_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_rule_1
    )
    model.mu_max_hat_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_rule_2
    )

    #############################
    model.mu_min_hat_sos = pyo.Var(model.gens, model.time, range(2))

    def mu_min_hat_sos1_rule(model, g, t):
        return ([model.mu_min_hat_sos[g, t, 0], model.mu_min_hat_sos[g, t, 1]], [1, 1])

    model.mu_min_hat_sos1 = pyo.SOSConstraint(
        model.gens, model.time, rule=mu_min_hat_sos1_rule, sos=1
    )

    def mu_min_hat_rule_1(model, i, t):
        return (
            model.mu_min_hat_sos[i, t, 0]
            == gens_df.at[i, "g_min"] * model.u[i, t] - model.g[i, t]
        )

    def mu_min_hat_rule_2(model, i, t):
        return model.mu_min_hat_sos[i, t, 1] == model.mu_min_hat[i, t]

    model.mu_min_hat_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=mu_min_hat_rule_1
    )
    model.mu_min_hat_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=mu_min_hat_rule_2
    )

    #############################
    model.nu_max_hat_sos = pyo.Var(model.time, range(2))

    def nu_max_hat_sos1_rule(model, t):
        return ([model.nu_max_hat_sos[t, 0], model.nu_max_hat_sos[t, 1]], [1, 1])

    model.nu_max_hat_sos1 = pyo.SOSConstraint(
        model.time, rule=nu_max_hat_sos1_rule, sos=1
    )

    def nu_max_hat_rule_1(model, t):
        return model.nu_max_hat_sos[t, 0] == model.d[t] - demand_df.at[t, "volume"]

    def nu_max_hat_rule_2(model, t):
        return model.nu_max_hat_sos[t, 1] == model.nu_max_hat[t]

    model.nu_max_hat_constr_1 = pyo.Constraint(model.time, rule=nu_max_hat_rule_1)
    model.nu_max_hat_constr_2 = pyo.Constraint(model.time, rule=nu_max_hat_rule_2)

    #############################
    model.nu_min_hat_sos = pyo.Var(model.time, range(2))

    def nu_min_hat_sos1_rule(model, t):
        return ([model.nu_min_hat_sos[t, 0], model.nu_min_hat_sos[t, 1]], [1, 1])

    model.nu_min_hat_sos1 = pyo.SOSConstraint(
        model.time, rule=nu_min_hat_sos1_rule, sos=1
    )

    def nu_min_hat_rule_1(model, t):
        return model.nu_min_hat_sos[t, 0] == model.d[t]

    def nu_min_hat_rule_2(model, t):
        return model.nu_min_hat_sos[t, 1] == model.nu_min_hat[t]

    model.nu_min_hat_constr_1 = pyo.Constraint(model.time, rule=nu_min_hat_rule_1)
    model.nu_min_hat_constr_2 = pyo.Constraint(model.time, rule=nu_min_hat_rule_2)

    #############################
    model.pi_u_hat_sos = pyo.Var(model.gens, model.time, range(2))

    def pi_u_hat_sos1_rule(model, g, t):
        return ([model.pi_u_hat_sos[g, t, 0], model.pi_u_hat_sos[g, t, 1]], [1, 1])

    model.pi_u_hat_sos1 = pyo.SOSConstraint(
        model.gens, model.time, rule=pi_u_hat_sos1_rule, sos=1
    )

    def pi_u_hat_rule_1(model, i, t):
        if t == 0:
            return (
                model.pi_u_hat_sos[i, t, 0]
                == model.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[i, "r_up"]
            )
        else:
            return (
                model.pi_u_hat_sos[i, t, 0]
                == model.g[i, t] - model.g[i, t - 1] - gens_df.at[i, "r_up"]
            )

    def pi_u_hat_rule_2(model, i, t):
        return model.pi_u_hat_sos[i, t, 1] == model.pi_u_hat[i, t]

    model.pi_u_hat_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_rule_1
    )
    model.pi_u_hat_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_rule_2
    )

    #############################
    model.pi_d_hat_sos = pyo.Var(model.gens, model.time, range(2))

    def pi_d_hat_sos1_rule(model, g, t):
        return ([model.pi_d_hat_sos[g, t, 0], model.pi_d_hat_sos[g, t, 1]], [1, 1])

    model.pi_d_hat_sos1 = pyo.SOSConstraint(
        model.gens, model.time, rule=pi_d_hat_sos1_rule, sos=1
    )

    def pi_d_hat_rule_1(model, i, t):
        if t == 0:
            return (
                model.pi_d_hat_sos[i, t, 0]
                == gens_df.at[i, "g_0"] - model.g[i, t] - gens_df.at[i, "r_down"]
            )
        else:
            return (
                model.pi_d_hat_sos[i, t, 0]
                == model.g[i, t - 1] - model.g[i, t] - gens_df.at[i, "r_down"]
            )

    def pi_d_hat_rule_2(model, i, t):
        return model.pi_d_hat_sos[i, t, 1] == model.pi_d_hat[i, t]

    model.pi_d_hat_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_rule_1
    )
    model.pi_d_hat_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_rule_2
    )

    # solve
    instance = model.create_instance()

    solver = SolverFactory("gurobi")
    options = {
        "LogToConsole": print_results,
        "TimeLimit": time_limit,
        "PreSOS1BigM": -1,
        "MIPGap": 0.02,
        "MIPFocus": 2,
    }
    results = solver.solve(instance, options=options, tee=print_results)

    if (
        results.solver.termination_condition
        == pyo.TerminationCondition.infeasibleOrUnbounded
    ):
        # If the model was infeasible, extract the IIS
        instance.solutions.store_to(results)
        write_iis(model, "iis.ilp")

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

    return main_df, supp_df, k_values, model


# %%
if __name__ == "__main__":
    case = "Case_1"

    big_w = 1  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    opt_gen = 1  # generator that is allowed to bid strategically

    start = pd.to_datetime("2019-03-02 00:00")
    end = pd.to_datetime("2019-03-02 23:00")

    # gens
    gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)

    main_df, supp_df, k_values, model = find_optimal_k_method_2(
        gens_df=gens_df,
        k_values_df=k_values_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
        time_limit=180,
        print_results=True,
        K=3,
    )

    print(main_df)
    print()
    print(k_values)
    # %%
