    # complementary slackness
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def mu_max_hat_binary_rule_1(model, i, t):
        return model.g[i, t] - gens_df.at[i, "g_max"] * model.u[i, t] <= bigM * (
            1 - model.mu_max_hat_binary[i, t]
        )

    def mu_max_hat_binary_rule_2(model, i, t):
        return model.mu_max_hat[i, t] <= bigM * model.mu_max_hat_binary[i, t]

    model.mu_max_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_1
    )
    model.mu_max_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_2
    )

    model.zeta_min_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def zeta_min_hat_binary_rule_1(model, i, t):
        return gens_df.at[i, "g_min"] * model.u[i, t] - model.g[i, t] <= bigM * (
            1 - model.zeta_min_hat_binary[i, t]
        )

    def zeta_min_hat_binary_rule_2(model, i, t):
        return model.zeta_min_hat[i, t] <= bigM * model.zeta_min_hat_binary[i, t]

    model.zeta_min_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=zeta_min_hat_binary_rule_1
    )
    model.zeta_min_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=zeta_min_hat_binary_rule_2
    )

    model.nu_max_hat_binary = pyo.Var(model.time, within=pyo.Binary)

    def nu_max_hat_binary_rule_1(model, t):
        return model.d[t] - demand_df.at[t, "volume"] <= bigM * (
            1 - model.nu_max_hat_binary[t]
        )

    def nu_max_hat_binary_rule_2(model, t):
        return model.nu_max_hat[t] <= bigM * model.nu_max_hat_binary[t]

    model.nu_max_hat_binary_constr_1 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_1
    )
    model.nu_max_hat_binary_constr_2 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_2
    )

    model.nu_min_hat_binary = pyo.Var(model.time, within=pyo.Binary)

    def nu_min_hat_binary_rule_1(model, t):
        return model.d[t] <= bigM * (1 - model.nu_min_hat_binary[t])

    def nu_min_hat_binary_rule_2(model, t):
        return model.nu_min_hat[t] <= bigM * model.nu_min_hat_binary[t]

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
            ) * gens_df.at[i, "k_up"] <= bigM * (1 - model.sigma_u_hat_binary[i, t])
        else:
            return model.c_up[i, t] - (model.u[i, t] - model.u[i, t - 1]) * gens_df.at[
                i, "k_up"
            ] <= bigM * (1 - model.sigma_u_hat_binary[i, t])

    def sigma_u_hat_binary_rule_2(model, i, t):
        return model.sigma_u_hat[i, t] <= bigM * model.sigma_u_hat_binary[i, t]

    model.sigma_u_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=sigma_u_hat_binary_rule_1
    )
    model.sigma_u_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=sigma_u_hat_binary_rule_2
    )

    model.sigma_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def sigma_d_hat_binary_rule_1(model, i, t):
        if t == 0:
            return model.c_down[i, t] - (
                gens_df.at[i, "u_0"] - model.u[i, t]
            ) * gens_df.at[i, "k_down"] <= bigM * (1 - model.sigma_d_hat_binary[i, t])
        else:
            return model.c_down[i, t] - (
                model.u[i, t - 1] - model.u[i, t]
            ) * gens_df.at[i, "k_down"] <= bigM * (1 - model.sigma_d_hat_binary[i, t])

    def sigma_d_hat_binary_rule_2(model, i, t):
        return model.sigma_d_hat[i, t] <= bigM * model.sigma_d_hat_binary[i, t]

    model.sigma_d_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=sigma_d_hat_binary_rule_1
    )
    model.sigma_d_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=sigma_d_hat_binary_rule_2
    )

    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_u_hat_binary_rule_1(model, i, t):
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[
                i, "r_up"
            ] <= bigM * (1 - model.pi_u_hat_binary[i, t])
        else:
            return model.g[i, t] - model.g[i, t - 1] - gens_df.at[i, "r_up"] <= bigM * (
                1 - model.pi_u_hat_binary[i, t]
            )

    def pi_u_hat_binary_rule_2(model, i, t):
        return model.pi_u_hat[i, t] <= bigM * model.pi_u_hat_binary[i, t]

    model.pi_u_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_1
    )
    model.pi_u_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_2
    )

    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_d_hat_binary_rule_1(model, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] <= bigM * (1 - model.pi_d_hat_binary[i, t])
        else:
            return model.g[i, t - 1] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] <= bigM * (1 - model.pi_d_hat_binary[i, t])

    def pi_d_hat_binary_rule_2(model, i, t):
        return model.pi_d_hat[i, t] <= bigM * model.pi_d_hat_binary[i, t]

    model.pi_d_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_1
    )
    model.pi_d_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_2
    )

    # solve
    instance = model.create_instance()

    solver = SolverFactory("gurobi")
    options = {"LogToConsole": print_results, "TimeLimit": time_limit}
