# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from uc_problem import solve_uc_problem
from utils import calculate_profits

# %%
case = "Case_1"
start = pd.to_datetime("2019-03-02 00:00")
end = pd.to_datetime("2019-03-03 00:00")

# gens
gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)
demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
demand_df.index = pd.to_datetime(demand_df.index)
demand_df = demand_df.loc[start:end]
# reset index to start at 0
demand_df = demand_df.reset_index(drop=True)

path = f"outputs/{case}/method_1"

# %% RL Part
market_orders = pd.read_csv(
    f"{path}/market_orders.csv",
    index_col=0,
    parse_dates=True,
)

rl_profits = pd.DataFrame(index=demand_df.index, columns=gens_df.index, data=0.0)
for opt_gen in gens_df.index:
    rl_unit_orders = market_orders[market_orders["unit_id"] == f"Unit_{opt_gen}"]
    rl_unit_orders = rl_unit_orders.loc[start:end]
    rl_unit_orders = rl_unit_orders.reset_index(drop=False)
    marginal_cost = gens_df.at[opt_gen, "mc"]
    rl_profits[opt_gen] = rl_unit_orders["accepted_volume"] * (
        rl_unit_orders["accepted_price"] - marginal_cost
    )

    # iterate over all rows and subtract start up and shut down costs if the unit turned on or off
    for t in range(1, len(rl_unit_orders)):
        if t == 1:
            if (
                rl_unit_orders.at[t, "accepted_volume"] > 0
                and gens_df.at[opt_gen, "u_0"] == 0
            ):
                rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_up"]
            elif (
                rl_unit_orders.at[t, "accepted_volume"] == 0
                and gens_df.at[opt_gen, "u_0"] > 0
            ):
                rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_down"]
        elif (
            rl_unit_orders.at[t, "accepted_volume"] == 0
            and rl_unit_orders.at[t - 1, "accepted_volume"] > 0
        ):
            rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_down"]
        elif (
            rl_unit_orders.at[t, "accepted_volume"] > 0
            and rl_unit_orders.at[t - 1, "accepted_volume"] == 0
        ):
            rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_up"]

# %%
uplift_method_rl, uplift_df_method_rl = calculate_uplift(
    rl_unit_orders, gens_df, opt_gen, "accepted_price", rl_unit_profits.sum()
)

total_profit_with_uplift_method_rl = rl_unit_profits.sum() + uplift_method_rl

# %%
# using plotly plot total profits per unit
fig = px.bar(
    rl_profits.sum(axis=0),
    title="Total profit per unit using DRL",
    labels={"index": "Unit", "Profit": "Profit [k€]"},
)
# renamy axis to profit in kEUR
fig.update_yaxes(title_text="Profit [k€]")
# remove legend
fig.update_layout(showlegend=False)
fig.show()

# %% load previously saved results for method_1
main_df_1 = pd.read_csv(f"outputs/{case}/method_1/main_df.csv", index_col=0)
supp_df_1 = pd.read_csv(f"outputs/{case}/method_1/supp_df.csv")
k_values_1 = pd.read_csv(f"outputs/{case}/method_1/k_values_df.csv", index_col=0)
k_values_1.columns = k_values_1.columns.astype(int)

# get true prices and profiles
updated_main_df_1, updated_supp_df_1 = solve_uc_problem(
    gens_df, demand_df, k_values_1
)
# get potential profits as difference between prices and marginal costs multiplied by generation
# and subtracting the startup and shutdown costs

profits_method_1 = calculate_profits(main_df_1, supp_df_1, gens_df)
updated_profits_method_1 = calculate_profits(updated_main_df_1, updated_supp_df_1, gens_df, price_column="price")

# make a dataframe with the total profits per unit
total_profits_method_1 = pd.DataFrame(
    index=profits_method_1.columns, columns=["Method 1"], data=profits_method_1.sum()
)
updated_total_profits_method_1 = pd.DataFrame(
    index=updated_profits_method_1.columns, columns=["Method 1 (updated)"], data=updated_profits_method_1.sum()
).astype(float)

# construct a dataframe with the total profits per unit
profits = pd.concat(
    [total_profits_method_1, updated_total_profits_method_1], axis=1, sort=False
)

# plot the profits as bars
fig = px.bar(
    profits,
    x=profits.index,
    y=profits.columns,
    title="Total profit per unit using Method 1",
    labels={"index": "Unit", "Profit": "Profit [k€]"},
    barmode="group",
)

fig.update_yaxes(title_text="Profit [€]")

fig.show()

# %%

#merge all profits
sum_rl_profits = pd.DataFrame(index=profits_method_1.columns, columns=["DRL"], data=rl_profits.sum())
all_profits = pd.concat([profits, sum_rl_profits], axis=1, sort=False)
all_profits /= 1000
all_profits = all_profits.round()
# plot the profits as bars
fig = px.bar(
    all_profits,
    x=all_profits.index,
    y=all_profits.columns,
    title="Total profit per unit",
    labels={"index": "Unit", "Profit": "Profit [k€]"},
    barmode="group",
)

#display values on top
fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
fig.update_yaxes(title_text="Profit [k€]")
#save figure as html
fig.write_html("outputs/total_profits.html")
# and as pdf
fig.write_image("outputs/total_profits.pdf")

fig.show()
# %%
