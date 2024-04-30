# %%
import pandas as pd
import plotly.express as px

from uc_problem import solve_uc_problem
from utils import calculate_profits

# %%
case = "Case_1"
start = pd.to_datetime("2019-03-02 06:00")
end = pd.to_datetime("2019-03-02 14:00")

# gens
gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)
demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
demand_df.index = pd.to_datetime(demand_df.index)
demand_df = demand_df.loc[start:end]
# reset index to start at 0
demand_df = demand_df.reset_index(drop=True)

k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
# %% load previously saved results for method_1
main_df_1 = pd.read_csv(f"outputs/{case}/method_1/main_df.csv", index_col=0)
supp_df_1 = pd.read_csv(f"outputs/{case}/method_1/supp_df.csv")
k_values_1 = pd.read_csv(f"outputs/{case}/method_1/k_values_df.csv", index_col=0)
k_values_1.columns = k_values_1.columns.astype(int)

# get true prices and profiles
updated_main_df_1, updated_supp_df_1 = solve_uc_problem(gens_df, demand_df, k_values_1)

profits_method_1 = calculate_profits(updated_main_df_1, updated_supp_df_1, gens_df)
# make a dataframe with the total profits per unit
total_profits_method_1 = pd.DataFrame(
    index=profits_method_1.columns,
    columns=["Method 1"],
    data=profits_method_1.sum(),
).astype(float)

# repeat the same for method_2
main_df_2 = pd.read_csv(f"outputs/{case}/method_2/main_df.csv", index_col=0)
supp_df_2 = pd.read_csv(f"outputs/{case}/method_2/supp_df.csv")
k_values_2 = pd.read_csv(f"outputs/{case}/method_2/k_values_df.csv", index_col=0)
k_values_2.columns = k_values_2.columns.astype(int)

updated_main_df_2, updated_supp_df_2 = solve_uc_problem(gens_df, demand_df, k_values_2)
profits_method_2 = calculate_profits(updated_main_df_2, updated_supp_df_2, gens_df)

total_profits_method_2 = pd.DataFrame(
    index=profits_method_2.columns,
    columns=["Method 2"],
    data=profits_method_2.sum(),
).astype(float)

# %% RL Part
market_orders = pd.read_csv(
    f"outputs/{case}/method_1/market_orders.csv",
    index_col=0,
    parse_dates=True,
)

k_values_df_3 = k_values_df.copy()
for gen in gens_df.index:
    unit_id = f"Unit_{gen}"
    rl_unit_orders = market_orders[market_orders["unit_id"] == unit_id]
    rl_unit_orders = rl_unit_orders.loc[start:end]
    rl_unit_orders = rl_unit_orders.reset_index(drop=False)

    k_values_df_3[gen] = rl_unit_orders["price"] / gens_df.at[gen, "mc"]

main_df_3, supp_df_3 = solve_uc_problem(gens_df, demand_df, k_values_df_3)

profits_method_3 = calculate_profits(main_df_3, supp_df_3, gens_df)

sum_rl_profits = pd.DataFrame(
    index=total_profits_method_1.index,
    columns=["DRL"],
    data=profits_method_3.sum(),
).astype(float)

# %%
# merge all profits

all_profits = pd.concat(
    [total_profits_method_1, total_profits_method_2, sum_rl_profits], axis=1
)
all_profits /= 1000
all_profits = all_profits.round()

# drop index 3
all_profits = all_profits.drop(3)

# rename index to i+1
all_profits.index = all_profits.index + 1

# plot the profits as bars
fig = px.bar(
    all_profits,
    x=all_profits.index,
    y=all_profits.columns,
    title=f"Total profits per unit",
    labels={"index": "Unit", "Profit": "Profit [k€]"},
    barmode="group",
)

# display values on top
fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
fig.update_yaxes(title_text="Profit [k€]")
# save figure as html
# fig.write_html(f"outputs/total_profits_{method}.html")
# and as pdf
fig.write_image(f"outputs/total_profits_diag.pdf")

fig.show()

# %% Market clearing price
mcp_method_1 = main_df_1["mcp"]
mcp_method_3 = main_df_3["mcp"]

mcp = pd.concat([mcp_method_1, mcp_method_3], axis=1)
mcp.columns = ["Method 1", "Method 3 (RL)"]

# convert all columns to numeric data types
mcp = mcp.apply(pd.to_numeric, errors="coerce")

# rename index to time
mcp.index.name = "Time step"

# plot bids over time
fig = px.line(
    mcp,
    title="Market Clearing Price",
    labels={"Time": "Time", "value": "MCP [€/MWh]"},
)

fig.update_yaxes(title_text="MCP [€/MWh]")
fig.update_layout(showlegend=True)
# save plot as html
# fig.write_html(f"outputs/{case}/mcp_{opt_gen}.html")
# save plot as pdf
# fig.write_image(f"outputs/{case}/mcp_{opt_gen}.pdf")
fig.show()
