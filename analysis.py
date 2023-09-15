# %%
import pandas as pd
import numpy as np
import plotly.express as px
from uc_problem import solve_and_get_prices
from utils import calculate_profits
import matplotlib.pyplot as plt

# %%
case = "Case_2"

cashflows = pd.read_csv(
    f"outputs/{case}/unit_dispatch.csv",
    index_col=0,
    parse_dates=True,
)
cashflows = cashflows.loc["2019-03-01"]

profits = pd.DataFrame(index=cashflows.index.unique())
# group by unit and iterate to get profit
for unit, df in cashflows.groupby("unit"):
    profit = df["energy_cashflow"] - df["energy_marginal_costs"]
    profits[f"{unit}_profit"] = round(profit / 1000, 0)
    # profits[f"{unit}_mc"] = (round(df["energy_marginal_costs"]/1000,0))
    # profits[f"{unit}_cf"] = (round(df["energy_cashflow"]/1000,0))

# delete demand_EOM_profit
profits = profits.drop(columns=["demand_EOM_profit"])

# print total profit per unit
print(profits.sum(axis=0))

# using plotly plot total profits per unit
fig = px.bar(
    profits.sum(axis=0),
    title="Total profit per unit using DRL",
    labels={"index": "Unit", "Profit": "Profit [k€]"},
)
# renamy axis to profit in kEUR
fig.update_yaxes(title_text="Profit [k€]")
# remove legend
fig.update_layout(showlegend=False)
fig.show()

# %% load previously saved results
start = pd.to_datetime("2019-03-01 00:00")
end = pd.to_datetime("2019-03-02 00:00")

# generators
gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

# 24 hours of demand first increasing and then decreasing
demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
demand_df.index = pd.to_datetime(demand_df.index)
demand_df = demand_df.loc[start:end]
# reset index to start at 0
demand_df = demand_df.reset_index(drop=True)

approx_main_df = pd.read_csv(f"outputs/{case}/approx_main_df.csv", index_col=0)
approx_supp_df = pd.read_csv(f"outputs/{case}/approx_supp_df.csv")
k_values_after_convergence = pd.read_csv(f"outputs/{case}/k_values_df.csv", index_col=0)
k_values_after_convergence.columns = k_values_after_convergence.columns.astype(int)

# get true prices and profiles
true_main_df, true_supp_df = solve_and_get_prices(
    gens_df, demand_df, k_values_after_convergence
)
# get potential profits as difference between prices and marginal costs multiplied by generation
# and subtracting the startup and shutdown costs
approx_profits = calculate_profits(approx_main_df, approx_supp_df, gens_df)
true_profits = calculate_profits(true_main_df, true_supp_df, gens_df)

# convert to t. EUR
approx_profits /= 1000
true_profits /= 1000

# calculate difference between approximated and true profits
profit_diff = approx_profits - true_profits

# plot the sum of true and approximated profits in one plot as a bar plot for each generator
fig, ax = plt.subplots()
x = np.arange(1, len(gens_df.index) + 1)  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(
    x - width / 2, round(approx_profits.sum()), width, label="Estimated profits"
)
rects2 = ax.bar(x + width / 2, round(true_profits.sum()), width, label="True profits")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Profits in t. EUR")
ax.set_title("Profits by generator")
ax.set_xticks(x)
ax.set_xticklabels(f"Gen {gen+1}" for gen in gens_df.index)
ax.legend()


# also print actual values above the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 2)
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig(f"outputs/{case}/profits.png")
plt.show()

# plot true and approximate prices
fig, ax = plt.subplots()
ax.plot(true_main_df["price"].iloc[1:], label="True prices")
ax.plot(approx_main_df["price"].iloc[1:], label="Approximated prices")
ax.set_title("Prices")
ax.set_ylabel("Price in EUR/MWh")
ax.set_xlabel("Time")
ax.legend()
plt.savefig(f"outputs/{case}/prices.png")
plt.show()


# %%
# plot all profits on one plot as seperate bars
profits_rl = profits
profits_opt = true_profits
# round to 0 decimals
profits_rl = profits_rl.round(0)
profits_opt = profits_opt.round(0)
# rename columns in profits_opt to the same as profits_rl
profits_opt.columns = profits_rl.columns

# plot all profits on one plot as seperate bars
fig = px.bar(
    profits_rl.sum(axis=0),
    title="Total profit per unit using DRL",
    labels={"index": "Unit", "Profit": "Profit [k€]"},
)

# add true profits
fig.add_bar(
    x=profits_opt.sum(axis=0).index,
    y=profits_opt.sum(axis=0).values,
    name="using opt",
)
# change name of the first bar
fig.data[0].name = "using DRL"

# plot side by side instaed of on top of each other
fig.update_layout(barmode="group")

# renamy axis to profit in kEUR
fig.update_yaxes(title_text="Profit [k€]")
# remove legend
# fig.update_layout(showlegend=False)
fig.show()

# %%
