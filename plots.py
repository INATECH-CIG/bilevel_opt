# %%
import os
from uc_problem import solve_uc_problem
from utils import calculate_profits

import pandas as pd
import plotly.express as px

case = "Case_1"

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

font_size = 22
# %%
# plot demand_df
# Create a line plot for demand data
fig = px.line(
    demand_df,
    x=demand_df.index,  # Assuming 'Time' is a column in demand_df
    y="volume",
    labels={"volume": "Demand [MWh]"},
    # title='Demand Profile Over Time'
)

# Add capacity of each unit as horizontal line and accumulate the capacity
capacity = 0
for i, row in gens_df.iterrows():
    capacity += row["g_max"]
    fig.add_hline(
        y=capacity,
        line_dash="dash",
        annotation_text=f"Unit {i+1} capacity: {row['g_max']} MW",
        annotation_position="top right",
    )

# Update y-axis range and title
fig.update_yaxes(range=[0, 4000], title_text="Demand [MW]")

# Update x-axis title
fig.update_xaxes(title_text="Time Step")

# Enhance the layout
fig.update_layout(
    template="plotly_white",  # Clean and professional template
    showlegend=False,
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins to fit the title
    font=dict(
        size=font_size, family="Arial", color="black"
    ),  # Set global font size and family
)

# Save plot as PDF
fig.write_image(f"outputs/{case}/demand.pdf")
# 
# Optionally display the figure
fig.show()

# %%
# plot marginal costs of each unit as bar plot and name it marginal cost
# Reshape the DataFrame to have separate rows for each unit's marginal cost and its multiplied value

df = pd.DataFrame(
    {
        "Unit": gens_df.index[:3] + 1,
        "Marginal cost": gens_df["mc"][:3],
        "Bidding interval": gens_df["mc"][:3],
        "Total interval": 2 * gens_df["mc"][:3],
    }
)

# Create the figure object
fig = px.bar(
    df,
    x="Unit",
    y=["Marginal cost", "Bidding interval"],
    labels={"x": "Unit", "y": "Marginal cost [€/MWh]"},
)

# Update the opacity for the bidding interval bars to make them semi-transparent
fig.update_traces(opacity=1, selector=dict(name="Bidding interval"))

# Display the values from df["Bidding interval"] on top of the bidding interval bars
fig.update_traces(
    text=df["Total interval"],
    textposition="outside",
    selector=dict(name="Bidding interval"),
    textfont=dict(color="black"),
)

# Display the values from df["Marginal cost"] on top of the marginal cost bars
fig.update_traces(
    text=df["Marginal cost"],
    textposition="outside",
    selector=dict(name="Marginal cost"),
)

# Update layout for a cleaner look
fig.update_layout(
    legend_title_text=" ",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=font_size, family="Arial", color="black"),
    template="plotly_white",
)

# Set y-axis range and title
fig.update_yaxes(range=[0, 200], title_text="Marginal cost [€/MWh]")

# Set x-axis to only show integer labels
fig.update_xaxes(tickvals=df["Unit"])

# Save plot as PDF
fig.write_image(f"outputs/{case}/marginal_costs.pdf")

# Optionally display the figure
fig.show()

# %%
for opt_gen in range(3):
    # load data
    path = f"outputs/{case}/gen_{opt_gen}"
    save_path = f"outputs/{case}/gen_{opt_gen}"

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)

    k_values_1 = pd.read_csv(f"{path}/k_values_1.csv", index_col=0)
    k_values_2 = pd.read_csv(f"{path}/k_values_2.csv", index_col=0)

    updated_main_df_1 = pd.read_csv(f"{path}/updated_main_df_1.csv", index_col=0)
    updated_supp_df_1 = pd.read_csv(f"{path}/updated_supp_df_1.csv", index_col=0)

    updated_main_df_2 = pd.read_csv(f"{path}/updated_main_df_2.csv", index_col=0)
    updated_supp_df_2 = pd.read_csv(f"{path}/updated_supp_df_2.csv", index_col=0)

    updated_profits_method_1 = calculate_profits(
        updated_main_df_1, updated_supp_df_1, gens_df, price_column="mcp"
    )
    updated_profits_method_2 = calculate_profits(
        updated_main_df_2, updated_supp_df_2, gens_df, price_column="mcp"
    )

    # RL Part
    market_orders = pd.read_csv(
        f"{path}/market_orders.csv",
        index_col=0,
        parse_dates=True,
    )
    unit_id = f"Unit_{opt_gen}"
    rl_unit_orders = market_orders[market_orders["unit_id"] == unit_id]
    rl_unit_orders = rl_unit_orders.loc[start:end]
    rl_unit_orders = rl_unit_orders.reset_index(drop=False)

    k_values_df_3 = k_values_df.copy()
    k_values_df_3[opt_gen] = rl_unit_orders["price"] / gens_df.at[opt_gen, "mc"]

    main_df_3, supp_df_3 = solve_uc_problem(gens_df, demand_df, k_values_df_3)

    profits_method_3 = calculate_profits(main_df_3, supp_df_3, gens_df)

    main_df_3.to_csv(f"{save_path}/updated_main_df_3.csv")
    supp_df_3.to_csv(f"{save_path}/updated_supp_df_3.csv")

# %%
# plot sum of both profits as bar chart
profits = pd.concat(
    [
        updated_profits_method_1[opt_gen],
        updated_profits_method_2[opt_gen],
        profits_method_3[opt_gen],
    ],
    axis=1,
)
profits.columns = [
    "Method 1 (after UC)",
    "Method 2 (after UC)",
    "Method 3 (RL)",
]

# %%
profits = profits/1e3

profits = profits.apply(pd.to_numeric, errors="coerce")
fig = px.bar(
    title=f"Total profits of Unit {opt_gen+1}",
    labels={"index": "Method", "Profit": "Profit [€]"},
)

# add Method 1 (after UC) bar
fig.add_bar(
    x=["Method 1"],
    y=[profits["Method 1 (after UC)"].sum()],
    name="Method 1",
)

# add Method 2 (after UC) bar
fig.add_bar(
    x=["Method 2 (with KKTs)"],
    y=[profits["Method 2 (after UC)"].sum()],
    name="Method 2",
)

# add Method 3 (RL) bar
fig.add_bar(
    x=["Method 3 (RL)"],
    y=[profits["Method 3 (RL)"].sum()],
    name="Method 3 (RL)",
)

# make all bares with Method 1 in name blue
for i in range(len(fig.data)):
    if "Method 1" in fig.data[i].name:
        fig.data[i].marker.color = "blue"

# make all bares with Method 2 in name orange
for i in range(len(fig.data)):
    if "Method 2" in fig.data[i].name:
        fig.data[i].marker.color = "orange"

# make all bares with Method 3 in name green
for i in range(len(fig.data)):
    if "Method 3" in fig.data[i].name:
        fig.data[i].marker.color = "green"

# display values on top of bars
fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")

# adjust y axis range to fit all bars and text above them
# fig.update_yaxes(range=[0, 0.7e6])

fig.update_yaxes(title_text="Profit [mln.€]")
fig.update_layout(showlegend=False)

# save plot as pdf
fig.write_image(f"outputs/{case}/profits_{opt_gen}.pdf")

# save plot as html
# fig.write_html(f"outputs/{case}/profits_{opt_gen}.html")
fig.show()
