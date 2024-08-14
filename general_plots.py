# %%
import plotly.express as px

from uc_problem import solve_uc_problem
from utils import calculate_profits

import pandas as pd

case = "Case_1"

start = pd.to_datetime("2019-03-02 06:00")
end = pd.to_datetime("2019-03-02 14:00")

# gens
gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

# 24 hours of demand first increasing and then decreasing
demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
demand_df.index = pd.to_datetime(demand_df.index)
demand_df = demand_df.loc[start:end]
# reset index to start at 0
demand_df = demand_df.reset_index(drop=True)

font_size = 22
# %% plot demand_df
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

# %% plot marginal costs of each unit as bar plot and name it marginal cost
# Reshape the DataFrame to have separate rows for each unit's marginal cost and its multiplied value

df = pd.DataFrame(
    {
        "Unit": gens_df.index[:4] + 1,
        "Marginal cost": gens_df["mc"][:4],
        "Bidding interval": gens_df["mc"][:4],
        "Total interval": 2 * gens_df["mc"][:4],
    }
)

df.at[3, "Bidding interval"] = -1
df.at[3, "Total interval"] = -1

# Create the figure object
fig = px.bar(
    df,
    x="Unit",
    y=["Marginal cost", "Bidding interval"],
    labels={"x": "Unit", "y": "Price [€/MWh]"},
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
fig.update_yaxes(range=[0, 200], title_text="Price [€/MWh]")

# Set x-axis to only show integer labels
fig.update_xaxes(tickvals=df["Unit"])

# Save plot as PDF
fig.write_image(f"outputs/{case}/marginal_costs.pdf")

# Optionally display the figure
fig.show()

# %%
# Define the cases, units, and other parameters
case = "Case_1"
opt_gens = [0, 1, 2]  # Assuming each case corresponds to a different generator
price_column = "mcp"
save_path = "outputs"

k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
columns = [
    "Method 1 (aligned with [34] as in Fig.1a)",
    "Method 2 (proposed model as in Fig.1b)",
    "Method 3 (DRL)",
]
# Initialize a DataFrame to store all profits
all_profits = pd.DataFrame(columns=columns, index=opt_gens, data=0.0)

# Loop through each case and generator
for opt_gen in opt_gens:
    # load data
    path = f"outputs/{case}/gen_{opt_gen}"

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

    profits = pd.concat(
        [
            updated_profits_method_1[opt_gen],
            updated_profits_method_2[opt_gen],
            profits_method_3[opt_gen],
        ],
        axis=1,
    )
    profits.columns = columns

    profits = profits.apply(pd.to_numeric, errors="coerce")

    all_profits.loc[opt_gen] = profits.sum()

# rename index in all_profits to Unit 1, Unit 2, Unit 3
# all_profits.index = [f"Unit {i+1}" for i in all_profits.index]
# rename index to Unit
all_profits.index.name = "Unit"

all_profits = all_profits / 1e3
# also round to 0 decimal places
all_profits = all_profits.round(0)

# Assuming df is your DataFrame
all_profits.reset_index(inplace=True)  # Reset the index to make 'Unit' a regular column

all_profits["Unit"] = all_profits["Unit"].apply(lambda x: f"{x + 1}")

# Melting the DataFrame
df_long = all_profits.melt(id_vars="Unit", var_name="Method", value_name="Profit")

# %%

# Assuming df_long is already prepared
# Create the bar plot
fig = px.bar(
    df_long,
    x="Unit",
    y="Profit",
    color="Method",  # This ensures different colors for each method
    barmode="group",
    labels={"Profit": "Profit [tsnd. €]"},  # Note: Label here is slightly different
)

# Update layout details
fig.update_layout(
    xaxis_title="Unit",
    yaxis_title="Profit [k.€]",
    legend_title="Methods",
    legend=dict(
        title="Method",
        orientation="v",
        yanchor="top",
        y=0.95,  # Adjust this value to move the legend up or down
        xanchor="right",
        x=1,  # Adjust this value to move the legend left or right
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(family="Arial", size=font_size, color="black"),
    template="plotly_white",
)

# Display integer values on top of bars
fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")

# Set y-axis range and title
fig.update_yaxes(range=[0, 550])

# set legend font size to 14
fig.update_layout(legend=dict(title_font_size=20, font=dict(size=20)))

# Save plot as PDF
fig.write_image(f"outputs/{case}/all_profits.pdf")

# Show the plot
fig.show()


# %%
