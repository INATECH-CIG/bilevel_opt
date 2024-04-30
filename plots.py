# %%
import os

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
    y='volume',
    labels={"volume": "Demand [MWh]"},
    # title='Demand Profile Over Time'
)

# Add capacity of each unit as horizontal line and accumulate the capacity
capacity = 0
for i, row in gens_df.iterrows():
    capacity += row['g_max']
    fig.add_hline(
        y=capacity, 
        line_dash="dash", 
        annotation_text=f"Unit {i+1} capacity: {row['g_max']} MW",
        annotation_position="top right"
    )

# Update y-axis range and title
fig.update_yaxes(range=[0, 4000], title_text="Demand [MW]")

# Update x-axis title
fig.update_xaxes(title_text="Time Step")

# Enhance the layout
fig.update_layout(
    template='plotly_white',  # Clean and professional template
    showlegend=False,
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins to fit the title
    font=dict(size=font_size, family='Arial', color='black') # Set global font size and family
)

# Save plot as PDF
fig.write_image(f"outputs/{case}/demand.pdf")

# Optionally display the figure
fig.show()

# %%
# plot marginal costs of each unit as bar plot and name it marginal cost
# Reshape the DataFrame to have separate rows for each unit's marginal cost and its multiplied value

df = pd.DataFrame(
    {
        "Unit": gens_df.index[:3]+1,
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
fig.update_traces(text=df["Total interval"], textposition='outside', selector=dict(name="Bidding interval"), textfont=dict(color='black'))

# Display the values from df["Marginal cost"] on top of the marginal cost bars
fig.update_traces(text=df["Marginal cost"], textposition='outside', selector=dict(name="Marginal cost"))

# Update layout for a cleaner look
fig.update_layout(
    legend_title_text=' ',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=font_size, family='Arial', color='black'),
    template='plotly_white'
)

# Set y-axis range and title
fig.update_yaxes(range=[0, 200], title_text="Marginal cost [€/MWh]")

# Set x-axis to only show integer labels
fig.update_xaxes(tickvals=df['Unit'])

# Save plot as PDF
fig.write_image(f"outputs/{case}/marginal_costs.pdf")

# Optionally display the figure
fig.show()
