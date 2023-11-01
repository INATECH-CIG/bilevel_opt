# %%
import os

import pandas as pd
import plotly.express as px

# from model_1 import find_optimal_k_method_1 as method_1
# from model_2 import find_optimal_k_method_2 as method_2
from model_1 import find_optimal_k_method_1 as method_1
from model_2_discrete import find_optimal_k_method_2 as method_2
from uc_problem import solve_uc_problem
from utils import calculate_profits, calculate_uplift

# %%
if __name__ == "__main__":
    case = "Case_1"
    opt_gen = 0  # generator that is allowed to bid strategically

    big_w = 1000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    time_limit = 60  # time limit in seconds for each optimization

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

    print_results = False
    optimize = True

    # %%
    if optimize:
        print("Solving using Method 1")
        main_df_1, supp_df_1, k_values_1 = method_1(
            gens_df=gens_df,
            k_values_df=k_values_df,
            demand_df=demand_df,
            k_max=k_max,
            opt_gen=opt_gen,
            big_w=big_w,
            time_limit=time_limit,
            print_results=print_results,
        )

        k_values_df_1 = k_values_df.copy()
        k_values_df_1[opt_gen] = k_values_1

        # print("Method 1 results")
        # print(main_df_1)

        updated_main_df_1, updated_supp_df_1 = solve_uc_problem(
            gens_df, demand_df, k_values_df_1
        )

        prices = pd.concat([main_df_1["mcp"], updated_main_df_1["mcp"]], axis=1)

        print("Solving using Method 2")
        main_df_2, supp_df_2, k_values_2 = method_2(
            gens_df=gens_df,
            k_values_df=k_values_df,
            demand_df=demand_df,
            k_max=k_max,
            opt_gen=opt_gen,
            big_w=big_w,
            time_limit=time_limit,
            print_results=print_results,
        )

        # %%

        k_values_df_2 = k_values_df.copy()
        k_values_df_2[opt_gen] = k_values_2

        # print("Method 2 results")
        # print(main_df_2)

        updated_main_df_2, updated_supp_df_2 = solve_uc_problem(
            gens_df, demand_df, k_values_df_2
        )

        # merge two k values dataframes
        k_values = pd.concat([k_values_1, k_values_2], axis=1)
        k_values.columns = ["Method 1", "Method 2"]
        # print("Merged k values")
        # print(k_values)

        prices = pd.concat([main_df_2["mcp_hat"], updated_main_df_2["mcp"]], axis=1)
        power = pd.concat(
            [main_df_2[f"gen_{opt_gen}"], updated_main_df_2[f"gen_{opt_gen}"]], axis=1
        )

        # save all results to csv
        save_path = f"outputs/{case}/gen_{opt_gen}"
        # check if path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        k_values_1.to_csv(f"{save_path}/k_values_1.csv")
        k_values_2.to_csv(f"{save_path}/k_values_2.csv")

        main_df_1.to_csv(f"{save_path}/main_df_1.csv")
        supp_df_1.to_csv(f"{save_path}/supp_df_1.csv")

        updated_main_df_1.to_csv(f"{save_path}/updated_main_df_1.csv")
        updated_supp_df_1.to_csv(f"{save_path}/updated_supp_df_1.csv")

        main_df_2.to_csv(f"{save_path}/main_df_2.csv")
        supp_df_2.to_csv(f"{save_path}/supp_df_2.csv")

        updated_main_df_2.to_csv(f"{save_path}/updated_main_df_2.csv")
        updated_supp_df_2.to_csv(f"{save_path}/updated_supp_df_2.csv")

        print("Finished solving. All results saved to csv.")

    # %%
    # load data
    path = f"outputs/{case}/gen_{opt_gen}"

    k_values_1 = pd.read_csv(f"{path}/k_values_1.csv", index_col=0)
    k_values_2 = pd.read_csv(f"{path}/k_values_2.csv", index_col=0)

    main_df_1 = pd.read_csv(f"{path}/main_df_1.csv", index_col=0)
    supp_df_1 = pd.read_csv(f"{path}/supp_df_1.csv", index_col=0)

    updated_main_df_1 = pd.read_csv(f"{path}/updated_main_df_1.csv", index_col=0)
    updated_supp_df_1 = pd.read_csv(f"{path}/updated_supp_df_1.csv", index_col=0)

    main_df_2 = pd.read_csv(f"{path}/main_df_2.csv", index_col=0)
    supp_df_2 = pd.read_csv(f"{path}/supp_df_2.csv", index_col=0)

    updated_main_df_2 = pd.read_csv(f"{path}/updated_main_df_2.csv", index_col=0)
    updated_supp_df_2 = pd.read_csv(f"{path}/updated_supp_df_2.csv", index_col=0)

    profits_method_1 = calculate_profits(main_df_1, supp_df_1, gens_df)
    updated_profits_method_1 = calculate_profits(
        updated_main_df_1, updated_supp_df_1, gens_df, price_column="mcp"
    )
    profits_method_2 = calculate_profits(
        main_df_2, supp_df_2, gens_df, price_column="mcp_hat"
    )
    updated_profits_method_2 = calculate_profits(
        updated_main_df_2, updated_supp_df_2, gens_df, price_column="mcp"
    )

    # calculate uplifts
    uplift_method_2, uplift_df_method_2 = calculate_uplift(
        main_df=updated_main_df_2,
        gens_df=gens_df,
        gen_unit=opt_gen,
        profits=updated_profits_method_2[opt_gen].sum(),
    )

    total_profit_with_uplift_method_2 = (
        updated_profits_method_2[opt_gen].sum() + uplift_method_2
    )

    # %% RL Part
    market_orders = pd.read_csv(
        f"{path}/market_orders.csv",
        index_col=0,
        parse_dates=True,
    )
    unit_id = f"Unit_{opt_gen}"
    rl_unit_orders = market_orders[market_orders["unit_id"] == unit_id]
    rl_unit_orders = rl_unit_orders.loc[start:end]
    rl_unit_orders = rl_unit_orders.reset_index(drop=False)

    marginal_cost = gens_df.at[opt_gen, "mc"]
    rl_unit_profits = pd.Series(index=rl_unit_orders.index)
    rl_unit_profits = rl_unit_orders["accepted_volume"] * (
        rl_unit_orders["accepted_price"] - marginal_cost
    )

    # iterate over all rows and subtract start up and shut down costs if the unit turned on or off
    for t in range(1, len(rl_unit_orders)):
        if t == 1:
            if (
                rl_unit_orders.at[t, "accepted_volume"] > 0
                and gens_df.at[opt_gen, "u_0"] == 0
            ):
                rl_unit_profits[t] -= gens_df.at[opt_gen, "k_up"]
            elif (
                rl_unit_orders.at[t, "accepted_volume"] == 0
                and gens_df.at[opt_gen, "u_0"] > 0
            ):
                rl_unit_profits[t] -= gens_df.at[opt_gen, "k_down"]
        elif (
            rl_unit_orders.at[t, "accepted_volume"] == 0
            and rl_unit_orders.at[t - 1, "accepted_volume"] > 0
        ):
            rl_unit_profits[t] -= gens_df.at[opt_gen, "k_down"]
        elif (
            rl_unit_orders.at[t, "accepted_volume"] > 0
            and rl_unit_orders.at[t - 1, "accepted_volume"] == 0
        ):
            rl_unit_profits[t] -= gens_df.at[opt_gen, "k_up"]

    uplift_method_rl, uplift_df_method_rl = calculate_uplift(
        main_df=rl_unit_orders,
        gens_df=gens_df,
        gen_unit=opt_gen,
        price_column="accepted_price",
        profits=rl_unit_profits.sum(),
    )

    total_profit_with_uplift_method_rl = rl_unit_profits.sum() + uplift_method_rl

    # %%
    # plot sum of both profits as bar chart
    profits = pd.concat(
        [
            profits_method_1[opt_gen],
            updated_profits_method_1[opt_gen],
            profits_method_2[opt_gen],
            updated_profits_method_2[opt_gen],
            rl_unit_profits,
        ],
        axis=1,
    )
    profits.columns = [
        "Method 1",
        "Method 1 (after UC)",
        "Method 2",
        "Method 2 (after UC)",
        "Method 3 (RL)",
    ]

    profits = profits.apply(pd.to_numeric, errors="coerce")
    fig = px.bar(
        title=f"Total profits of Unit {opt_gen+1}",
        labels={"index": "Method", "Profit": "Profit [€]"},
    )

    # add Method 1 bar
    fig.add_bar(
        x=["Method 1"],
        y=[profits["Method 1"].sum()],
        name="Method 1",
    )

    # add Method 1 (after UC) bar
    fig.add_bar(
        x=["Method 1 (after UC)"],
        y=[profits["Method 1 (after UC)"].sum()],
        name="Method 1 (after UC)",
    )

    # add Method 2 bar
    fig.add_bar(
        x=["Method 2"],
        y=[profits["Method 2"].sum()],
        name="Method 2",
    )

    # add Method 2 (after UC) bar
    fig.add_bar(
        x=["Method 2 (after UC)"],
        y=[profits["Method 2 (after UC)"].sum()],
        name="Method 2 (after UC)",
    )

    # add an extra bar of the total profit with uplift
    # use the same color as the original bar
    fig.add_bar(
        x=["Method 2 (with uplift)"],
        y=[total_profit_with_uplift_method_2],
        name="Method 2 (after UC) with uplift",
    )

    # add Method 3 (RL) bar
    fig.add_bar(
        x=["Method 3 (RL)"],
        y=[profits["Method 3 (RL)"].sum()],
        name="Method 3 (RL)",
    )

    # add rl with uplift
    fig.add_bar(
        x=["Method 3 (RL with uplift)"],
        y=[total_profit_with_uplift_method_rl],
        name="Method 3 (RL) with uplift",
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

    fig.update_yaxes(title_text="Profit [€]")
    fig.update_layout(showlegend=False)

    # save plot as pdf
    # fig.write_image(f"outputs/{case}/profits_{opt_gen}.pdf")

    # save plot as html
    # fig.write_html(f"outputs/{case}/profits_{opt_gen}.html")
    fig.show()

    # %% Bids of the unit
    bids_method_1 = k_values_1 * gens_df.at[opt_gen, "mc"]
    bids_method_2 = k_values_2 * gens_df.at[opt_gen, "mc"]
    bids_method_3 = rl_unit_orders["price"]

    bids = pd.concat([bids_method_1, bids_method_2, bids_method_3], axis=1)
    bids.columns = ["Method 1", "Method 2", "Method 3 (RL)"]

    # convert all columns to numeric data types
    bids = bids.apply(pd.to_numeric, errors="coerce")

    # rename index to time
    bids.index.name = "Time"

    # plot bids over time
    fig = px.line(
        bids,
        title=f"Bids of Unit {opt_gen+1}",
        labels={"Time": "Time", "value": "Bid [€/MWh]"},
    )

    fig.update_yaxes(title_text="Bid [€/MWh]")
    fig.update_layout(showlegend=True)
    # save plot as html
    # fig.write_html(f"outputs/{case}/bids_{opt_gen}.html")
    # save plot as pdf
    # fig.write_image(f"outputs/{case}/bids_{opt_gen}.pdf")
    fig.show()

    # %% Dispatch of the unit
    dispatch_method_1 = updated_main_df_1[f"gen_{opt_gen}"]
    dispatch_method_2 = updated_main_df_2[f"gen_{opt_gen}"]
    dispatch_method_3 = rl_unit_orders["accepted_volume"]

    dispatch = pd.concat(
        [dispatch_method_1, dispatch_method_2, dispatch_method_3], axis=1
    )
    dispatch.columns = ["Method 1", "Method 2", "Method 3 (RL)"]

    # convert all columns to numeric data types
    dispatch = dispatch.apply(pd.to_numeric, errors="coerce")

    # rename index to time
    dispatch.index.name = "Time"

    # plot bids over time
    fig = px.line(
        dispatch,
        title=f"Dispatch of Unit {opt_gen+1}",
        labels={"Time": "Time", "value": "Dispatch [MW]"},
    )

    fig.update_yaxes(title_text="Dispatch [MW]")
    fig.update_layout(showlegend=True)

    # save plot as html
    # fig.write_html(f"outputs/{case}/dispatch_{opt_gen}.html")
    # save plot as pdf
    # fig.write_image(f"outputs/{case}/dispatch_{opt_gen}.pdf")
    fig.show()

    # %% Market clearing price
    mcp_method_1 = main_df_1["mcp"]
    mcp_method_2 = main_df_2["mcp_hat"]
    mcp_method_3 = market_orders[market_orders["unit_id"] == "demand_EOM"][
        "accepted_price"
    ]
    mcp_method_3 = mcp_method_3.loc[start:end]
    mcp_method_3 = mcp_method_3.reset_index(drop=True)

    mcp = pd.concat([mcp_method_1, mcp_method_2, mcp_method_3], axis=1)
    mcp.columns = ["Method 1", "Method 2", "Method 3 (RL)"]

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

    # %%
    # plot demand_df
    fig = px.line(
        demand_df["volume"],
        title="Demand",
        labels={"Time": "Time", "value": "Demand [MW]"},
    )
    # rename the line into Demand [MWh]
    fig.data[0].name = "Demand [MW]"

    # add capacity of each unit as horizontal line
    # accumulate the capacity
    # and name the horizontal line after the unit
    capacity = 0
    for i in range(len(gens_df)):
        capacity += gens_df.at[i, "g_max"]
        fig.add_hline(y=capacity, line_dash="dash", annotation_text=f"Unit {i+1}")

    # y axis from 0 to 3500
    fig.update_yaxes(range=[0, 4000])
    fig.update_yaxes(title_text="Demand [MW]")
    # rename x axis to Time step
    fig.update_xaxes(title_text="Time step")
    fig.update_layout(showlegend=False)
    # save plot as html
    # fig.write_html(f"outputs/{case}/demand.html")
    # save plot as pdf
    # fig.write_image(f"outputs/{case}/demand.pdf")
    fig.show()

    # %%
    # plot marginal costs of each unit as bar plot and name it marginal cost
    # Reshape the DataFrame to have separate rows for each unit's marginal cost and its multiplied value

    # Reshape the DataFrame to have separate rows for each unit's marginal cost and its multiplied value
    df = pd.DataFrame(
        {
            "Unit": gens_df.index,
            "Marginal cost": gens_df["mc"],
            "Bidding interval": gens_df["mc"] * 2,
        }
    )

    # Create the figure object
    fig = px.bar(
        df,
        x="Unit",
        y=["Marginal cost", "Bidding interval"],
        barmode="group",  # Set the barmode to "group" for side-by-side bars
        title="Marginal costs",
        labels={"x": "Unit", "y": "Marginal cost [€/MWh]"},
    )

    # rename y axis to Marginal cost [€/MWh]
    fig.update_yaxes(title_text="Marginal cost [€/MWh]")

    # Set the opacity of the second set of bars to make them semitransparent
    fig.update_traces(opacity=0.6, selector=dict(name="Bidding interval"))

    # display values on top of bars
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")

    # display the legend below the plot
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )

    # y axis form 0 to 200
    fig.update_yaxes(range=[0, 200])

    # save plot as html
    fig.write_html(f"outputs/{case}/marginal_costs.html")
    # save plot as pdf
    fig.write_image(f"outputs/{case}/marginal_costs.pdf")

    fig.show()

    # %%
    # plot mcp and mcp_hat from main_df_2
    fig = px.line(
        main_df_2[["mcp", "mcp_hat"]],
        title="Market Clearing Price",
        labels={"Time": "Time", "value": "MCP [€/MWh]"},
    )

    # rename the lines into MCP, MCP_hat
    fig.data[0].name = "MCP"
    fig.data[1].name = "MCP_hat"

    # also add the price from updated_main_df_2
    fig.add_scatter(
        x=updated_main_df_2.index,
        y=updated_main_df_2["mcp"],
        name="MCP after UC",
    )
    fig.update_xaxes(title_text="Time step")
    # save plot as html
    fig.write_html(f"outputs/{case}/mcp.html")
    # save plot as pdf
    fig.write_image(f"outputs/{case}/mcp.pdf")
    fig.show()

    # %%
    # plot profits_method_2 and updated_profits_method_2 for opt_gen
    profits = pd.concat(
        [
            profits_method_1[opt_gen],
            updated_profits_method_1[opt_gen],
            profits_method_2[opt_gen],
            updated_profits_method_2[opt_gen],
        ],
        axis=1,
    )
    profits.columns = [
        "Method 1",
        "Method 1 (after UC)",
        "Method 2",
        "Method 2 (after UC)",
    ]

    profits = profits.apply(pd.to_numeric, errors="coerce")
    fig = px.bar(
        title=f"Total profits of Unit {opt_gen+1}",
        labels={"index": "Method", "Profit": "Profit [€]"},
    )

    # add Method 1 bar
    fig.add_bar(
        x=["Method 1"],
        y=[profits["Method 1"].sum()],
        name="Method 1",
    )

    # add Method 1 (after UC) bar
    fig.add_bar(
        x=["Method 1 (after UC)"],
        y=[profits["Method 1 (after UC)"].sum()],
        name="Method 1 (after UC)",
    )

    # add Method 2 bar
    fig.add_bar(
        x=["Method 2"],
        y=[profits["Method 2"].sum()],
        name="Method 2",
    )

    # add Method 2 (after UC) bar
    fig.add_bar(
        x=["Method 2 (after UC)"],
        y=[profits["Method 2 (after UC)"].sum()],
        name="Method 2 (after UC)",
    )

    # display values on top of bars
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")

    # disable legend
    fig.update_layout(showlegend=False)

    # save plot as pdf
    fig.write_image(f"outputs/{case}/profits.pdf")

    fig.show()


# %%
