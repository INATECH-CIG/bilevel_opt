# %%
import os

import pandas as pd
import plotly.express as px

from model_1 import find_optimal_k_method_1 as method_1
from model_2 import find_optimal_k_method_2 as method_2
from uc_problem import solve_uc_problem
from utils import calculate_profits, calculate_mc

if __name__ == "__main__":
    case = "Case_2"
    opt_gen = 35  # generator that is allowed to bid strategically

    k_max = 2  # maximum multiplier for strategic bidding
    time_limit = 600*2  # time limit in seconds for each optimization
    K = 10

    start = pd.to_datetime("2021-03-03 00:00")
    end = pd.to_datetime("2021-03-03 12:00")

    # gens
    gens_df = pd.read_csv(f"inputs/{case}/gens.csv")
    emission_factors = pd.read_csv(f"inputs/{case}/EmissionFactors.csv", index_col=0)
    gens_df["ef"] = gens_df["fuel"].map(emission_factors["emissions"])

    fuel_prices = pd.read_csv(f"inputs/{case}/fuel_prices.csv", index_col=0)

    gens_df["mc"] = gens_df.apply(calculate_mc, axis=1, fuel_prices=fuel_prices).round(
        2
    )

    gens_df["k_up"] *= (gens_df["g_max"] * 2 / 3).round(2)
    gens_df["k_down"] *= (gens_df["g_max"] * 1 / 3).round(2)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df["price"] = 500
    # resaple to 1 hour
    demand_df = demand_df.resample("1h").mean()
    demand_df = demand_df.loc[start:end]

    vre_gen = pd.read_csv(f"inputs/{case}/vre_gen.csv", index_col=0)
    vre_gen.index = pd.to_datetime(vre_gen.index)
    # resaple to 1 hour
    vre_gen = vre_gen.resample("1h").mean()
    vre_gen = vre_gen.loc[start:end]

    # subtract vre generation from demand
    demand_df["volume"] -= vre_gen.sum(axis=1)

    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    # active power plant
    active_pp = gens_df["g_max"].cumsum() <= demand_df["volume"].iloc[0]

    # set g_0 as g_max for the first 60 power plants
    gens_df["g_0"] = 0
    gens_df.loc[active_pp, "g_0"] = gens_df.loc[active_pp, "g_max"]

    # set u_0 as 1 for the first 60 power plants
    gens_df["u_0"] = 0
    gens_df.loc[active_pp, "u_0"] = 1

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)

    print_results = True
    optimize = True

    save_path = f"outputs/{case}/gen_{opt_gen}"
    # check if path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    big_w_values = {
        0: {"model_1": 100, "model_2": 100},
        opt_gen: {"model_1": 100, "model_2": 1000},
        25: {"model_1": 1000, "model_2": 1000},
    }

    # default_big_w_values = 1000

    # %%
    if optimize:
        print("Solving using Method 1")
        main_df_1, supp_df_1, k_values_1 = method_1(
            gens_df=gens_df,
            k_values_df=k_values_df,
            demand_df=demand_df,
            k_max=k_max,
            opt_gen=opt_gen,
            big_w=big_w_values[opt_gen]["model_1"],
            time_limit=time_limit,
            print_results=print_results,
            K=K,
        )

        k_values_df_1 = k_values_df.copy()
        k_values_df_1[opt_gen] = k_values_1

        updated_main_df_1, updated_supp_df_1 = solve_uc_problem(
            gens_df, demand_df, k_values_df_1
        )
        k_values_1.to_csv(f"{save_path}/k_values_1.csv")
        updated_main_df_1.to_csv(f"{save_path}/updated_main_df_1.csv")
        updated_supp_df_1.to_csv(f"{save_path}/updated_supp_df_1.csv")

    # %%
    if optimize:
        print("Solving using Method 2")
        main_df_2, supp_df_2, k_values_2 = method_2(
            gens_df=gens_df,
            k_values_df=k_values_df,
            demand_df=demand_df,
            k_max=k_max,
            opt_gen=opt_gen,
            big_w=big_w_values[opt_gen]["model_2"],
            time_limit=time_limit,
            print_results=print_results,
            K=K,
        )

        k_values_df_2 = k_values_df.copy()
        k_values_df_2[opt_gen] = k_values_2

        updated_main_df_2, updated_supp_df_2 = solve_uc_problem(
            gens_df, demand_df, k_values_df_2
        )
        # save all results to csv
        k_values_2.to_csv(f"{save_path}/k_values_2.csv")
        updated_main_df_2.to_csv(f"{save_path}/updated_main_df_2.csv")
        updated_supp_df_2.to_csv(f"{save_path}/updated_supp_df_2.csv")

        print("Finished solving. All results saved to csv.")

    # %%
    # load data for Method 1
    path = f"outputs/{case}/gen_{opt_gen}"

    k_values_1 = pd.read_csv(f"{path}/k_values_1.csv", index_col=0)

    updated_main_df_1 = pd.read_csv(f"{path}/updated_main_df_1.csv", index_col=0)
    updated_supp_df_1 = pd.read_csv(f"{path}/updated_supp_df_1.csv", index_col=0)
    
    updated_profits_method_1 = calculate_profits(
        updated_main_df_1, updated_supp_df_1, gens_df, price_column="mcp"
    )
    # %%
    # load data for Method 2
    k_values_2 = pd.read_csv(f"{path}/k_values_2.csv", index_col=0)

    updated_main_df_2 = pd.read_csv(f"{path}/updated_main_df_2.csv", index_col=0)
    updated_supp_df_2 = pd.read_csv(f"{path}/updated_supp_df_2.csv", index_col=0)

    updated_profits_method_2 = calculate_profits(
        updated_main_df_2, updated_supp_df_2, gens_df, price_column="mcp"
    )

    # %% load data for Method 3 (RL)
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
            # profits_method_3[opt_gen],
        ],
        axis=1,
    )
    profits.columns = [
        "Method 1 (after UC)",
        "Method 2 (after UC)",
        # "Method 3 (DRL)",
    ]

    profits = profits / 1e3

    profits = profits.apply(pd.to_numeric, errors="coerce")
    fig = px.bar(
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
    # fig.add_bar(
    #     x=["Method 3 (RL)"],
    #     y=[profits["Method 3 (DRL)"].sum()],
    #     name="Method 3 (DRL)",
    # )

    # make all bares with Method 1 in name blue
    for i in range(len(fig.data)):
        if "Method 1" in fig.data[i].name:
            fig.data[i].marker.color = "blue"

    # make all bares with Method 2 in name orange
    for i in range(len(fig.data)):
        if "Method 2" in fig.data[i].name:
            fig.data[i].marker.color = "orange"

    # make all bares with Method 3 in name green
    # for i in range(len(fig.data)):
    #     if "Method 3" in fig.data[i].name:
    #         fig.data[i].marker.color = "green"

    # display values on top of bars
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")

    # adjust y axis range to fit all bars and text above them
    # fig.update_yaxes(range=[0, 0.7e6])

    fig.update_yaxes(title_text="Profit [tsnd.€]")
    fig.update_layout(showlegend=False)

    # save plot as pdf
    # fig.write_image(f"{save_path}/profits_{opt_gen}.pdf")

    # save plot as html
    # fig.write_html(f"outputs/{case}/profits_{opt_gen}.html")
    fig.show()

    # %% Bids of the unit
    bids_method_1 = k_values_1 * gens_df.at[opt_gen, "mc"]
    bids_method_2 = k_values_2 * gens_df.at[opt_gen, "mc"]
    bids_method_3 = k_values_df_3[opt_gen] * gens_df.at[opt_gen, "mc"]

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
    fig.write_image(f"outputs/{case}/bids_{opt_gen}.pdf")
    fig.show()

    # %% Dispatch of the unit
    dispatch_method_1 = updated_main_df_1[f"gen_{opt_gen}"]
    dispatch_method_2 = updated_main_df_2[f"gen_{opt_gen}"]
    dispatch_method_3 = main_df_3[f"gen_{opt_gen}"]

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
    fig.write_image(f"outputs/{case}/dispatch_{opt_gen}.pdf")
    fig.show()

    # %% Market clearing price
    mcp_method_1 = updated_main_df_1["mcp"]
    mcp_method_2 = updated_main_df_2["mcp"]
    mcp_method_3 = main_df_3["mcp"]

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
    # fig.write_html(f"outputs/{case}/mcp.html")
    # save plot as pdf
    fig.write_image(f"outputs/{case}/mcp.pdf")
    fig.show()

    # %%
    # plot merit order of all units with cumsum of g_max on x axis and mc on y axis
    # Sort the DataFrame by marginal cost
    sorted_gens_df = gens_df.sort_values("mc")

    # Create a line plot
    fig = px.line(
        x=sorted_gens_df["g_max"].cumsum(),
        y=sorted_gens_df["mc"],
        title="Merit Order",
        labels={"x": "Cumulative Capacity [MW]", "y": "Marginal Cost [€/MWh]"},
    )

    # Show the plot
    fig.show()

# %%
