# %%
import os

import pandas as pd
import plotly.express as px

from model_1 import find_optimal_k_method_1 as method_1
from model_2 import find_optimal_k_method_2 as method_2
from uc_problem import solve_uc_problem
from utils import calculate_profits

# %%
if __name__ == "__main__":
    case = "Case_1"
    method = "method_2"

    big_w = 10  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    time_limit = 300  # time limit in seconds for each optimization

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

    path = f"outputs/{case}/method_1"

    market_orders = pd.read_csv(
        f"{path}/market_orders.csv",
        index_col=0,
        parse_dates=True,
    )
    k_values_df = pd.DataFrame(index=demand_df.index, columns=gens_df.index, data=0.0)
    for opt_gen in gens_df.index:
        rl_unit_orders = market_orders[market_orders["unit_id"] == f"Unit_{opt_gen}"]
        rl_unit_orders = rl_unit_orders.loc[start:end]
        rl_unit_orders = rl_unit_orders.reset_index(drop=False)
        marginal_cost = gens_df.at[opt_gen, "mc"]
        k_values_df[opt_gen] = rl_unit_orders["price"] / marginal_cost

    main_df_3, supp_df_3 = solve_uc_problem(gens_df, demand_df, k_values_df)
    profits_method_3 = calculate_profits(main_df_3, supp_df_3, gens_df)

    print_results = False

    if method == "method_1":
        find_optimal_k = method_1
    elif method == "method_2":
        find_optimal_k = method_2

    # %%
    new_k_values = k_values_df.copy()
    profits_method_1 = pd.DataFrame(index=demand_df.index, columns=gens_df.index, data=0.0)

    print(f"Solving using {method}")
    for opt_gen in gens_df.index:
        print(f"Optimizing for Unit {opt_gen+1}")
        main_df, supp_df, k_values = find_optimal_k(
            gens_df=gens_df,
            k_values_df=k_values_df,
            demand_df=demand_df,
            k_max=k_max,
            opt_gen=opt_gen,
            big_w=big_w,
            time_limit=time_limit,
            print_results=print_results,
        )

        new_k_values.loc[:, opt_gen] = k_values["k"]
        temp_k_values = k_values_df.copy()
        temp_k_values[opt_gen] = k_values["k"]

        main_df, supp_df = solve_uc_problem(gens_df, demand_df, temp_k_values)
        temp_profits = calculate_profits(main_df, supp_df, gens_df)

        profits_method_1.loc[:, opt_gen] = temp_profits[opt_gen]

    # %%
    total_profits_method_1 = pd.DataFrame(
        index=profits_method_1.columns,
        columns=["Method 1"],
        data=profits_method_1.sum(),
    ).astype(float)

    total_profits_method_3 = pd.DataFrame(
        index=total_profits_method_1.index,
        columns=["DRL"],
        data=profits_method_3.sum(),
    ).astype(float)

    all_profits = pd.concat([total_profits_method_1, total_profits_method_3], axis=1)
    all_profits /= 1000
    all_profits = all_profits.round()

    # drop index 3
    all_profits = all_profits.drop(3)
    # plot the profits as bars
    fig = px.bar(
        all_profits,
        x=all_profits.index+1,
        y=all_profits.columns,
        title="Total profit per unit",
        labels={"index": "Unit", "Profit": "Profit [k€]"},
        barmode="group",
    )

    # display values on top
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
    fig.update_yaxes(title_text="Profit [k€]")
    # save figure as html
    # fig.write_html("outputs/total_profits.html")
    # and as pdf
    # fig.write_image("outputs/total_profits.pdf")

    fig.show()

    # %%
    # compare k_values_df and k_values_new for opt_gen
    # print("Previous k_values")
    # print(k_values_df)
    # print("\nNew k_values")
    # print(new_k_values)

    # plot both k_values_df for each unit
    fig = px.line(
        k_values_df,
        title="Previous k values",
        labels={"index": "Time", "value": "k"},
    )

    # rename lines to Previous k values for Unit x
    for opt_gen in gens_df.index:
        fig.data[opt_gen].name = f"Prev. k: Unit {opt_gen+1}"

    # also plot new k_values_df for each unit
    for opt_gen in gens_df.index:
        fig.add_scatter(
            x=new_k_values.index,
            y=new_k_values[opt_gen],
            name=f"New k: Unit {opt_gen+1}",
        )

    # make the plot bigger
    fig.update_layout(height=400, width=600)

    fig.show()
    # %%

    diff = k_values_df - new_k_values
    diff = diff.abs()
    print("\nDifference")
    print(diff)


# %%
