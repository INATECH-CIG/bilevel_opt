# %%
import pandas as pd
import plotly.express as px
import numpy as np

from model_1 import find_optimal_k_method_1 as method_1
from model_2 import find_optimal_k_method_2 as method_2
from uc_problem import solve_uc_problem
from utils import calculate_profits

# %%
if __name__ == "__main__":
    case = "Case_1"
    start = pd.to_datetime("2019-03-02 06:00")
    end = pd.to_datetime("2019-03-02 14:00")

    k_max = 2  # maximum multiplier for strategic bidding
    K = 10
    time_limit = 300  # time limit in seconds for each optimization

    # gens
    gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    # %%
    # RL Part
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

    total_profits_method_3 = pd.DataFrame(
        index=gens_df.index,
        columns=["DRL"],
        data=profits_method_3.sum(),
    ).astype(float)

    # %%
    total_profits = pd.DataFrame(
        index=gens_df.index,
        columns=["method_1", "method_2"],
        data=0.0,
    ).astype(float)

    big_w_values = {
        0: {"method_1": 1000, "method_2": 100},
        1: {"method_1": 10, "method_2": 1},
        2: {"method_1": 1000, "method_2": 100},
    }

    for method in ["method_1", "method_2"]:
        new_k_values = k_values_df.copy()
        profits = pd.DataFrame(index=demand_df.index, columns=gens_df.index, data=0.0)
        if method == "method_1":
            find_optimal_k = method_1

        elif method == "method_2":
            find_optimal_k = method_2

        print(f"Solving using {method}")
        for opt_gen in gens_df.index:
            if opt_gen == 3:
                continue
            print(f"Optimizing for Unit {opt_gen+1}")
            main_df, supp_df, k_values = find_optimal_k(
                gens_df=gens_df,
                k_values_df=k_values_df,
                demand_df=demand_df,
                k_max=k_max,
                opt_gen=opt_gen,
                big_w=big_w_values[opt_gen][method],
                time_limit=time_limit,
                print_results=False,
                K=K,
            )

            new_k_values.loc[:, opt_gen] = k_values["k"].values.astype(np.float64)
            temp_k_values = k_values_df.copy()
            temp_k_values[opt_gen] = k_values["k"]

            main_df, supp_df = solve_uc_problem(gens_df, demand_df, temp_k_values)
            temp_profits = calculate_profits(main_df, supp_df, gens_df)

            profits.loc[:, opt_gen] = temp_profits[opt_gen].values.astype(np.float64)

            total_profits[method] = profits.sum()

    # %%
    all_profits = pd.concat([total_profits, total_profits_method_3], axis=1)
    all_profits /= 1000
    all_profits = all_profits.round()

    # drop index 3
    all_profits = all_profits.drop(3)

    # rename index to i+1
    all_profits.index = all_profits.index + 1

    all_profits = all_profits.astype(float)

    # %%
    # plot the profits as bars
    fig = px.bar(
        all_profits,
        x=all_profits.index,
        y=all_profits.columns,
        labels={"index": "Unit", "Profit": "Profit [k€]"},
        barmode="group",
    )

    # display values on top
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
    fig.update_yaxes(title_text="Profit [tsnd. €]", range=[0, 1000])
    # Update layout details
    fig.update_layout(
        xaxis_title="Unit",
        yaxis_title="Profit [tsnd. €]",
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
        font=dict(family="Arial", size=22, color="black"),
        template="plotly_white",
    )
    # and as pdf
    fig.write_image("outputs/total_profits_eq_check.pdf")

    fig.show()

    # %%
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
