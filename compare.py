# %%
import pandas as pd
import plotly.express as px

from model_1 import find_optimal_k_method_1 as method_1
from model_2 import find_optimal_k_method_2 as method_2
from uc_problem import uc_problem
from utils import calculate_profits

# %%
if __name__ == "__main__":
    case = "Case_1"

    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    time_limit = 30  # time limit in seconds for each optimization

    start = pd.to_datetime("2019-03-03 00:00")
    end = pd.to_datetime("2019-03-04 00:00")

    # gens
    gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
    opt_gen = 2  # generator that is allowed to bid strategically

    print_results = False

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

    updated_main_df_1, updated_supp_df_1 = uc_problem(gens_df, demand_df, k_values_df_1)

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

    k_values_df_2 = k_values_df.copy()
    k_values_df_2[opt_gen] = k_values_2

    # print("Method 2 results")
    # print(main_df_2)

    updated_main_df_2, updated_supp_df_2 = uc_problem(gens_df, demand_df, k_values_df_2)

    # merge two k values dataframes
    k_values = pd.concat([k_values_1, k_values_2], axis=1)
    k_values.columns = ["Method 1", "Method 2"]
    # print("Merged k values")
    # print(k_values)

    #

    prices = pd.concat([main_df_2["mcp_hat"], updated_main_df_2["price"]], axis=1)
    power = pd.concat(
        [main_df_2[f"gen_{opt_gen}"], updated_main_df_2[f"gen_{opt_gen}"]], axis=1
    )

    # %%
    profits_method_1 = calculate_profits(main_df_1, supp_df_1, gens_df)
    updated_profits_method_1 = calculate_profits(
        updated_main_df_1, updated_supp_df_1, gens_df, price_column="price"
    )
    profits_method_2 = calculate_profits(
        main_df_2, supp_df_2, gens_df, price_column="mcp_hat"
    )
    updated_profits_method_2 = calculate_profits(
        updated_main_df_2, updated_supp_df_2, gens_df, price_column="price"
    )

    # plot sum of both profits as bar chart
    profits = pd.concat(
        [
            profits_method_1.sum(axis=0),
            updated_profits_method_1.sum(axis=0),
            profits_method_2.sum(axis=0),
            updated_profits_method_2.sum(axis=0),
        ],
        axis=1,
    )
    profits.columns = [
        "Method 1",
        "Method 1 (updated)",
        "Method 2",
        "Method 2 (updated)",
    ]

    profits = profits.apply(pd.to_numeric, errors="coerce")
    fig = px.bar(
        profits.sum(axis=0),
        title="Total profits",
        labels={"index": "Method", "Profit": "Profit [k€]"},
    )
    fig.update_yaxes(title_text="Profit [k€]")
    fig.update_layout(showlegend=False)
    fig.show()

# %%
