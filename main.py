"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

# %%
# Imports
import pandas as pd

from opt import find_optimal_k
from uc_problem import solve_and_get_prices

# %%
if __name__ == "__main__":
    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding

    start = pd.to_datetime("2019-03-01 00:00")
    end = pd.to_datetime("2019-03-02 00:00")

    # generators
    gens_df = pd.read_csv("inputs/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv("inputs/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
    # %%
    i = 1
    while True:
        print()
        print(f"Iteration {i}")
        last_k_values = k_values_df.copy()

        for opt_gen in gens_df.index:
            main_df, supp_df, k = find_optimal_k(
                gens_df=gens_df,
                k_values_df=k_values_df,
                demand_df=demand_df,
                k_max=k_max,
                opt_gen=opt_gen,
                big_w=big_w,
            )

            k_values_df[opt_gen] = k

        diff_in_k = k_values_df - last_k_values

        print(abs(diff_in_k).max())

        if (abs(diff_in_k).max() < 0.01).all():
            print(f"Convergence reached at iteration {i}")
            break

        i += 1

    print()
    print("Final results:")
    print(main_df)
    print()
    print("Final bidding decisions:")
    print(k_values_df)

    main_df.to_csv("outputs/main_df.csv")
    supp_df.to_csv("outputs/supp_df.csv")
    k_values_df.to_csv("outputs/k_values_df.csv")

    # %%
    # get true prices and profiles
    true_main_df, true_supp_df = solve_and_get_prices(gens_df, demand_df, k_values_df)

    # %%
    # get potential profits as difference between prices and marginal costs multiplied by generation
    # and subtracting the startup and shutdown costs

    profits = pd.DataFrame(index=main_df.index, columns=gens_df.index)
    for gen in gens_df.index:
        profits[gen] = (
            main_df[f"gen_{gen}"] * (main_df['price'] - gens_df.at[gen, "mc"]
            - gens_df.loc[gen, "startup_cost"] * true_supp_df[f"startup_{gen}"]
            - gens_df.loc[gen, "shutdown_cost"] * true_supp_df[f"shutdown_{gen}"]
        ))
