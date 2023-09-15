"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


from opt import find_optimal_k
from uc_problem import solve_and_get_prices
from utils import calculate_profits

# %% load data and define parameters
if __name__ == "__main__":
    solve_diagonalization = False
    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding

    case = "Case_2"
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

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
    profit_values = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=0.0)
    # %% solve diagonalization and save results
    if solve_diagonalization:
        i = 1
        while True:
            print()
            print(f"Iteration {i}")
            last_k_values = k_values_df.copy()
            last_profit_values = profit_values.copy()

            for opt_gen in gens_df.index:
                print(f"Optimizing for generator {opt_gen}")
                main_df, supp_df, k = find_optimal_k(
                    gens_df=gens_df,
                    k_values_df=k_values_df,
                    demand_df=demand_df,
                    k_max=k_max,
                    opt_gen=opt_gen,
                    big_w=big_w,
                )

                k_values_df[opt_gen] = k
                profit_values[opt_gen] = calculate_profits(main_df, supp_df, gens_df)[
                    opt_gen
                ]
                print()

            diff_in_k = k_values_df - last_k_values
            diff_in_profit = profit_values.sum(axis=0) - last_profit_values.sum(axis=0)
            diff_in_profit /= 1000

            print("Difference in k values:")
            print(abs(diff_in_k).max())

            print("Difference in profits:")
            print(diff_in_profit)

            if (abs(diff_in_k).max() < 0.01).all():
                print(f"Actions did not change. Convergence reached at iteration {i}")
                break

            if (abs(diff_in_profit) < 1).all():
                print(f"Profits did not change. Convergence reached at iteration {i}")
                break

            i += 1

        print()
        print("Final results:")
        print(main_df)
        print()
        print("Final bidding decisions:")
        print(k_values_df)

        # make sure output folder exists
        if not os.path.exists(f"outputs/{case}"):
            os.makedirs(f"outputs/{case}")

        main_df.to_csv(f"outputs/{case}/approx_main_df.csv")
        supp_df.to_csv(f"outputs/{case}/approx_supp_df.csv")
        k_values_df.to_csv(f"outputs/{case}/k_values_df.csv")

    # %% execute for a single agent
    opt_gen = 2
    print()
    last_k_values = k_values_df.copy()
    last_profit_values = profit_values.copy()

    print(f"Optimizing for generator {opt_gen}")
    main_df, supp_df, k = find_optimal_k(
        gens_df=gens_df,
        k_values_df=k_values_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
    )

    k_values_df[opt_gen] = k
    profit_values[opt_gen] = calculate_profits(main_df, supp_df, gens_df)[opt_gen]

    print()
    print("Final results:")
    print(main_df)
    print()
    print("Final bidding decisions:")
    print(k_values_df)

    # make sure output folder exists
    if not os.path.exists(f"outputs/{case}"):
        os.makedirs(f"outputs/{case}")

    main_df.to_csv(f"outputs/{case}/approx_main_df.csv")
    supp_df.to_csv(f"outputs/{case}/approx_supp_df.csv")
    k_values_df.to_csv(f"outputs/{case}/k_values_df.csv")

# %%
