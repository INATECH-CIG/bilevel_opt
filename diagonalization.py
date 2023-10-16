"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

import os

import matplotlib.pyplot as plt
import numpy as np

# %%
# Imports
import pandas as pd

from model_1 import find_optimal_k_method_1 as method_1
from model_2 import find_optimal_k_method_2 as method_2
from utils import calculate_profits


# %% load data and define parameters
def run_diagonalization(
    case,
    start,
    end,
    method,
    k_max=2,
    time_limit=30,
    big_w=10000,
    print_results=False,
):
    if method == 1:
        find_optimal_k = method_1
    elif method == 2:
        find_optimal_k = method_2

    # gens
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
                time_limit=time_limit,
                print_results=print_results,
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

    print("Final results:")
    print(main_df)
    print()
    print("Final bidding decisions:")
    print(k_values_df)

    save_results_path = f"outputs/{case}/method_{method}"
    # make sure output folder exists
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    main_df.to_csv(f"{save_results_path}/main_df.csv")
    supp_df.to_csv(f"{save_results_path}/supp_df.csv")
    k_values_df.to_csv(f"{save_results_path}/k_values_df.csv")


# %%
