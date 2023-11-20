"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""
# %%
import os

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
    K=3,
    time_limit=180,
    big_w=10,
    print_results=False,
):
    if method == "method_1":
        find_optimal_k = method_1
    elif method == "method_2":
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
    print(f"Starting diagonalization using {method}")

    i = 1
    lowest_diff_in_profits = profit_values.copy()
    lowest_diff_in_profits = lowest_diff_in_profits.sum(axis=0) + 10e6
    while True:
        print()
        print(f"Iteration {i}")
        last_k_values = k_values_df.copy()
        last_profit_values = profit_values.copy()

        # iterate over units in reverse order
        for opt_gen in gens_df.index:
            if opt_gen == 3:
                continue
            print(f"Optimizing for Unit {opt_gen+1}")
            try:
                main_df, supp_df, k = find_optimal_k(
                    gens_df=gens_df,
                    k_values_df=k_values_df,
                    demand_df=demand_df,
                    k_max=k_max,
                    opt_gen=opt_gen,
                    big_w=big_w,
                    time_limit=time_limit,
                    print_results=print_results,
                    K=K,
                )
            except Exception as e:
                print(f"Error: {e}")
                print(f"Optimization for Unit {opt_gen+1} failed. Continuing...")
                continue

            k_values_df[opt_gen] = k
            profit_values[opt_gen] = calculate_profits(main_df, supp_df, gens_df)[
                opt_gen
            ]

        diff_in_k = k_values_df - last_k_values
        diff_in_profit = profit_values.sum(axis=0) - last_profit_values.sum(axis=0)

        print("Difference in profits:")
        print(diff_in_profit)

        if (abs(diff_in_k).max() < 0.01).all():
            print(f"Actions did not change. Convergence reached at iteration {i}")
            break

        if (abs(diff_in_profit) < 3000).all():
            print(f"Profits did not change. Convergence reached at iteration {i}")
            break

        if (abs(diff_in_profit) <= 0.01*profit_values.sum(axis=0)).all():
            print(f"Profits change is below threshold. Convergence reached at iteration {i}")
            break

        if (diff_in_profit <= lowest_diff_in_profits).all():
            lowest_diff_in_profits = diff_in_profit
            save_results_path = f"outputs/{case}/{method}/temp"
            save_results(save_results_path, main_df, supp_df, k_values_df)

        i += 1

    print("Final results:")
    print(main_df)
    print()
    print("Final bidding decisions:")
    print(k_values_df)

    save_results_path = f"outputs/{case}/{method}"
    save_results(save_results_path, main_df, supp_df, k_values_df)


def save_results(save_results_path, main_df, supp_df, k_values_df):
    # make sure output folder exists
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    main_df.to_csv(f"{save_results_path}/main_df.csv")
    supp_df.to_csv(f"{save_results_path}/supp_df.csv")
    k_values_df.to_csv(f"{save_results_path}/k_values_df.csv")


# %% run diagonalization
if __name__ == "__main__":
    case = "Case_1"

    big_w = 10  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    time_limit = 1000  # time limit in seconds for each optimization
    K = 3

    start = pd.to_datetime("2019-03-02 06:00")
    end = pd.to_datetime("2019-03-02 14:00")

    print_results = False
    solve_diag = True

    method = "method_2"

    if solve_diag:
        run_diagonalization(
            case=case,
            start=start,
            end=end,
            method=method,
            k_max=k_max,
            time_limit=time_limit,
            big_w=big_w,
            print_results=print_results,
            K=K,
        )

# %%
