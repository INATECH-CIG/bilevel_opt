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
    K=10,
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

    max_average_profits = -1e6
    iterations_without_change = 0
    i = 1
    while True:
        print()
        print(f"Iteration {i}")
        last_k_values = k_values_df.copy()
        last_profit_values = profit_values.copy()

        # iterate over units in reverse order
        for opt_gen in gens_df.index:
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
                    K=10,
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
        diff_in_profit /= 1000

        print("Difference in profits:")
        print(diff_in_profit)

        if (abs(diff_in_k).max() < 0.01).all():
            print(f"Actions did not change. Convergence reached at iteration {i}")
            break

        if (abs(diff_in_profit) < 1).all():
            print(f"Profits did not change. Convergence reached at iteration {i}")
            break

        average_profits = profit_values.mean(axis=1).mean()
        print("Average profit", average_profits)

        if average_profits > max_average_profits:
            print("New best solution found. Saving results...")
            max_average_profits = average_profits

            # save preliminary results
            save_results_path = f"outputs/{case}/{method}/preliminary"
            save_results(save_results_path, main_df, supp_df, k_values_df)
            iterations_without_change = 0
        else:
            iterations_without_change += 1

        if iterations_without_change > 10:
            print("No improvement in 10 iterations. Stopping.")
            break

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

    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    time_limit = 30  # time limit in seconds for each optimization

    start = pd.to_datetime("2019-03-02 00:00")
    end = pd.to_datetime("2019-03-03 00:00")

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
        )

# %%
