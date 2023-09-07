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

    case = "Case_1"
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

            for opt_gen in gens_df:
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

    # %% load previously saved results
    approx_main_df = pd.read_csv(f"outputs/{case}/approx_main_df.csv", index_col=0)
    approx_supp_df = pd.read_csv(f"outputs/{case}/approx_supp_df.csv")
    k_values_after_convergence = pd.read_csv(
        f"outputs/{case}/k_values_df.csv", index_col=0
    )
    k_values_after_convergence.columns = k_values_after_convergence.columns.astype(int)

    # get true prices and profiles
    true_main_df, true_supp_df = solve_and_get_prices(
        gens_df, demand_df, k_values_after_convergence
    )
    # get potential profits as difference between prices and marginal costs multiplied by generation
    # and subtracting the startup and shutdown costs
    approx_profits = calculate_profits(approx_main_df, approx_supp_df, gens_df)
    true_profits = calculate_profits(true_main_df, true_supp_df, gens_df)

    # convert to t. EUR
    approx_profits /= 1000
    true_profits /= 1000

    # calculate difference between approximated and true profits
    profit_diff = approx_profits - true_profits

    # plot the sum of true and approximated profits in one plot as a bar plot for each generator
    fig, ax = plt.subplots()
    x = np.arange(1, len(gens_df.index) + 1)  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(
        x - width / 2, round(approx_profits.sum()), width, label="Estimated profits"
    )
    rects2 = ax.bar(
        x + width / 2, round(true_profits.sum()), width, label="True profits"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Profits in t. EUR")
    ax.set_title("Profits by generator")
    ax.set_xticks(x)
    ax.set_xticklabels(f"Gen {gen+1}" for gen in gens_df.index)
    ax.legend()

    # also print actual values above the bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 2)
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(f"outputs/{case}/profits.png")
    plt.show()

    # plot true and approximate prices
    fig, ax = plt.subplots()
    ax.plot(true_main_df["price"].iloc[1:], label="True prices")
    ax.plot(approx_main_df["price"].iloc[1:], label="Approximated prices")
    ax.set_title("Prices")
    ax.set_ylabel("Price in EUR/MWh")
    ax.set_xlabel("Time")
    ax.legend()
    plt.savefig(f"outputs/{case}/prices.png")
    plt.show()
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
