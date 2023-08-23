"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

# %%
# Imports
import pandas as pd

from opt import find_optimal_k

# %%
if __name__ == "__main__":
    # generators
    gens_df = pd.read_csv("inputs/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv("inputs/demand.csv", index_col=0)

    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding
    opt_gen = 2  # generator that is allowed to bid strategically

    i = 1
    while True:
        print()
        print(f"Iteration {i}")
        last_k_values = gens_df["k"]
        gens_df["bid"] = gens_df["k"] * gens_df["mc"]

        for opt_gen in gens_df.index:
            main_df, supp_df, k = find_optimal_k(
                gens_df=gens_df,
                demand_df=demand_df,
                k_max=k_max,
                opt_gen=opt_gen,
                big_w=big_w,
            )

            gens_df.at[opt_gen, "k"] = k

        diff_in_k = gens_df["k"] - last_k_values

        if (abs(diff_in_k) < 0.001).all():
            print(f"Convergence reached at iteration {i}")
            break

        i += 1

    print()
    print("Final results:")
    print(main_df)
    print()
    print("Final bidding decisions:")
    print(gens_df[['k', 'bid']])

# %%
