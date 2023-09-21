# %%
import pandas as pd

from model_1 import find_optimal_k_method_1 as method_1
from model_2 import find_optimal_k_method_2 as method_2

# %%
if __name__ == "__main__":
    case = "Case_2"

    big_w = 10000  # weight for duality gap objective
    k_max = 2  # maximum multiplier for strategic bidding

    start = pd.to_datetime("2019-03-02 00:00")
    end = pd.to_datetime("2019-03-03 00:00")

    # gens
    gens_df = pd.read_csv(f"inputs/{case}/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv(f"inputs/{case}/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.DataFrame(columns=gens_df.index, index=demand_df.index, data=1.0)
    opt_gen = 1  # generator that is allowed to bid strategically

    instance_1, main_df_1, k_values_1 = method_1(
        gens_df=gens_df,
        k_values_df=k_values_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
        print_results=True,
    )

    print("Method 1 results")
    print(main_df_1)

    instance_2, main_df_2, k_values_2 = method_2(
        gens_df=gens_df,
        k_values_df=k_values_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
        print_results=True,
    )

    print("Method 2 results")
    print(main_df_2)

    # merge two k values dataframes
    k_values = pd.concat([k_values_1, k_values_2], axis=1)
    k_values.columns = ["Method 1", "Method 2"]
    print("Merged k values")
    print(k_values)

# %%
