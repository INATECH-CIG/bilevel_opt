import pandas as pd


def calculate_profits(main_df, supp_df, gens_df):
    supp_df = supp_df.reset_index()
    supp_df.index = supp_df["generators"]
    profits = pd.DataFrame(index=main_df.index, columns=gens_df.index)
    for gen in gens_df.index:
        temp = supp_df.loc[gen, ["time", "start-up", "shut-down"]]
        temp.index = temp["time"]
        profits[gen] = (
            main_df[f"gen_{gen}"] * (main_df["price"] - gens_df.at[gen, "mc"])
            - temp["start-up"]
            - temp["shut-down"]
        )

    return profits
