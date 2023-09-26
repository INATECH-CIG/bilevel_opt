import pandas as pd


def calculate_profits(main_df, supp_df, gens_df, price_column="mcp"):
    profits = pd.DataFrame(index=main_df.index, columns=gens_df.index)
    for gen in gens_df.index:
        profits[gen] = (
            main_df[f"gen_{gen}"] * (main_df[price_column] - gens_df.at[gen, "mc"])
            - supp_df[f"start_up_{gen}"]
            - supp_df[f"shut_down_{gen}"]
        )

    return profits
