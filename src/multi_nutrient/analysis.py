import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Goals:
# 1. Identify topk MAE examples for any trial
# 2. Compare topk MAE examples for any number of trials
# 3. Identify top disagreement answers for 2 trials






def filter_df(path, name):
    df = pd.read_json(path, lines=True)

    doc_df = pd.json_normalize(df["doc"])
    result_df = pd.json_normalize(df["result"])

    result_df = result_df.rename(columns = {"mae" : name + "_mae"})
    new = pd.concat([
        df[["doc_id", "pred", "resps"]].rename(columns={"pred" : name + "_pred", "resps" : name + "_resp"}),
        doc_df[["description", "country", "queries", "carb", "fat", "energy", "protein"]],
        result_df[[name + "_mae"]]
    ], axis=1)

    # new[name + "_mae_norm"] = new[name + "_mae"] / new[nutrient].mean()
    return new


def combine_dfs(dfs, doc_columns=["description", "country", "queries", "carb", "fat", "energy", "protein"]):
    base_df = dfs[0].copy()
    for df in dfs[1:]:
        df = df.drop(columns=doc_columns, errors="ignore")
        base_df = base_df.merge(df, on="doc_id", how="inner")
    
    return base_df

def get_topk_mae(df, name, k):
    return df.sort_values(by=name + "_mae", ascending=False).head(k)

def get_topk_disagreement(dfs, names, k):
    combined_df = combine_dfs(dfs)
    pred_cols = [name + "_pred" for name in names]
    combined_df["std"] = combined_df[pred_cols].std(axis=1)
    return combined_df.sort_values(by="std", ascending=False).head(k)


def bar_plot_errors(df, names):
    plt.figure(figsize=(12, 8))
    for name in names:
        errors = df[name + "_mae"]
        plt.bar(x=np.arange(0, len(df)), height=errors, alpha=0.5, label=name)

    plt.axhline(y=7.5, color='black', linestyle='--', linewidth=0.5, label="7.5g threshold")
    plt.xlabel("index")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_errors(path_dict, k):
    dfs = []
    keys = [key for key in path_dict]
    for key in path_dict:
        dfs.append(filter_df(path_dict[key], key))
    
    combined = combine_dfs(dfs)
    topk = get_topk_mae(combined, keys[0], k)
    bar_plot_errors(topk, keys)
        



if __name__ == "__main__":
    path = "/data/lucasjia/projects/nutri/results/multi-nutrient/sub1/samples_carb_base_20250722_043618.jsonl"
    df = filter_df(path, "base_carb")
    print(df.head())
