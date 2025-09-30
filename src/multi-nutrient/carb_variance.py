import pandas as pd
import numpy as np

# ---------------- LOAD DATA ----------------
repeat_files = [
    "../../../nutri/results/multi-nutrient/sub9_no_water/samples_carb_base_20250831_232738.jsonl",
    "../../../nutri/results/multi-nutrient/sub9_no_water/samples_carb_base_20250831_233833.jsonl",
    "../../../nutri/results/multi-nutrient/sub9_no_water/samples_carb_base_20250831_235724.jsonl",
    "../../../nutri/results/multi-nutrient/sub9_no_water/samples_carb_base_20250901_003316.jsonl",
    "../../../nutri/results/multi-nutrient/sub9_no_water/samples_carb_base_20250901_004333.jsonl",
]

dfs = [pd.read_json(f, lines=True) for f in repeat_files]

# ---------------- ALIGN DATA ----------------
# Assumes same queries & order across files
aligned = pd.DataFrame({
    "doc_id": dfs[0]["doc_id"],
    "truth": dfs[0]["doc"].apply(lambda x: x["carb"])  # true carb value
})

# Collect predictions
for i, df in enumerate(dfs):
    # If pred is a float, just assign it
    aligned[f"pred_{i}"] = df["pred"]

# ---------------- COMPUTE ERRORS ----------------
for i in range(len(dfs)):
    aligned[f"err_{i}"] = aligned[f"pred_{i}"] - aligned["truth"]

# ---------------- VARIANCE ACROSS REPEATS ----------------
aligned["err_variance"] = aligned[[f"err_{i}" for i in range(len(dfs))]].var(axis=1, ddof=1)
aligned["err_std"] = np.sqrt(aligned["err_variance"])

# ---------------- SUMMARY ----------------
print("=== Run-to-run variance of carb errors across repeats ===")
print(f"Mean variance: {aligned['err_variance'].mean():.3f}")
print(f"Mean std: {aligned['err_std'].mean():.3f}")
print(f"Median std: {aligned['err_std'].median():.3f}")

# Optionally inspect distribution
print("\nSample rows:")
print(aligned[["doc_id", "err_std"]].head())
