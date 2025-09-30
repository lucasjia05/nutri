import pandas as pd
from scipy import stats
import numpy as np

# ---------------- CONFIG ----------------
nutrient = "protein"  # Change this to "fat", "energy", or "protein" as needed

file_without = "../../../nutri/results/multi-nutrient/sub9_no_water/samples_protein_base_20250901_012255.jsonl"  # without water
file_with = "../../../nutri/results/multi-nutrient/sub10_water/samples_protein_base_20250901_013225.jsonl"      # with water

# ---------------- LOAD DATA ----------------
df_with = pd.read_json(file_with, lines=True)
df_without = pd.read_json(file_without, lines=True)

# ---------------- EXTRACT SIGNED ERRORS ----------------
df_with[f"{nutrient}_error"] = df_with["pred"] - df_with["doc"].apply(lambda x: x[nutrient])
df_without[f"{nutrient}_error"] = df_without["pred"] - df_without["doc"].apply(lambda x: x[nutrient])

# ---------------- ALIGN DATA ----------------
df_aligned = pd.DataFrame({
    "doc_id": df_with["doc_id"],
    "with_water": df_with[f"{nutrient}_error"],
    "without_water": df_without[f"{nutrient}_error"]
})

# ---------------- PAIRED T-TEST ----------------
with_errors = df_aligned["with_water"].dropna()
without_errors = df_aligned["without_water"].dropna()

t_stat, p_val = stats.ttest_rel(with_errors, without_errors)
mean_diff = (with_errors - without_errors).mean()

# ---------------- PRINT RESULTS ----------------
print(f"\n=== {nutrient.upper()} ===")
print(f"Mean signed error WITH water: {with_errors.mean():.3f}")
print(f"Mean signed error WITHOUT water: {without_errors.mean():.3f}")
print(f"Mean difference (with - without): {mean_diff:.3f}")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.5f}")