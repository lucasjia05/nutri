import pandas as pd
import numpy as np
from scipy import stats

# ---------------- LOAD DATA ----------------
file_without = "../../../nutri/results/multi-nutrient/sub9_no_water/samples_combined_base_20250901_021119.jsonl"      # without water
file_with = "../../../nutri/results/multi-nutrient/sub10_water/samples_combined_base_20250901_032336.jsonl"  # with water

df_with = pd.read_json(file_with, lines=True)
df_without = pd.read_json(file_without, lines=True)

# ---------------- EXTRACT SIGNED ERRORS ----------------
nutrients = ["carb", "fat", "energy", "protein"]

for nutrient in nutrients:
    df_with[f"{nutrient}_error"] = df_with["pred"].apply(lambda x: x[nutrient]) - df_with["doc"].apply(lambda x: x[nutrient])
    df_without[f"{nutrient}_error"] = df_without["pred"].apply(lambda x: x[nutrient]) - df_without["doc"].apply(lambda x: x[nutrient])

# ---------------- ALIGN DATA ----------------
# Assuming same queries / order
df_aligned = pd.DataFrame({"doc_id": df_with["doc_id"]})
for nutrient in nutrients:
    df_aligned[f"{nutrient}_with"] = df_with[f"{nutrient}_error"]
    df_aligned[f"{nutrient}_without"] = df_without[f"{nutrient}_error"]

# ---------------- PAIRED T-TESTS ----------------
# Run-to-run std for carbs from 5 repeats
run_to_run_std_carb = 3.181  # replace with your computed value

results = {}
for nutrient in nutrients:
    with_errors = df_aligned[f"{nutrient}_with"].dropna()
    without_errors = df_aligned[f"{nutrient}_without"].dropna()

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(with_errors, without_errors)

    mean_diff = (with_errors - without_errors).mean()
    
    # Compute shift-to-noise ratio only for carbs
    if nutrient == "carb":
        shift_to_noise = mean_diff / run_to_run_std_carb
    else:
        shift_to_noise = np.nan

    results[nutrient] = {
        "mean_with": with_errors.mean(),
        "mean_without": without_errors.mean(),
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value": p_val,
        "shift_to_noise": shift_to_noise
    }

# ---------------- PRINT RESULTS ----------------
for nutrient, vals in results.items():
    print(f"\n=== {nutrient.upper()} ===")
    print(f"Mean signed error WITH water: {vals['mean_with']:.3f}")
    print(f"Mean signed error WITHOUT water: {vals['mean_without']:.3f}")
    print(f"Mean difference (with - without): {vals['mean_diff']:.3f}")
    print(f"T-statistic: {vals['t_stat']:.3f}")
    print(f"P-value: {vals['p_value']:.5f}")
    
    if nutrient == "carb":
        print(f"Shift-to-noise ratio (carb only): {vals['shift_to_noise']:.3f}")
        if abs(vals['shift_to_noise']) > 2:
            print("➡ Shift is much larger than run-to-run noise → likely meaningful")
        elif abs(vals['shift_to_noise']) > 1:
            print("➡ Shift is somewhat larger than run-to-run noise → possibly meaningful")
        else:
            print("➡ Shift is comparable to run-to-run noise → interpret cautiously")
    
    if vals["p_value"] < 0.05:
        print("➡ Statistically significant difference (p < 0.05)")
    else:
        print("➡ No significant difference")
