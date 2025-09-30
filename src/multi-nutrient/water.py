import pandas as pd
import numpy as np

# ---------------- LOAD JSONL ----------------
file_path = "../../../nutri/results/multi-nutrient/sub8_water/samples_combined_base_20250824_200411.jsonl"
df = pd.read_json(file_path, lines=True)

# ---------------- CALCULATE ERRORS ----------------
nutrients = ["carb", "protein", "fat", "energy"]

metrics = {}

for n in nutrients:
    # Extract ground truth + prediction
    df[f"ground_truth_{n}"] = df["doc"].apply(lambda x: x.get(n, None))
    df[f"pred_{n}"] = df["pred"].apply(lambda x: x.get(n, None))

    # Signed error
    df[f"signed_error_{n}"] = df[f"pred_{n}"] - df[f"ground_truth_{n}"]

    # Metrics
    mean_signed_error = df[f"signed_error_{n}"].mean()
    mae = df[f"signed_error_{n}"].abs().mean()
    rmse = np.sqrt((df[f"signed_error_{n}"] ** 2).mean())

    metrics[n] = {
        "mean_signed_error": mean_signed_error,
        "mae": mae,
        "rmse": rmse
    }

# ---------------- PRINT RESULTS ----------------
print("Nutrient-level error metrics:")
for n, vals in metrics.items():
    print(f"{n}:")
    print(f"  Mean Signed Error: {vals['mean_signed_error']:.4f}")
    print(f"  MAE: {vals['mae']:.4f}")
    print(f"  RMSE: {vals['rmse']:.4f}")
