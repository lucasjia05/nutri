# ----------------- IMPORTS -----------------
import pandas as pd

# ----------------- LOAD CSV -----------------
df = pd.read_csv("nutribench_v2_from_laya.csv")

# ----------------- FILTER TO QUERIES WITHOUT WATER -----------------
df_no_water = df[~df["queries"].str.contains("water", case=False, na=False)].copy()

# ----------------- SAMPLE ABOUT 1000 -----------------
sample_size = min(1000, len(df_no_water))
subset = df_no_water.sample(sample_size, random_state=42).reset_index(drop=True)

# ----------------- CREATE WATER-VERSION -----------------
df_with_water = subset.copy()

# Add ingredient to description (assuming it’s stored as a Python-like list in the CSV)
df_with_water["description"] = df_with_water["description"].str.rstrip("]") + ", 'Water, bottled, unsweetened']"

# Add bottled water mention to query
df_with_water["queries"] = df_with_water["queries"].str.rstrip(".") + ", and 507 grams of bottled water."

# ----------------- SAVE TO TWO MATCHING CSVs -----------------
subset.to_csv("sub9_no_water.csv", index=False)
df_with_water.to_csv("sub10_with_water.csv", index=False)

print(f"✅ Saved {sample_size} rows in sub9_no_water.csv and {sample_size} rows in sub9_with_water.csv (corresponding rows)")
