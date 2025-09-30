import pandas as pd

# Load data
df = pd.read_csv("../data/nutribench_v2_from_laya.csv")

# --- Separate by unit type ---
df_natural = df[df['amount_type'] == 'natural']
df_metric = df[df['amount_type'] == 'metric']

# --- Subset 1: Natural Language ---
# USA: 530 samples
subset_natural = df_natural[df_natural['country'] == 'USA'].sample(n=530, random_state=42)

# --- Subset 2: Metric ---
# USA: 530 samples
subset_metric = df_metric[df_metric['country'] == 'USA'].sample(n=530, random_state=42)

"""# Foreign countries: 10 samples each
foreign_countries_metric = df_metric[df_metric['country'] != 'USA']['country'].unique()
foreign_metric_samples = []

for country in foreign_countries_metric:
    country_df = df_metric[df_metric['country'] == country]
    if len(country_df) >= 10:
        sample = country_df.sample(n=10, random_state=42)
        foreign_metric_samples.append(sample)
    else:
        print(f"Warning: {country} has less than 10 metric samples.")

foreign_metric = pd.concat(foreign_metric_samples)

subset_metric = pd.concat([usa_metric, foreign_metric])"""

subset_natural.to_csv("../data/subset_natural_language.csv", index=False)
subset_metric.to_csv("../data/subset_metric_USA.csv", index=False)
