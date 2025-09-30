import pandas as pd
from openai import OpenAI
from tqdm import tqdm  # <-- import tqdm

# Load dataset
df = pd.read_csv("nutribench_v2_from_laya.csv")

client = OpenAI()
responses = []

# Wrap df.iterrows() with tqdm
for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing meals"):
    meal_description = row["queries"]

    prompt = f"""
You are helping clean a dataset of meal descriptions. 
Your task is to decide whether each query should be kept or removed.

Rules:
- Answer "keep" if the meal description clearly specifies what was eaten and the amounts are specific or understandable. 
- Answer "remove" only if the ingredients are ambiguous, vague, incomplete, nonsensical, or inedible. 
- Ignore phrases like "for breakfast", "for lunch", "for a snack", or any timing/context information — only focus on the clarity of the food and amounts.
- Remove if the ingredients or amounts are not clearly specified (e.g., "vegetables like carrots and broccoli", "a vegetable side", "some", "a little", "or" between options like "chicken or turkey").
- Ignore nutrition, calories, and healthiness.

Respond with exactly one word: "keep" or "remove".

Meal: {meal_description}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3
    )

    answer = completion.choices[0].message.content.strip().split()[0].lower()
    responses.append(answer)

# Add filter_flag only for sampled rows
df["filter_flag"] = responses
df.to_csv("nutribench_filtered.csv", index=False)

print("✅ Finished! Saved nutribench_filtered.csv with keep/remove labels.")
