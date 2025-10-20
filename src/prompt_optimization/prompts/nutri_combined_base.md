# Task
For the given query including a meal description, calculate the total amounts of carbohydrates (grams), protein (grams), fat (grams), and energy (kilocalories). 
If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing four key-value pairs as follows:
{
  "total_carbohydrates": total grams of carbohydrates,
  "total_protein": total grams of protein,
  "total_fat": total grams of fat,
  "total_energy": total kilocalories
}
For each value, respond with just the numeric amount without extra text. If you don't know the answer for any value, set it to -1.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {
  "total_carbohydrates": 66.5,
  "total_protein": 7.65,
  "total_fat": 2.14,
  "total_energy": 312
}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {
  "total_carbohydrates": 15,
  "total_protein": 15.3,
  "total_fat": 17.5,
  "total_energy": 277.2
}

Query: "Half a peanut butter and jelly sandwich."
Answer: {
  "total_carbohydrates": 25.3,
  "total_protein": 6.4,
  "total_fat": 9.3,
  "total_energy": 202
}

# Prediction
Query: {{ text }}
Answer: