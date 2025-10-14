# Task
"For the given query, estimate the total carbohydrates in grams based on the provided meal description. Assume standard serving sizes when unspecified, using common nutritional guidelines. Pay special attention to ingredients with high carbohydrate variability, such as flavored drinks, processed foods, or specific types of beverages like alcoholic drinks. For complex meal items like burritos or calzones, consider typical proportions of ingredients, including tortillas, rice, beans, and fillings. Differentiate between beverage types, as their carbohydrate content can vary significantly. Use nutritional databases or guidelines to guide your estimates.

Provide your response as a dictionary object containing the total carbohydrates: {"total_carbohydrates": total grams of carbohydrates}. Provide only the numeric amount without any additional text. If the meal description is too vague or uncertain, respond with: {"total_carbohydrates": -1}.

Example Queries and Answers:

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}

Query: "For lunch, I\'m having a 512g cola and a 340g burrito filled with meat, beans, rice, and sour cream."
Answer: {"total_carbohydrates": 107.10}

Query: "I’ve got a diet cola that weighs 720 grams and a meat and cheese calzone for dinner, which is 424 grams."
Answer: {"total_carbohydrates": 133.23}

Query: "For my snack, I have 57 grams of Doritos tortilla chips and 600 grams of bottled flavored Vitamin Water."
Answer: {"total_carbohydrates": 67.60}"

Query: "For dinner, I\'m having 115 grams of Ritz butter crackers with 425 grams of ready-to-heat pasta in tomato-based sauce with meat."
Answer: {"total_carbohydrates": 150.14}

Query: "This morning, I’m eating 350g of boiled coffee beans with salt, 279g of leavened bread made from corn and sorghum, and 118g of roasted corn."
Answer: {"total_carbohydrates": 159.50}

Query: "For a snack, I\'m having 250g of lemon-flavored powdered drink mixed with water, 16g of pancakes, and 40g of pasta with tomato sauce."
Answer: {"total_carbohydrates": 33.16}

# Prediction
Query: {{ text }}
Answer: