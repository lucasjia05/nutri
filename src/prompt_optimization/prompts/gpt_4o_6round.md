# Task
For the given query including a meal description, estimate the amount of carbohydrates in grams. Assume that the serving size of any item not specified is a single standard serving based on common nutritional guidelines (e.g., USDA). When dealing with ambiguous descriptions or mixed-item meals, make reasonable assumptions based on common recipes or typical food compositions. However, pay special attention to items that typically have low or negligible carbohydrate content, such as certain flavored drinks or small servings of snacks. Do not overestimate carbohydrates for these items.

Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}

Ensure that the total carbohydrates are provided as a numeric amount without extra text. If it's impossible to give a reasonable estimate due to lack of information, respond with:
{"total_carbohydrates": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}

Query: "I had a simple breakfast with 3.9 grams of black tea, 60.8 grams of white sugar, and a hefty serving of 426 grams of boiled yams."
Answer: {"total_carbohydrates": 183.90}

Query: "I grabbed a 259-gram breakfast taco filled with egg, potato, and some kind of breakfast meat."
Answer: {"total_carbohydrates": 45.95}

Query: "For a snack, I'm having 250g of lemon-flavored powdered drink mixed with water, 16g of pancakes, and 40g of pasta with tomato sauce."
Answer: {"total_carbohydrates": 33.16}

Query: "For my snack, I have 57 grams of Doritos tortilla chips and 600 grams of bottled flavored Vitamin Water."
Answer: {"total_carbohydrates": 67.60}

# Prediction
Query: {{ text }}
Answer: