# Task
"For the provided query, accurately estimate the total carbohydrate content in grams based on the detailed meal description. Consider the weights of each ingredient, and utilize reputable nutritional databases to determine the carbohydrate values for each component. Be thorough in accounting for all items listed, including beverages and any ingredients that may contain hidden carbohydrates.

1. Break down the meal description into individual components.
2. For each component, look up its carbohydrate value per the specified weight.
3. If a specific carbohydrate value is not available, refer to standard serving sizes from established nutritional guidelines (e.g., USDA).
4. Include beverages and their common nutritional values based on their preparation.

Once all components have been evaluated, sum the carbohydrate values to derive the total. Format your answer as a dictionary object:
{"total_carbohydrates": total grams of carbohydrates for the meal}

If it is impossible to determine the total carbohydrates due to insufficient information or ambiguity in the description, respond with:
{"total_carbohydrates": -1}.

Query: "I had a bagel with cream cheese and a side of strawberries for breakfast."
Answer: {"total_carbohydrates": 65.2}


# Prediction
Query: {{ text }}
Answer: