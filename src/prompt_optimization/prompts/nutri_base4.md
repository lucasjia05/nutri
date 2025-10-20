# Task
You are tasked with estimating the total amount of carbohydrates in grams for a described meal. Use the following steps:
1. **Break Down the Meal**: Identify and list each component of the meal, including any specific quantities mentioned.
2. **Research Specific Ingredients**: Use nutritional databases, such as the USDA or other reliable sources, to find specific carbohydrate values for each ingredient, considering variations in preparation and regional recipes.
3. **Account for Variations**: If the meal includes ingredients that can vary significantly (e.g., chapatti, injera), consider multiple potential values and select the most likely based on the given description.
4. **Estimate Carbohydrates**: Calculate the carbohydrate content for each component, taking into account the specific details provided (e.g., weight, preparation method).
5. **Sum Up**: Calculate the total carbohydrates by summing the estimates for each meal component.

Respond with a dictionary object containing:
{"total_carbohydrates": total grams of carbohydrates}

If the carbohydrate content cannot be accurately estimated due to lack of information or unusual meal components, respond with:
{"total_carbohydrates": -1}

Example Queries and Answers:
Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}

Query: "I have a decaffeinated cola that weighs 744 grams along with a 255-gram diet macaroni and cheese for lunch."
Answer: {"total_carbohydrates": 122.35}

Query: "For lunch, I'm having a large pepper soft drink, a large order of fast food french fries, and a Quarter Pounder with cheese from McDonald's."
Answer: {"total_carbohydrates": 191.11}

For the given queries, ensure the estimations are as accurate as possible based on known nutritional data, and adjust for any variations in preparation or ingredient type.

# Prediction
Query: {{ text }}
Answer:
