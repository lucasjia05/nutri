# ------------------ PROMPTS ------------------
prompt_carb = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, respond with:
{"total_carbohydrates": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}'''

prompt_protein = '''For the given query including a meal description, calculate the amount of protein in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer,
respond with:
{"total_protein": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_protein": 7.65}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_protein": 6.4}'''

prompt_fat = '''For the given query including a meal description, calculate the amount of fat in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, respond with:
{"total_fat": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_fat": 2.14}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_fat": 9.3}'''

prompt_energy = '''For the given query including a meal description, calculate the amount of energy in kilocalories (kcal). If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount without extra text. If you don't know the answer, respond with:
{"total_energy": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_energy": 202}'''

prompt_combined = '''For the given query including a meal description, calculate the total amounts of carbohydrates (grams), protein (grams), fat (grams), and energy (kilocalories). 
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
}'''

prompt_carb_cot = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size.
3. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 27g carbs.
1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.
1 glass of orange juice has 26g carbs.
So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5
Output: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 2g carbs.
1 toast has 13g carbs.
So the total grams of carbs in the meal = (2 + 13) = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
So the total grams of carbs in the meal = 25.3
Output: {"total_carbohydrates": 25.3}'''

prompt_fat_cot='''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of fat in grams for the specific serving size.
3. Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, set the value of "total_fat" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 1.1g fat.
1 banana has 0.4g fat so half a banana has (0.4*(1/2)) = 0.2g fat.
1 glass of orange juice has 0.84g fat.
So the total grams of fat in the meal = (1.1 + 0.2 + 0.84) = 2.14
Output: {"total_fat": 2.14}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 16.5g fat.
1 toast has 0.985g fat.
So the total grams of fat in the meal = (16.5 + 1) = 17.5
Output: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
So the total grams of fat in the meal = 9.3
Output: {"total_fat": 9.3}'''


prompt_energy_cot = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of energy in kilocalories (kcal) for the specific serving size.
3. Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount of kilocalories without extra text. If you don't know the answer, set the value of "total_energy" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 134 kcal.
1 banana has 122 kcal so half a banana has (122*(1/2)) = 61 kcal.
1 glass of orange juice has 117 kcal.
So the total kilocalories in the meal = (134 + 61 + 117) = 312
Output: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 204 kcal.
1 toast has 73.2 kcal.
So the total kilocalories in the meal = (204 + 73.2) = 277.2
Output: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
So the total kilocalories in the meal = 202
Output: {"total_energy": 202}'''

prompt_protein_cot = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of protein in grams for the specific serving size.
3. Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, set the value of "total_protein" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 5.1g protein.
1 banana has 1.3g protein so half a banana has (0.9*(1/2)) = 0.45g protein.
1 glass of orange juice has 1.9g protein.
So the total grams of protein in the meal = (5.1 + 0.45 + 1.9) = 7.65
Output: {"total_protein": 7.65}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 12.7g protein.
1 toast has 2.6g protein.
So the total grams of protein in the meal = (12.7 + 2.6) = 15.3
Output: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
So the total grams of protein in the meal = 6.4
Output: {"total_protein": 6.4}'''



prompt_combined_cot = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of carbohydrates (grams), protein (grams), fat (grams), and energy (kilocalories) for the specific serving size.
3. Respond with a dictionary object containing the total nutrients as follows:
{
  "total_carbohydrates": total grams of carbohydrates,
  "total_protein": total grams of protein,
  "total_fat": total grams of fat,
  "total_energy": total kilocalories
}
For each value, respond with just the numeric amount without extra text. If you don't know the answer for any value, set it to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana, and 1 glass of orange juice.
1 cup of oatmeal has 27g carbohydrates, 5.1g protein, 1.1g fat, and 134 kcal.
1 banana has 27g carbohydrates, 1.3g protein, 0.4g fat, and 122 kcal, so half a banana has (27*(1/2))=13.5g carbohydrates, (1.3*(1/2))=0.65g protein, (0.4*(1/2))=0.2g fat, and (122*(1/2))=61 kcal.
1 glass of orange juice has 26g carbohydrates, 1.9g protein, 0.84g fat, and 117 kcal.
So the total nutrients in the meal = (27 + 13.5 + 26) = 66.5g carbohydrates, (5.1 + 0.65 + 1.9) = 7.65g protein, (1.1 + 0.2 + 0.84) = 2.14g fat, (134 + 61 + 117) = 312 kcal.
Output: {
  "total_carbohydrates": 66.5,
  "total_protein": 7.65,
  "total_fat": 2.14,
  "total_energy": 312
}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 2g carbohydrates, 12.7g protein, 16.5g fat, and 204 kcal.
1 toast has 13g carbohydrates, 2.6g protein, 0.985g fat, and 73.2 kcal.
So the total nutrients in the meal = (2 + 13)=15g carbohydrates, (12.7 + 2.6)=15.3g protein, (16.5 + 0.985)=17.5g fat, (204 + 73.2)=277.2 kcal.
Output: {
  "total_carbohydrates": 15,
  "total_protein": 15.3,
  "total_fat": 17.5,
  "total_energy": 277.2
}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 50.6g carbohydrates, 12.8g protein, 18.6g fat, and 404 kcal.
So half a sandwich has (50.6*(1/2))=25.3g carbohydrates, (12.8*(1/2))=6.4g protein, (18.6*(1/2))=9.3g fat, (404*(1/2))=202 kcal.
Output: {
  "total_carbohydrates": 25.3,
  "total_protein": 6.4,
  "total_fat": 9.3,
  "total_energy": 202
}'''

prompt_carb_context_energy2 = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. The query may include additional known nutrient information (e.g., energy in kilocalories). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, respond with:
{"total_carbohydrates": -1}.

Query: "The following meal contains 312 kcal. This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_carbohydrates": 15}

Query: "The following meal contains 202 kcal. Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_carbohydrates": 16.5}'''


prompt_carb_cot_context_energy7= '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams.
4. Add the carbohydrates from all items. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the meal}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. The meal consists of:
- Scrambled eggs made with 2 eggs
- 1 toast
2. No energy provided.
3. Estimate carbohydrate content in grams.
Scrambled eggs made with 2 eggs has 2g carbs. 1 toast has 13g carbs.
4. Add carbohydrate values.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "The following meal contains 106.6 kcal. I've got a drink made from 248 grams of oatmeal and water for breakfast."
Answer: Let's think step by step.
1. The meal consists of:
- A drink made from 248g of oatmeal and water
2. Estimate energy contributions per item.
248g of oatmeal: Approx. 70 kcal / 100g
248g * 0.7 kcal / g =  173.6 kcal. This is far above 106.6 kcal, so something is off.
Reconsider oatmeal: 248g may be referring to the entire drink, rather than the amount of dry oatmeal.
248g of oatmeal drink: Approx. 45 kcal / 100g
248 * 0.45 kcal / g = 111.6 kcal - acceptable.
3. Estimate carbohydrate content in grams.
An oatmeal drink typically has around 10.5g carbs per 100g:
248g * 0.105 = 26.04g carbs
4. Add the carbs from all items and return the total as a dictionary.
Output: {"total_carbohydrates": 26.04}

Query: "The following meal contains 1470 kcal. For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate energy contributions per item.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 884 * 0.619 = 547 kcal
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has ~50-60 kcal per 100g.
729g * 0.55 = 400.95 kcal.
Total estimated: 547 + 401 = 948 kcal. This is far below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali or polenta consistency).
Try 729 * 1.27 = 926.8 kcal. (127 kcal/100g assumption)
New total: 547 (oil) + 927 (porridge) ≈ 1474 kcal, matching well. So, we assume maize porridge at 127 kcal/100g.
3. Estimate carbohydrate content in grams.
Red palm oil: 0g carbs (pure fat)
Maize porridge: At 127 kcal/100g, and assuming 80% of energy is from carbs:
127 kcal * 0.90 = 114.3 kcal from carbs
So 114.3 / 4 = 28.6g carbs per 100g
Using 28.6g/100g as a realistic estimate:
729g * 0.286 = 208.5g carbs
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 208.5}'''


# ------------------ PROMPTS FOR ROTATING BASE NUTRIENT CONTEXTS ------------------

carb_w_protein = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. The query may include additional known nutrient information (e.g., protein in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, respond with:
{"total_carbohydrates": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 8 grams of protein."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich. The meal contained 6 grams of protein."
Answer: {"total_carbohydrates": 25.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_carbohydrates": 2.75}'''

carb_w_fat = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. The query may include additional known nutrient information (e.g., fat in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, respond with:
{"total_carbohydrates": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 2 grams of fat."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich. The meal contained 9 grams of fat."
Answer: {"total_carbohydrates": 25.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_carbohydrates": 2.75}'''

carb_w_energy = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. The query may include additional known nutrient information (e.g., energy in kilocalories). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, respond with:
{"total_carbohydrates": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 312 kcal."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich. The meal contained 202 kcal."
Answer: {"total_carbohydrates": 25.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_carbohydrates": 2.75}'''

protein_w_carb = '''For the given query including a meal description, calculate the amount of protein in grams. The query may include additional known nutrient information (e.g., carbohydrates in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, respond with:
{"total_protein": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 67 grams of carbohydrates."
Answer: {"total_protein": 7.7}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich. The meal contained 25 grams of carbohydrates."
Answer: {"total_protein": 6.4}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_protein": 1.7}'''

protein_w_fat = '''For the given query including a meal description, calculate the amount of protein in grams. The query may include additional known nutrient information (e.g., fat in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, respond with:
{"total_protein": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 2 grams of fat."
Answer: {"total_protein": 7.7}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich. The meal contained 9 grams of fat."
Answer: {"total_protein": 6.4}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_protein": 1.7}'''

protein_w_energy = '''For the given query including a meal description, calculate the amount of protein in grams. The query may include additional known nutrient information (e.g., energy in kilocalories). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, respond with:
{"total_protein": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 312 kcal."
Answer: {"total_protein": 7.7}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich. The meal contained 202 kcal."
Answer: {"total_protein": 6.4}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_protein": 1.7}'''

fat_w_carb = '''For the given query including a meal description, calculate the amount of fat in grams. The query may include additional known nutrient information (e.g., carbohydrates in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, respond with:
{"total_fat": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 67 grams of carbohydrates."
Answer: {"total_fat": 2.1}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich. The meal contained 25 grams of carbohydrates."
Answer: {"total_fat": 9.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_fat": 1.0}'''

fat_w_protein = '''For the given query including a meal description, calculate the amount of fat in grams. The query may include additional known nutrient information (e.g., protein in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, respond with:
{"total_fat": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 8 grams of protein."
Answer: {"total_fat": 2.1}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich. The meal contained 6 grams of protein."
Answer: {"total_fat": 9.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_fat": 1.0}'''

fat_w_energy = '''For the given query including a meal description, calculate the amount of fat in grams. The query may include additional known nutrient information (e.g., energy in kilocalories). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, respond with:
{"total_fat": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 312 kcal."
Answer: {"total_fat": 2.1}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich. The meal contained 202 kcal."
Answer: {"total_fat": 9.3}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_fat": 1.0}'''

energy_w_carb = '''For the given query including a meal description, calculate the amount of energy in kilocalories. The query may include additional known nutrient information (e.g., carbohydrates in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount of energy without extra text. If you don't know the answer, respond with:
{"total_energy": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 67 grams of carbohydrates."
Answer: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich. The meal contained 25 grams of carbohydrates."
Answer: {"total_energy": 202}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_energy": 27}'''

energy_w_protein = '''For the given query including a meal description, calculate the amount of energy in kilocalories. The query may include additional known nutrient information (e.g., protein in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount of energy without extra text. If you don't know the answer, respond with:
{"total_energy": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 8 grams of protein."
Answer: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich. The meal contained 6 grams of protein."
Answer: {"total_energy": 202}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_energy": 27}'''

energy_w_fat = '''For the given query including a meal description, calculate the amount of energy in kilocalories. The query may include additional known nutrient information (e.g., fat in grams). If provided, use this information to improve your estimate. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount of energy without extra text. If you don't know the answer, respond with:
{"total_energy": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice. The meal contained 2 grams of fat."
Answer: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich. The meal contained 9 grams of fat."
Answer: {"total_energy": 202}

Query: "A large cup of cappuccino made with whole milk."
Answer: {"total_energy": 27}'''



# ------------------ PROMPTS FOR ROTATING CoT NUTRIENT CONTEXTS ------------------

# ------------------ CARB + CONTEXT PROMPTS ------------------

carb_cot_w_protein = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total protein (in grams) is provided, estimate the amount of protein each item contributes to the total. If the total protein estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent protein estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams.
4. Add the carbohydrate values from all items. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the meal}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No protein provided.
3. Estimate carbohydrate content in grams.
Scrambled eggs made with 2 eggs has 2g carbs. 1 toast has 13g carbs.
4. Add carbohydrate values and respond with total.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich. The meal contained 6 grams of protein."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate protein contributions per item.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
Matches the provided total (no need to adjust assumptions).
3. Estimate carbohydrate content in grams.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 25.3}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 20g protein."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate protein contributions per item.
Red palm oil: ~0g protein.
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 2g protein per 100g.
729g * 0.02 = 14.6g protein.
Total estimated: 0 + 14.6 = 14.6g protein. This is more than 20% below the stated 20g protein.
Adjustment needed: The porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 2.8g protein per 100g (closer to stiff ugali consistency).
729 * 0.028 = 20.4g protein, which matches the provided 20g context.
So we assume a denser porridge at 2.8g protein/100g.
3. Estimate carbohydrate content in grams.
Red palm oil: 0g carbs (pure fat).
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), carb content is typically 28.6g carbs per 100g.
729g * 0.286 = 208.5g carbs.
4. Add carbohydrate values and respond with total.
Total = 0 + 208.5 = 208.5
Output: {"total_carbohydrates": 208.5}'''



carb_cot_w_fat = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total fat (in grams) is provided, estimate the amount of fat each item contributes to the total. If the total fat estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent fat estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams.
4. Add the carbohydrate values from all items. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the meal}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No fat provided.
3. Estimate carbohydrate content in grams.
Scrambled eggs made with 2 eggs has 2g carbs. 1 toast has 13g carbs.
4. Add carbohydrate values and respond with total.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich. The meal contained 9 grams of fat."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate fat contributions per item.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
Matches the provided total (no need to adjust assumptions).
3. Estimate carbohydrate content in grams.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 25.3}

Query: "For breakfast, I had 729.0g of ready-to-eat maize porridge. The meal contained 10 grams of fat."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 729g of ready-to-eat maize porridge
2. Estimate fat contributions per item.
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 1g fat per 100g.
729g * 0.1 = 7.3 grams. This is more than 20% below the stated 10g fat.
Adjustment needed: The maize porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 1.5 grams/100g (closer to stiff ugali consistency).
Try 729 * 0.014 = 10.2 grams (1.4g fat per 100g assumption), matching well. So, we assume maize porridge at 1.4g/100g.
3. Estimate carbohydrate content in grams.
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), carb content is typically 28.6g carbs per 100g.
729g * 0.286 = 208.5g carbs.
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 208.5}'''


carb_cot_w_energy = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams.
4. Add the carbohydrates from all items. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the meal}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No energy provided.
3. Estimate carbohydrate content in grams.
Scrambled eggs made with 2 eggs has 2g carbs. 1 toast has 13g carbs.
4. Add carbohydrate values.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "I've got a drink made from 248 grams of oatmeal and water for breakfast. The meal contained 106.6 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- A drink made from 248g of oatmeal and water
2. Estimate energy contributions per item.
248g of oatmeal: Approx. 70 kcal / 100g
248g * 0.7 kcal / g =  173.6 kcal. This is far above 106.6 kcal, so something is off.
Reconsider oatmeal: 248g may be referring to the entire drink, rather than the amount of dry oatmeal.
248g of oatmeal drink: Approx. 45 kcal / 100g
248 * 0.45 kcal / g = 111.6 kcal - acceptable.
3. Estimate carbohydrate content in grams.
An oatmeal drink typically has around 10.5g carbs per 100g:
248g * 0.105 = 26.04g carbs
4. Add the carbs from all items and return the total as a dictionary.
Output: {"total_carbohydrates": 26.04}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 1470 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate energy contributions per item.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 884 * 0.619 = 547 kcal
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has ~50-60 kcal per 100g.
729g * 0.55 = 400.95 kcal.
Total estimated: 547 + 401 = 948 kcal. This is far below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali consistency).
Try 729 * 1.27 = 926.8 kcal. (127 kcal/100g assumption)
New total: 547 (oil) + 927 (porridge) ≈ 1474 kcal, matching well. So, we assume maize porridge at 127 kcal/100g.
3. Estimate carbohydrate content in grams.
Red palm oil: 0g carbs (pure fat)
Maize porridge: At 127 kcal/100g, and assuming 80% of energy is from carbs:
127 kcal * 0.90 = 114.3 kcal from carbs
So 114.3 / 4 = 28.6g carbs per 100g
Using 28.6g/100g as a realistic estimate:
729g * 0.286 = 208.5g carbs
4. Add carbohydrate values and respond with total.
0 + 208.5 = 208.5
Output: {"total_carbohydrates": 208.5}'''


carb_cot_w_energy2 = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (in kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams.
4. Add the carbohydrate values from all items. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the meal}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No energy provided.
3. Estimate carbohydrate content in grams.
Scrambled eggs made with 2 eggs has 2g carbs. 1 toast has 13g carbs.
4. Add carbohydrate values and respond with total.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich. The meal contained 202 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate energy contributions per item.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
Matches the provided total (no need to adjust assumptions).
3. Estimate carbohydrate content in grams.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 25.3}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 1470 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate energy contributions per item.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 884 * 0.619 = 547 kcal
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has ~50-60 kcal per 100g.
729g * 0.55 = 400.95 kcal.
Total estimated: 547 + 401 = 948 kcal. This is more than 20% below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali consistency).
Try 729 * 1.27 = 926 kcal. (127 kcal/100g assumption)
New total: 547 (oil) + 926 (porridge) ≈ 1473 kcal, matching well. So, we assume maize porridge at 127 kcal/100g.
3. Estimate carbohydrate content in grams.
Red palm oil: 0g carbs (pure fat)
Maize porridge: At 127 kcal/100g, and assuming 80% of energy is from carbs:
127 kcal * 0.90 = 114.3 kcal from carbs
So 114.3 / 4 = 28.6g carbs per 100g
Using 28.6g/100g as a realistic estimate:
729g * 0.286 = 208.5g carbs
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 208.5}'''


# ------------------ PROTEIN + CONTEXT PROMPTS ------------------

protein_cot_w_carb = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total carbohydrates (in grams) are provided, estimate the amount of carbs each item contributes to the total. If the total carbohydrates estimated differ by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent carbohydrate estimate is obtained.
3. For each item settled on, estimate its protein content in grams.
4. Add the protein values from all items. Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the meal}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, set the value of "total_protein" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No carbohydrates provided.
3. Estimate protein content in grams.
Scrambled eggs made with 2 eggs has 12.7g protein.
1 toast has 2.6g protein.
4. Add protein values and respond with total.
Total = 12.7 + 2.6 = 15.3
Output: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich. The meal contained 25 grams of carbohydrates."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate carb contributions per item.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
Matches the provided total (no need to adjust assumptions).
3. Estimate protein content in grams.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
4. Add protein values and respond with total.
Output: {"total_protein": 6.4}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 209 grams of carbohydrates."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate carb contributions per item.
Red palm oil: Carb content is about 0g per 100g.
For 61.9g: 61.9 * 0 = 0g
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 15g carbs per 100g.
729g * 0.15 = 109.35 grams.
Total estimated: 0 + 109.35 = 109.35g carbs. This is more than 20% below the stated 209 grams.
Adjustment needed: The maize porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 30 grams/100g (closer to stiff ugali consistency).
Try 729 * 0.286 = 208.5 grams. (28.6g/100g assumption)
New total: 0 (oil) + 208.5 (porridge) ≈ 208.5 grams, matching well. So, we assume maize porridge at 28.6g/100g.
3. Estimate protein content in grams.
Red palm oil: 0g protein (pure fat)
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), protein content is typically 2.8g protein per 100g.
729g * 0.028 = 20.4g protein.
4. Add protein values and respond with total.
Total = 0 + 20.4 = 20.4
Output: {"total_protein": 20.4}'''


protein_cot_w_fat = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total fat (in grams) is provided, estimate the amount of fat each item contributes to the total. If the total fat estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent fat estimate is obtained.
3. For each item settled on, estimate its protein content in grams.
4. Add the protein values from all items. Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the meal}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, set the value of "total_protein" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No fat provided.
3. Estimate protein content in grams.
Scrambled eggs made with 2 eggs has 12.7g protein.
1 toast has 2.6g protein.
4. Add protein values and respond with total.
Total = 12.7 + 2.6 = 15.3
Output: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich. The meal contained 9 grams of fat."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate fat contributions per item.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
Matches the provided total (no need to adjust assumptions).
3. Estimate protein content in grams.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
4. Add protein values and respond with total.
Output: {"total_protein": 6.4}

Query: "For breakfast, I had 729.0g of ready-to-eat maize porridge. The meal contained 10 grams of fat."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 729g of ready-to-eat maize porridge
2. Estimate fat contributions per item.
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 1g fat per 100g.
729g * 0.01 = 7.3 grams. This is more than 20% below the stated 10g fat.
Adjustment needed: The maize porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 1.5 grams/100g (closer to stiff ugali consistency).
Try 729 * 0.014 = 10.2 grams (1.4g fat per 100g assumption), matching well. So, we assume maize porridge at 1.4g/100g.
3. Estimate protein content in grams.
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), protein content is typically 2.8g protein per 100g.
729g * 0.028 = 20.4g protein.
4. Add protein values and respond with total.
Output: {"total_protein": 20.4}'''

protein_cot_w_energy = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (in kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its protein content in grams.
4. Add the protein values from all items. Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the meal}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, set the value of "total_protein" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No energy provided.
3. Estimate protein content in grams.
Scrambled eggs made with 2 eggs has 12.7g protein.
1 toast has 2.6g protein.
4. Add protein values and respond with total.
Total = 12.7 + 2.6 = 15.3
Output: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich. The meal contained 202 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate energy contributions per item.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
Matches the provided total (no need to adjust assumptions).
3. Estimate protein content in grams.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
4. Add protein values and respond with total.
Output: {"total_protein": 6.4}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 1470 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate energy contributions per item.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 884 * 0.619 = 547 kcal
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has ~50-60 kcal per 100g.
729g * 0.55 = 400.95 kcal.
Total estimated: 547 + 401 = 948 kcal. This is more than 20% below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali consistency).
Try 729 * 1.27 = 926 kcal. (127 kcal/100g assumption)
New total: 547 (oil) + 926 (porridge) ≈ 1473 kcal, matching well. So, we assume maize porridge at 127 kcal/100g.
3. Estimate protein content in grams.
Red palm oil: 0g protein (pure fat)
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), protein content is typically 2.8g protein per 100g.
729g * 0.028 = 20.4g protein.
4. Add protein values and respond with total.
Total = 0 + 20.4 = 20.4
Output: {"total_protein": 20.4}'''


# ------------------ FAT + CONTEXT PROMPTS ------------------

fat_cot_w_carb = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total carbohydrates (in grams) are provided, estimate the amount of carbs each item contributes to the total. If the total carbohydrates estimated differ by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent carbohydrate estimate is obtained.
3. For each item settled on, estimate its fat content in grams.
4. Add the fat values from all items. Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total fat in grams for the meal}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, set the value of "total_fat" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No carbohydrates provided.
3. Estimate fat content in grams.
Scrambled eggs made with 2 eggs has 16.5g fat.
1 toast has 1g fat.
4. Add fat values and respond with total.
Total = (16.5 + 1) = 17.5
Output: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich. The meal contained 25 grams of carbohydrates."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate carb contributions per item.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
Matches the provided total (no need to adjust assumptions).
3. Estimate fat content in grams.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
4. Add fat values and respond with total.
Output: {"total_fat": 9.3}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 209 grams of carbohydrates."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate carb contributions per item.
Red palm oil: Carb content is about 0g per 100g.
For 61.9g: 61.9 * 0 = 0g
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 15g carbs per 100g.
729g * 0.15 = 109.35 grams.
Total estimated: 0 + 109.35 = 109.35g carbs. This is more than 20% below the stated 209 grams.
Adjustment needed: The maize porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 30 grams/100g (closer to stiff ugali consistency).
Try 729 * 0.286 = 208.5 grams. (28.6g/100g assumption)
New total: 0 (oil) + 208.5 (porridge) ≈ 208.5 grams, matching well. So, we assume maize porridge at 28.6g/100g.
3. Estimate fat content in grams.
Red palm oil: fat content is about 100g per 100g (pure fat)
For 61.9g: 61.9 * 1 = 61.9g
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), fat content is typically 1.4g fat per 100g.
729g * 0.014 = 10.2g fat.
4. Add fat values and respond with total.
Total = 61.9 + 10.2 = 72.1
Output: {"total_fat": 72.1}'''


fat_cot_w_protein = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total protein (in grams) is provided, estimate the amount of protein each item contributes to the total. If the total protein estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent protein estimate is obtained.
3. For each item settled on, estimate its fat content in grams.
4. Add the fat values from all items. Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total fat in grams for the meal}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, set the value of "total_fat" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No protein provided.
3. Estimate fat content in grams.
Scrambled eggs made with 2 eggs has 16.5g fat.
1 toast has 1g fat.
4. Add fat values and respond with total.
Total = (16.5 + 1) = 17.5
Output: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich. The meal contained 6 grams of protein."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate protein contributions per item.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
Matches the provided total (no need to adjust assumptions).
3. Estimate fat content in grams.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
4. Add fat values and respond with total.
Output: {"total_fat": 9.3}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 20g protein."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate protein contributions per item.
Red palm oil: ~0g protein.
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 2g protein per 100g.
729g * 0.02 = 14.6g protein.
Total estimated: 0 + 14.6 = 14.6g protein. This is more than 20% below the stated 20g protein.
Adjustment needed: The porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 2.8g protein per 100g (closer to stiff ugali consistency).
729 * 0.028 = 20.4g protein, which matches the provided 20g context.
So we assume a denser porridge at 2.8g protein/100g.
3. Estimate fat content in grams.
Red palm oil: fat content is about 100g per 100g (pure fat)
For 61.9g: 61.9 * 1 = 61.9g
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), fat content is typically 1.4g fat per 100g.
729g * 0.014 = 10.2g fat.
4. Add fat values and respond with total.
Total = 61.9 + 10.2 = 72.1
Output: {"total_fat": 72.1}'''


fat_cot_w_energy = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (in kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its fat content in grams.
4. Add the fat values from all items. Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total fat in grams for the meal}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, set the value of "total_fat" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No energy provided.
3. Estimate fat content in grams.
Scrambled eggs made with 2 eggs has 16.5g fat.
1 toast has 1g fat.
4. Add fat values and respond with total.
Total = (16.5 + 1) = 17.5
Output: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich. The meal contained 202 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate energy contributions per item.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
Matches the provided total (no need to adjust assumptions).
3. Estimate fat content in grams.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
4. Add fat values and respond with total.
Output: {"total_fat": 9.3}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 1470 kcal."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate energy contributions per item.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 884 * 0.619 = 547 kcal
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has ~50-60 kcal per 100g.
729g * 0.55 = 400.95 kcal.
Total estimated: 547 + 401 = 948 kcal. This is more than 20% below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali consistency).
Try 729 * 1.27 = 926 kcal. (127 kcal/100g assumption)
New total: 547 (oil) + 926 (porridge) ≈ 1473 kcal, matching well. So, we assume maize porridge at 127 kcal/100g.
3. Estimate fat content in grams.
Red palm oil: fat content is about 100g per 100g (pure fat)
For 61.9g: 61.9 * 1 = 61.9g
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), fat content is typically 1.4g fat per 100g.
729g * 0.014 = 10.2g fat.
4. Add fat values and respond with total.
Total = 61.9 + 10.2 = 72.1
Output: {"total_fat": 72.1}'''



# ------------------ ENERGY + CONTEXT PROMPTS ------------------

energy_cot_w_carb = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total carbohydrates (in grams) are provided, estimate the amount of carbs each item contributes to the total. If the total carbohydrates estimated differ by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent carbohydrate estimate is obtained.
3. For each item settled on, estimate its energy content in kcal.
4. Add the energy values from all items. Respond with a dictionary object containing the total energy in kcal as follows:
{"total_energy": total energy in kcal for the meal}
For the total energy, respond with just the numeric amount of energy without extra text. If you don't know the answer, set the value of "total_energy" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No carbohydrates provided.
3. Estimate energy content in kcal.
Scrambled eggs made with 2 eggs has 204 kcal.
1 toast has 73.2 kcal.
4. Add energy values and respond with total.
Total = (204 + 73.2) = 277.2
Output: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich. The meal contained 25 grams of carbohydrates."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate carb contributions per item.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
Matches the provided total (no need to adjust assumptions).
3. Estimate energy content in kcal.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
4. Add energy values and respond with total.
Output: {"total_energy": 202}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 209 grams of carbohydrates."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate carb contributions per item.
Red palm oil: Carb content is about 0g per 100g.
For 61.9g: 61.9 * 0 = 0g
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 15g carbs per 100g.
729g * 0.15 = 109.35 grams.
Total estimated: 0 + 109.35 = 109.35g carbs. This is more than 20% below the stated 209 grams.
Adjustment needed: The maize porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 30 grams/100g (closer to stiff ugali consistency).
Try 729 * 0.286 = 208.5 grams. (28.6g/100g assumption)
New total: 0 (oil) + 208.5 (porridge) ≈ 208.5 grams, matching well. So, we assume maize porridge at 28.6g/100g.
3. Estimate energy content in kcal.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 61.9 * 8.84 = 547 kcal
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), energy content is typically 127 kcal/100g.
For 729g: 729 * 1.27 kcal = 926 kcal
4. Add energy values and respond with total.
Total = 547 (oil) + 926 (porridge) = 1473 kcal
Output: {"total_energy": 1473}'''

energy_cot_w_protein = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total protein (in grams) is provided, estimate the amount of protein each item contributes to the total. If the total protein estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent protein estimate is obtained.
3. For each item settled on, estimate its energy content in kcal.
4. Add the energy values from all items. Respond with a dictionary object containing the total energy in kcal as follows:
{"total_energy": total energy in kcal for the meal}
For the total energy, respond with just the numeric amount of energy without extra text. If you don't know the answer, set the value of "total_energy" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No protein provided.
3. Estimate energy content in kcal.
Scrambled eggs made with 2 eggs has 204 kcal.
1 toast has 73.2 kcal.
4. Add energy values and respond with total.
Total = (204 + 73.2) = 277.2
Output: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich. The meal contained 6 grams of protein."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate protein contributions per item.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
Matches the provided total (no need to adjust assumptions).
3. Estimate energy content in kcal.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
4. Add energy values and respond with total.
Output: {"total_energy": 202}

Query: "For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge. The meal contained 20g protein."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
2. Estimate protein contributions per item.
Red palm oil: ~0g protein.
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 2g protein per 100g.
729g * 0.02 = 14.6g protein.
Total estimated: 0 + 14.6 = 14.6g protein. This is more than 20% below the stated 20g protein.
Adjustment needed: The porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 2.8g protein per 100g (closer to stiff ugali consistency).
729 * 0.028 = 20.4g protein, which matches the provided 20g context.
So we assume a denser porridge at 2.8g protein/100g.
3. Estimate energy content in kcal.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 61.9 * 8.84 = 547 kcal
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), energy content is typically 127 kcal/100g.
For 729g: 729 * 1.27 kcal = 926 kcal
4. Add energy values and respond with total.
Total = 547 (oil) + 926 (porridge) ≈ 1473 kcal
Output: {"total_energy": 1473}'''


energy_cot_w_fat = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total fat (in grams) is provided, estimate the amount of fat each item contributes to the total. If the total fat estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent fat estimate is obtained.
3. For each item settled on, estimate its energy content in kcal.
4. Add the energy values from all items. Respond with a dictionary object containing the total energy in kcal as follows:
{"total_energy": total energy in kcal for the meal}
For the total energy, respond with just the numeric amount of energy without extra text. If you don't know the answer, set the value of "total_energy" to -1.

Follow the format of the following examples when answering:

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Scrambled eggs made with 2 eggs
- 1 toast
2. No fat provided.
3. Estimate energy content in kcal.
Scrambled eggs made with 2 eggs has 204 kcal.
1 toast has 73.2 kcal.
4. Add energy values and respond with total.
Total = (204 + 73.2) = 277.2
Output: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich. The meal contained 9 grams of fat."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- Half a peanut butter and jelly sandwich
2. Estimate fat contributions per item.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
Matches the provided total (no need to adjust assumptions).
3. Estimate energy content in kcal.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
4. Add energy values and respond with total.
Output: {"total_energy": 202}

Query: "For breakfast, I had 729.0g of ready-to-eat maize porridge. The meal contained 10 grams of fat."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 729g of ready-to-eat maize porridge
2. Estimate fat contributions per item.
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has 1g fat per 100g.
729g * 0.01 = 7.3 grams. This is more than 20% below the stated 10g fat.
Adjustment needed: The maize porridge is likely more concentrated.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 1.5 grams/100g (closer to stiff ugali consistency).
Try 729 * 0.014 = 10.2 grams (1.4g fat per 100g assumption), matching well. So, we assume maize porridge at 1.4g/100g.
3. Estimate energy content in kcal.
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), energy content is typically 127 kcal/100g.
For 729g: 729 * 1.27 kcal = 926 kcal
4. Add energy values and respond with total.
Output: {"total_energy": 926}'''




# ------------------ prompts with "ready-to-eat" added ------------------

prompt_carb2 = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items. 
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, respond with:
{"total_carbohydrates": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}'''


prompt_protein2 = '''For the given query including a meal description, calculate the amount of protein in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items. 
Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer,
respond with:
{"total_protein": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_protein": 7.65}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_protein": 6.4}'''

prompt_fat2 = '''For the given query including a meal description, calculate the amount of fat in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items. 
Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, respond with:
{"total_fat": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_fat": 2.14}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_fat": 9.3}'''

prompt_energy2 = '''For the given query including a meal description, calculate the amount of energy in kilocalories (kcal). If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items.
Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount without extra text. If you don't know the answer, respond with:
{"total_energy": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_energy": 202}'''

prompt_carb_cot2 = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size. Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items. 
3. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 27g carbs.
1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.
1 glass of orange juice has 26g carbs.
So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5
Output: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 2g carbs.
1 toast has 13g carbs.
So the total grams of carbs in the meal = (2 + 13) = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
So the total grams of carbs in the meal = 25.3
Output: {"total_carbohydrates": 25.3}'''

prompt_fat_cot2='''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of fat in grams for the specific serving size. Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items.
3. Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer, set the value of "total_fat" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 1.1g fat.
1 banana has 0.4g fat so half a banana has (0.4*(1/2)) = 0.2g fat.
1 glass of orange juice has 0.84g fat.
So the total grams of fat in the meal = (1.1 + 0.2 + 0.84) = 2.14
Output: {"total_fat": 2.14}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 16.5g fat.
1 toast has 0.985g fat.
So the total grams of fat in the meal = (16.5 + 1) = 17.5
Output: {"total_fat": 17.5}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 18.6g fat so half a peanut butter and jelly sandwich has (18.6*(1/2)) = 9.3g fat.
So the total grams of fat in the meal = 9.3
Output: {"total_fat": 9.3}'''


prompt_energy_cot2 = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of energy in kilocalories (kcal) for the specific serving size. Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items.
3. Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount of kilocalories without extra text. If you don't know the answer, set the value of "total_energy" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 134 kcal.
1 banana has 122 kcal so half a banana has (122*(1/2)) = 61 kcal.
1 glass of orange juice has 117 kcal.
So the total kilocalories in the meal = (134 + 61 + 117) = 312
Output: {"total_energy": 312}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 204 kcal.
1 toast has 73.2 kcal.
So the total kilocalories in the meal = (204 + 73.2) = 277.2
Output: {"total_energy": 277.2}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 404 kcal so half a peanut butter and jelly sandwich has (404*(1/2)) = 202 kcal.
So the total kilocalories in the meal = 202
Output: {"total_energy": 202}'''

prompt_protein_cot2 = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of protein in grams for the specific serving size. Unless explicitly stated, interpret serving sizes as referring to the edible, ready-to-eat forms of meal items.
3. Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer, set the value of "total_protein" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 5.1g protein.
1 banana has 1.3g protein so half a banana has (0.9*(1/2)) = 0.45g protein.
1 glass of orange juice has 1.9g protein.
So the total grams of protein in the meal = (5.1 + 0.45 + 1.9) = 7.65
Output: {"total_protein": 7.65}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 12.7g protein.
1 toast has 2.6g protein.
So the total grams of protein in the meal = (12.7 + 2.6) = 15.3
Output: {"total_protein": 15.3}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 12.8g protein so half a peanut butter and jelly sandwich has (12.8*(1/2)) = 6.4g protein.
So the total grams of protein in the meal = 6.4
Output: {"total_protein": 6.4}'''