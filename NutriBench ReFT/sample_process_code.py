import pandas as pd
import re
import json

cot_prompt = """For the given query including a meal description, think step by step as follows:
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
Output: {"total_carbohydrates": 25.3}"""


def get_cot_output(food_components, carb_value):
    """
    Get output in the format of CoT prompting.
    Always convert to a list internally, then check if there's only one item.
    """
        
    # 1) Convert single strings into 1-element lists for uniform handling
    if isinstance(food_components, str):
        try:
            food_components = eval(food_components)
        except:
            pass
        if isinstance(food_components, str):
            food_components = [food_components]
        elif isinstance(food_components, list):
            pass
        else:
            raise ValueError(f"Invalid food_components: {food_components}")
    if isinstance(carb_value, str):
        carb_value = eval(carb_value)
        if isinstance(carb_value, (float, int)):
            carb_value = [carb_value]
    elif isinstance(carb_value, (float, int)):
        carb_value = [carb_value]

    # 2) If there's exactly 1 item, generate a short chain-of-thought
    if len(food_components) == 1:
        food_components = food_components[0]
        crb = carb_value[0]

        # Format carbs to two decimals
        crb_str = f"{crb:.2f}"

        output = (
            f"The meal consists of {food_components}.\n"
            f"{food_components.capitalize()} has {crb_str}g carbs.\n"
            f"So the total carbs in the meal = {crb_str}.\n"
            f"Output: {{\"total_carbohydrates\": {crb_str}}}"
        )
        return output

    # 3) Otherwise, multiple items => more detailed breakdown
    output_2 = (
        "The meal consists of "
        + ", ".join(
            f"{food_component}" for food_component in food_components
        )
        + ".\n"
    )

    sentences = []
    for food_component, carb in zip(food_components, carb_value):
        crb_str = f"{carb:.2f}"
        sentences.append(f"{food_component.capitalize()} has {crb_str}g carbs.")

    output_3 = "\n".join(sentences) + "\n"

    # Show each carb_value in the formula with two decimals
    formula_components = [f"{c:.2f}" for c in carb_value]
    formula = " + ".join(formula_components)

    # Format the total to two decimals
    total = sum(carb_value)
    total_str = f"{total:.2f}"

    output_4 = (
        f"So the total grams of carbs in the meal = ({formula}) = {total_str}.\n"
        f"Output: {{\"total_carbohydrates\": {total_str}}}"
    )

    output = output_2 + output_3 + output_4
    return output


def create_json_objects(system_message, user_messages, assistant_messages, model='openai'):
    json_objects = []
    if model == 'openai':
        for i in range(len(user_messages)):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_messages[i]},
                {"role": "assistant", "content": assistant_messages[i]}
            ]
            json_objects.append({"messages": messages})
    elif model == 'gemma':
        for i in range(len(user_messages)):
            messages = [
                {"role": "user", "content":system_message + '\n\n' + user_messages[i]},
                {"role": "assistant", "content": assistant_messages[i]}
            ]
            json_objects.append({"messages": messages})
    return json_objects




if __name__ == "__main__":

    data_df = pd.read_csv('/data/lucasjia/projects/NutriBench ReFT/train_v2.csv')

    save_path = "train_v2.jsonl"
    model = "gemma"
    system_message = cot_prompt

    data_df['cot_output'] = data_df.apply(lambda x: get_cot_output(x['components'], x['carb']), axis=1)

    user_messages = data_df['query'].apply(lambda x: f"Query: {x}\nAnswer:").to_list()
    assistant_messages = data_df['cot_output'].to_list()

    json_objects = create_json_objects(system_message, user_messages, assistant_messages, model=model)

    with open(save_path, 'w') as file:
        for obj in json_objects:
            file.write(json.dumps(obj) + '\n')
            
    print(f"SAVED {len(json_objects)} MESSAGES")

    print('done')

