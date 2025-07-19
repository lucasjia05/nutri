import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import os
import re
import ast
import json
import hashlib


# FUNCTIONS
# load_data - csv to list of dictionaries
# get_response - prompt through openai api, return response
# clean_output - extract numerical prediction
# process_output - return acc, mae, mse as dict for single output
# json functions - info for each response and overall evaluation summary
# run_eval - full eval pipeline
# run_combined_eval - full eval pipeline for multiple nutrients

# utils
# prompts - carb,  TO ADD: fat, calorie, protein, joint


# ------------------ LOAD DATA ------------------
def load_data( # for if data is alr processed
    path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv"
    ):
    nb_df = pd.read_csv(path)
    return nb_df.to_dict(orient='records')

def load_data0( # for unprocessed data
    path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv"
    ):
    nb_df = pd.read_csv(path)
    for nutrient in ["carb", "protein", "fat", "energy"]:
        nb_df[nutrient] = nb_df[nutrient].apply(process_gt)
    return nb_df.to_dict(orient='records')


# ------------------ GET RESPONSE ------------------
def get_response(client, prompt, query, method_name, model="gpt-4o-2024-08-06", temp=0.1, top_p=0.1):
    # queries are pretty slow
    if "cot" in method_name.lower():
        answer_prompt = "Let's think step by step."
    else:
        answer_prompt = ""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f'Query: {query}\nAnswer: {answer_prompt}'}      
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        top_p=top_p
    )
    return response.choices[0].message.content



# ------------------ CLEAN OUTPUT ------------------
# copied from nutribench github
def clean_output(raw_output, query, method_name, nutrition_name):
    if "cot" in method_name.lower():
        # discard all output which is part of the reasoning process
        splits = raw_output.split("Output:")
        if len(splits) > 1: # split into reasoning and answer part
            raw_output = splits[1]
    raw_output = raw_output.strip()
    # print(f"Raw output: {raw_output}")
    
    # match this pattern to find the total carb estimate
    if nutrition_name == 'fat':
        pattern = r'["\']\s*total_fat["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'protein':
        pattern = r'["\']\s*total_protein["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'energy':
        pattern = r'["\']\s*total_energy["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'carb':
        pattern = r'["\']\s*total_carbohydrates["\']:\s*(?:["\']?(-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)["\']?|\[(-?[0-9]+(?:\.[0-9]*)?(?:,\s*-?[0-9]+(?:\.[0-9]*)?)*)\])'
    else:
        raise NotImplementedError
    
    match = re.search(pattern, raw_output)
    if match:
        if match.group(1):
            pred_carbs = match.group(1) # extract the numeric part
            if is_number(pred_carbs):
                return float(pred_carbs)
            else:
                # check if output is a range
                pred_carbs_list = pred_carbs.split('-')
                if len(pred_carbs_list) == 2 and is_number(pred_carbs_list[0]) and is_number(pred_carbs_list[1]):
                    p0 = float(pred_carbs_list[0])
                    p1 = float(pred_carbs_list[1])
                    return (p0+p1)/2.0
                else:
                    print(f"EXCEPTION AFTER MATCHING")
                    print(f"Matched output: {raw_output}")
                    print(f"Query: {query}")
                    return -1
        elif match.group(2):
            try:
                pred_carbs_list = match.group(2).split(',')
                p0 = float(pred_carbs_list[0])
                p1 = float(pred_carbs_list[1])
                return (p0+p1)/2.0
            except:
                print(f"EXCEPTION AFTER MATCHING")
                print(f"Matched output: {raw_output}")
                print(f"Query: {query}")
                return -1
    else:
        if is_number(raw_output):
            return float(raw_output)
        else:
            print(f"EXCEPTION")
            print(f"Matched output: {raw_output}")
            print(f"Query: {query}")
            return -1
        

# ------------------ PROCESS OUTPUT ------------------
def process_output(doc, pred, threshold=7.5, nutrient="carb"):
    gt = doc[nutrient]
    mae = abs(pred - gt)
    mse = mae ** 2

    metrics = {
        "acc": mae < threshold,
        "mae": mae,
        "mse" : mse,
        "answer_rate" : pred != -1
    }
    return metrics



# ------------------ JSON FUNCTIONS ------------------
def construct_json_obj(doc_id, doc, nutrient, method_name, prompt, query, model, temp, top_p, response, pred, threshold, result):
    # not the same as spencer sent me but simpler imo

    obj = {
        "doc_id" : doc_id,
        "doc" : doc, # metadata relating to query (country, nutrient values, serving_type)
        "nutrient" : nutrient, # carb/fat/energy/protein
        "arguments" : { 
            "method" : method_name,
            "prompt" : prompt,
            "query" : query, # meal description also in doc
            "model" : model,
            "temperature" : temp,
            "top_p" : top_p,
        },
        "resps" : response,
        "pred" : pred,
        "threshold" : threshold,
        "result" : result,
    }
    return obj

def summary_json(nutrient, method, prompt, model, temp, top_p, path, test_flag, save_json, threshold, eval_result):
    summary = {
        "nutrient" : nutrient, # carb/fat/energy/protein
        "arguments" : { 
            "method" : method,
            "prompt" : prompt,
            "model" : model,
            "temperature" : temp,
            "top_p" : top_p,
        },
        "data_path" : path,
        "test_flag" : test_flag,
        "save_json" : save_json,
        "threshold" : threshold,
        "eval_result" : eval_result,
    }
    return summary

def combined_json_obj(doc_id, doc, method_name, prompt, query, model, temp, top_p, response, pred, thresholds, result):
    obj = {
        "doc_id" : doc_id,
        "doc" : doc,
        "nutrient" : "combined",
        "arguments" : { 
            "method" : method_name,
            "prompt" : prompt,
            "query" : query,
            "model" : model,
            "temperature" : temp,
            "top_p" : top_p,
        },
        "resps" : response,
        "pred" : pred,
        "thresholds" : thresholds,
        "result" : result,
    }
    return obj

# ------------------ RUN EVAL ------------------
def run_eval(
            prompt,
            nutrient="carb",
            method_name="cot",
            model="gpt-4o-2024-08-06",
            temp=0.1,
            top_p=0.1,
            path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv",
            test_flag=False,
            save_json=True,
            results_dir="/data/lucasjia/projects/nutri/results/multi-nutrient/"
            ):
    # OPENAI api
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    data = load_data(path=path)
    if test_flag:
        data = data[0:5]
    # percentage will likely be more informative, probably pass as argument later
    thresholds = {"carb" : 7.5, "protein" : 2.5, "fat" : 2.5, "energy" : 50} 
    threshold = thresholds[nutrient]

    # initialize
    acc_count = total_mae = total_mse = answers = doc_id = 0
    json_list = []
    
    for doc in tqdm(data):
        query = doc['queries']
    
        response = get_response(client=client, prompt=prompt, query=query, method_name=method_name, model=model, temp=temp, top_p = top_p)
        pred = clean_output(response, query, method_name, nutrient)
        result = process_output(doc=doc, pred=pred, threshold=threshold, nutrient=nutrient)

        acc_count += result["acc"]
        total_mae += result["mae"]
        total_mse += result["mse"]
        answers += result["answer_rate"]

        if save_json:
            obj = construct_json_obj(doc_id, doc, nutrient, method_name, prompt, query, model, temp, top_p, response, pred, threshold, result) 
            json_list.append(obj)
            doc_id += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_json:
        output_path = results_dir + f"samples_{nutrient}_{method_name}_{timestamp}.jsonl"
        with open(output_path, "w") as f:
            for obj in json_list:
                f.write(json.dumps(obj) + "\n")
        print(f"\nPer-entry results saved to {output_path}")

    # overall metrics
    acc = acc_count / len(data)
    mae = total_mae / len(data)
    rmse = (total_mse / len(data)) ** 0.5
    ans_rate = answers / len(data)
    eval_results = {
        "acc": acc,
        "mae": mae,
        "mse" : rmse,
        "answer_rate" : ans_rate
    }
    print(f"\nFinal Results - {nutrient} - {method_name}: Acc={acc:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    results_path = results_dir + f"eval_summary_{nutrient}_{method_name}_{timestamp}.json"
    summary = summary_json(nutrient, method_name, prompt, model, temp, top_p, path, test_flag, save_json, threshold, eval_results)
    with open(results_path, "w") as f:
        f.write(json.dumps(summary, indent=2))
    print(f"\nSummary results saved to {results_path}")

    return eval_results


# ------------------ RUN COMBINED EVAL ------------------
# almost same code as run_eval
def run_combined_eval(
            prompt,
            method_name="cot",
            model="gpt-4o-2024-08-06",
            temp=0.1,
            top_p=0.1,
            path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv",
            test_flag=False,
            save_json=True,
            results_dir="/data/lucasjia/projects/nutri/results/multi-nutrient/"
            ):
    # load OPENAI api + load data
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    data = load_data(path=path)
    if test_flag:
        data = data[0:5]

    # initialize
    acc_counts = {"carb":0, "fat":0, "energy":0, "protein":0}
    total_mae = {"carb":0, "fat":0, "energy":0, "protein":0}
    total_mse = {"carb":0, "fat":0, "energy":0, "protein":0}
    answers = {"carb":0, "fat":0, "energy":0, "protein":0}
    doc_id = 0
    json_list = []
    # percentage will likely be more informative, probably pass as argument later
    thresholds = {"carb" : 7.5, "protein" : 2.5, "fat" : 2.5, "energy" : 50} 
    
    for doc in tqdm(data):
        query = doc['queries']
        response = get_response(client=client, prompt=prompt, query=query, method_name=method_name, model=model, temp=temp, top_p = top_p)
        result = {}
        pred = {}
        for nutrient in ["carb", "fat", "energy", "protein"]:
            threshold = thresholds[nutrient]
            pred[nutrient] = clean_output(response, query, method_name, nutrient)
            result[nutrient] = process_output(doc=doc, pred=pred[nutrient], threshold=threshold, nutrient=nutrient)

            acc_counts[nutrient] += result[nutrient]["acc"]
            total_mae[nutrient] += result[nutrient]["mae"]
            total_mse[nutrient] += result[nutrient]["mse"]
            answers[nutrient] += result[nutrient]["answer_rate"]

        if save_json:
            obj = combined_json_obj(doc_id, doc, method_name, prompt, query, model, temp, top_p, response, pred, thresholds, result) 
            json_list.append(obj)
            doc_id += 1

    # save samples in json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_json:
        output_path = os.path.join(results_dir, f"samples_combined_{method_name}_{timestamp}.jsonl")
        with open(output_path, "w") as f:
            for obj in json_list:
                f.write(json.dumps(obj) + "\n")
        print(f"\nPer-entry results saved to {output_path}")

    # overall metrics
    n_samples = len(data)
    eval_results = {}
    for nutrient in ["carb", "fat", "energy", "protein"]:
        acc = acc_counts[nutrient] / n_samples
        mae = total_mae[nutrient] / n_samples
        rmse = (total_mse[nutrient] / n_samples) ** 0.5
        ans_rate = answers[nutrient] / n_samples
        eval_results[nutrient] = {
            "acc": acc,
            "mae": mae,
            "mse" : rmse,
            "answer_rate" : ans_rate
        }
        print(f"\nFinal Results - {nutrient} - {method_name}: Acc={acc:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    # save final results to json
    results_path = os.path.join(results_dir, f"eval_summary_combined_{method_name}_{timestamp}.json")
    summary = summary_json("combined", method_name, prompt, model, temp, top_p, path, test_flag, save_json, thresholds, eval_results)
    with open(results_path, "w") as f:
        f.write(json.dumps(summary, indent=2))
    print(f"\nSummary results saved to {results_path}")

    return eval_results




# ------------------ UTILS ------------------
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_gt(x):
    if isinstance(x, str):
        values = ast.literal_eval(x)
        return sum(values)
    elif isinstance(x, (int, float)):
        return x
    else:
        return float(x)
    
# not used (for obtain_json_obj() function)
def get_hash(v):
    return hashlib.sha256(v.encode('utf-8')).hexdigest()
    

# ------------------ PROMPTS ------------------
# refine prompts, and refine prompt handling in jsons (store prompt path instead of full prompt)
prompt_carb = '''For the given query including a meal description, calculate the amount of carbohydrates in grams. If the serving size of any item in the
meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{{"total_carbohydrates": total grams of carbohydrates for the serving}}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer,
respond with:
{{"total_carbohydrates": -1}}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {{"total_carbohydrates": 66.5}}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {{"total_carbohydrates": 15}}

Query: "Half a peanut butter and jelly sandwich."
Answer: {{"total_carbohydrates": 25.3}}'''


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

prompt_protein_temp = '''For the given query including a meal description, calculate the amount of protein in grams. If the serving size of any item in the
meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total protein in grams as follows:
{"total_protein": total grams of protein for the serving}
For the total protein, respond with just the numeric amount of protein without extra text. If you don't know the answer,
respond with:
{"total_protein": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_protein": 7.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_protein": 16}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_protein": 6.5}'''

prompt_fat_temp = '''For the given query including a meal description, calculate the amount of fat in grams. If the serving size of any item in the
meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total fat in grams as follows:
{"total_fat": total grams of fat for the serving}
For the total fat, respond with just the numeric amount of fat without extra text. If you don't know the answer,
respond with:
{"total_fat": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_fat": 6.4}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_fat": 21}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_fat": 9.8}'''

prompt_energy_temp = '''For the given query including a meal description, calculate the amount of energy in kilocalories (kcal). If the serving size of any item in the
meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total energy in kilocalories as follows:
{"total_energy": total kilocalories for the serving}
For the total energy, respond with just the numeric amount without extra text. If you don't know the answer,
respond with:
{"total_energy": -1}.

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_energy": 330}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_energy": 320}

Query: "Half a peanut butter and jelly sandwich."
Answer: {"total_energy": 280}'''

prompt_combined_temp = '''For the given query including a meal description, calculate the total amounts of carbohydrates (grams), protein (grams), fat (grams), and energy (kilocalories). 
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
  "total_protein": 7.5,
  "total_fat": 6.4,
  "total_energy": 330
}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: {
  "total_carbohydrates": 15,
  "total_protein": 16,
  "total_fat": 21,
  "total_energy": 320
}

Query: "Half a peanut butter and jelly sandwich."
Answer: {
  "total_carbohydrates": 25.3,
  "total_protein": 6.5,
  "total_fat": 9.8,
  "total_energy": 280
}'''


if __name__ == "__main__":
    # about $1 per 
    nutrient="carb"
    prompt = prompt_carb

    results = run_eval( prompt=prompt, 
                        nutrient=nutrient, 
                        method_name="base", 
                        model="gpt-4o-2024-08-06",
                        path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv",
                        temp=0.1,
                        top_p=0.1,
                        test_flag=False,
                        save_json=True,
                        results_dir="/data/lucasjia/projects/nutri/results/multi-nutrient/"
                        )

    # results = run_combined_eval(prompt=prompt, 
    #                             method_name="base", 
    #                             model="gpt-4o-2024-08-06",
    #                             path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv",
    #                             temp=0.1,
    #                             top_p=0.1,
    #                             test_flag=True,
    #                             save_json=True,
    #                             results_dir="/data/lucasjia/projects/nutri/results/multi-nutrient/"
    #                             )
