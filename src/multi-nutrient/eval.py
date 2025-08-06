import pandas as pd
import numpy as np
from scipy import stats
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
# prompts 


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
def get_response(client, prompt, query, method_name, model="gpt-4o-2024-08-06", temp=0.1, top_p=0.1, n=1):
    if "cot" in method_name.lower():
        answer_prompt = "Let's think step by step.\n"
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
        top_p=top_p,
        n=n
    )
    return [choice.message.content for choice in response.choices]



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
    if not is_number(gt):
        gt = process_gt(gt)
    mae = abs(pred - gt)
    mse = mae ** 2
    # debug
    # print(threshold, type(threshold))
    # print(mae, type(mae))
    # print("mae < threshold: ", mae < threshold)

    metrics = {
        "acc": mae < threshold,
        "mae": mae,
        "mse" : mse,
        "answer_rate" : pred != -1
    }
    return metrics



# ------------------ JSON FUNCTIONS ------------------
def construct_json_obj(doc_id, doc, nutrient, context, method_name, prompt, query, model, temp, top_p, mbr, n, response, pred, threshold, result):
    obj = {
        "doc_id" : doc_id,
        "doc" : doc, # metadata relating to query (country, nutrient values, serving_type)
        "nutrient" : nutrient, # carb/fat/energy/protein
        "context" : context,
        "arguments" : { 
            "method" : method_name,
            "prompt" : prompt,
            "query" : query, # meal description also in doc
            "model" : model,
            "temperature" : temp,
            "top_p" : top_p,
            "mbr" : mbr,
            "n" : n,
        },
        "resps" : response,
        "pred" : pred,
        "threshold" : threshold,
        "percent" : False,
        "result" : result,
    }
    return obj

def summary_json(nutrient, context, method, prompt, model, temp, top_p, mbr, n, path, test_flag, save_json, threshold, eval_result):
    summary = {
        "nutrient" : nutrient, # carb/fat/energy/protein
        "context" : context,
        "arguments" : { 
            "method" : method,
            "prompt" : prompt,
            "model" : model,
            "temperature" : temp,
            "top_p" : top_p,
            "mbr" : mbr,
            "n" : n
        },
        "data_path" : path,
        "test_flag" : test_flag,
        "save_json" : save_json,
        "threshold" : threshold,
        "percent" : False,
        "eval_result" : eval_result,
    }
    return summary

def combined_json_obj(doc_id, doc, method_name, prompt, query, model, temp, top_p, mbr, n, response, pred, thresholds, result):
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
            "mbr" : mbr,
            "n" : n,
        },
        "resps" : response,
        "pred" : pred,
        "thresholds" : thresholds,
        "percent" : False,
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
            mbr=None,
            n=1,
            path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv",
            test_flag=False,
            save_json=True,
            thresholds = {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0},
            # percentage will likely be more informative, probably pass as another argument later
            results_dir="/data/lucasjia/projects/nutri/results/multi-nutrient/",
            context=None
            ):
    if nutrient.lower() == "combined":
        return run_combined_eval(prompt=prompt, method_name=method_name, model=model, temp=temp, top_p=top_p, mbr=mbr, n=n, path=path, test_flag=test_flag, save_json=save_json, thresholds=thresholds, results_dir=results_dir)
    
    # OPENAI api
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    data = load_data(path=path)
    if test_flag:
        data = data[1:6]
    threshold = thresholds[nutrient]
    if not mbr:
        n=1

    # initialize
    acc_count = total_mae = total_mse = answers = doc_id = 0
    json_list = []
    
    for doc in tqdm(data):
        if context is not None:
            query = query_add_context(doc, context)
        else:
            query = doc['queries']

        if not mbr:
            responses = get_response(client=client, prompt=prompt, query=query, method_name=method_name, model=model, temp=temp, top_p = top_p)
            pred = clean_output(responses[0], query, method_name, nutrient)
        else:
            preds = []
            responses = get_response(client=client, prompt=prompt, query=query, method_name=method_name, model=model, temp=temp, top_p=top_p, n=n)
            for response in responses:
                preds.append(clean_output(response, query, method_name, nutrient))

            if mbr.lower() == "mae":
                pred = float(np.median(np.array(preds)))
            elif mbr.lower() == "rmse":
                pred = float(np.mean(np.array(preds)))
            elif mbr.lower() == "mode":
                pred = float(stats.mode(np.array(preds), keepdims=True).mode[0])
            else:
                raise ValueError(f"Unsupported mbr option: {mbr}")
        
        result = process_output(doc=doc, pred=pred, threshold=threshold, nutrient=nutrient)
        acc_count += result["acc"]
        total_mae += result["mae"]
        total_mse += result["mse"]
        answers += result["answer_rate"]

        if save_json:
            obj = construct_json_obj(doc_id, doc, nutrient, context, method_name, prompt, query, model, temp, top_p, mbr, n, responses, pred, threshold, result) 
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
        "rmse" : rmse,
        "answer_rate" : ans_rate
    }
    print(f"\nFinal Results - {nutrient} - {method_name}: Acc={acc:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    results_path = results_dir + f"eval_{nutrient}_{method_name}_{timestamp}.json"
    summary = summary_json(nutrient, context, method_name, prompt, model, temp, top_p, mbr, n, path, test_flag, save_json, threshold, eval_results)
    with open(results_path, "w") as f:
        f.write(json.dumps(summary, indent=2))
    print(f"\nSummary results saved to {results_path}")

    return eval_results


# ------------------ RUN COMBINED EVAL ------------------
def run_combined_eval(
            prompt,
            method_name="cot",
            model="gpt-4o-2024-08-06",
            temp=0.1,
            top_p=0.1,
            mbr=None,
            n=1,
            path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv",
            test_flag=False,
            save_json=True,
            thresholds = {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0},
            # percentage will likely be more informative, probably pass as another argument later
            results_dir="/data/lucasjia/projects/nutri/results/multi-nutrient/"
            ):
    
    # load OPENAI api + load data
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    data = load_data(path=path)
    if test_flag:
        data = data[1:6]
    if not mbr:
        n=1

    # initialize
    acc_counts = {"carb":0, "fat":0, "energy":0, "protein":0}
    total_mae = {"carb":0, "fat":0, "energy":0, "protein":0}
    total_mse = {"carb":0, "fat":0, "energy":0, "protein":0}
    answers = {"carb":0, "fat":0, "energy":0, "protein":0}
    doc_id = 0
    json_list = []
    
    for doc in tqdm(data):
        query = doc['queries']
        if not mbr:
            responses = get_response(client=client, prompt=prompt, query=query, method_name=method_name, model=model, temp=temp, top_p = top_p)
            result = {}
            pred = {}
            for nutrient in ["carb", "fat", "energy", "protein"]:
                threshold = thresholds[nutrient]
                pred[nutrient] = clean_output(responses[0], query, method_name, nutrient)
                result[nutrient] = process_output(doc=doc, pred=pred[nutrient], threshold=threshold, nutrient=nutrient)

                acc_counts[nutrient] += result[nutrient]["acc"]
                total_mae[nutrient] += result[nutrient]["mae"]
                total_mse[nutrient] += result[nutrient]["mse"]
                answers[nutrient] += result[nutrient]["answer_rate"]

        else:
            preds = {"carb" : [], "fat" : [], "energy" : [], "protein" : []}
            result = {}
            pred = {}
            responses = get_response(client=client, prompt=prompt, query=query, method_name=method_name, model=model, temp=temp, top_p = top_p, n=n)
            for i in range(n):
                for nutrient in ["carb", "fat", "energy", "protein"]:
                    preds[nutrient].append(clean_output(responses[i], query, method_name, nutrient))
            
            for nutrient in ["carb", "fat", "energy", "protein"]:
                threshold = thresholds[nutrient]
                if mbr.lower() == "mae":
                    pred[nutrient] = float(np.median(np.array(preds[nutrient])))
                elif mbr.lower() == "rmse":
                    pred[nutrient] = float(np.mean(np.array(preds[nutrient])))
                elif mbr.lower() == "mode":
                    pred[nutrient] = float(stats.mode(np.array(preds[nutrient]), keepdims=True).mode[0])
                else:
                    raise ValueError(f"Unsupported mbr option: {mbr}")
            
                result[nutrient] = process_output(doc=doc, pred=pred[nutrient], threshold=threshold, nutrient=nutrient)
                acc_counts[nutrient] += result[nutrient]["acc"]
                total_mae[nutrient] += result[nutrient]["mae"]
                total_mse[nutrient] += result[nutrient]["mse"]
                answers[nutrient] += result[nutrient]["answer_rate"]

        if save_json:
            obj = combined_json_obj(doc_id, doc, method_name, prompt, query, model, temp, top_p, mbr, n, responses, pred, thresholds, result) 
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
            "rmse" : rmse,
            "answer_rate" : ans_rate
        }
        print(f"\nFinal Results - {nutrient} - {method_name}: Acc={acc:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    # save final results to json
    results_path = os.path.join(results_dir, f"eval_combined_{method_name}_{timestamp}.json")
    summary = summary_json("combined", method_name, prompt, model, temp, top_p, mbr, n, path, test_flag, save_json, thresholds, eval_results)
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
    
def query_add_context(doc, context):
    base_query = doc['queries']
    base_context = "The following meal contains "
    if len(context) == 1:
        key = context[0]
        value = doc[key]
        if not is_number(value):
            value = process_gt(value)
        unit = "kcal" if key == "energy" else "grams"
        info = f"{value} {unit}"
    elif len(context) == 2:
        key1, key2 = context
        val1 = doc[key1]
        val2 = doc[key2]
        unit1 = "kcal" if key1 == "energy" else "grams"
        unit2 = "kcal" if key2 == "energy" else "grams"
        info = f"{val1} {unit1} and {val2} {unit2}"
    else:
        raise ValueError("Only 1 or 2 context nutrients are supported.")
    
    query = base_context + info + ". " + base_query
    return query

    

# ------------------ PROMPTS ------------------
# refine prompts, and refine prompt handling in jsons (store prompt path instead of full prompt)
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

prompt_carb_context_energy = '''For the given query including a meal's energy content in kilocalories (kcal) followed by the meal description, calculate the amount of carbohydrates in grams. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer,
respond with: {"total_carbohydrates": -1}.

Query: "The following meal contains 312 kcal. This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: {"total_carbohydrates": 66.5}

Query: "The following meal contains 277.2 kcal. I ate scrambled eggs made with 2 eggs and a toast for breakfast"
Answer: {"total_carbohydrates": 15}

Query: "The following meal contains 202 kcal. Half a peanut butter and jelly sandwich."
Answer: {"total_carbohydrates": 25.3}'''


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

prompt_carb_cot_context_energy3 = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).
2. If the query includes additional known nutrient information (such as total energy in kilocalories), use this information to improve your assessment of portion sizes and food forms (e.g., cooked vs. raw).
3. For each food or beverage item, estimate the amount of carbohydrates in grams based on its serving size.
4. Add up the carbohydrates across all items. If the estimated carbohydrate total seems implausible or inconsistent with known nutrient information, revise your assumptions and recompute. 
5. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "The following meal contains 106.6 kcal. I've got a drink made from 248 grams of oatmeal and water for breakfast."
Answer: Let's think step by step.
The meal consists of a drink made from 248g of oatmeal and water. The total energy content is 106.6 kcal.
Without context, 248g could refer to dry oatmeal. But the total energy (106.6 kcal) is far too low for that — 248g of dry oatmeal would be over 900 kcal.
So it's more likely that this refers to cooked oatmeal, which has about 12g of carbs per 100g.
248g * (12 / 100) = 29.76g carbs
Water contributes 0g carbs.
Consistency check: 
- 29.76g carbs * 4 kcal/g = 119.04 kcal.
- This exceeds the stated energy (106.6 kcal), which suggests the carb content may be slightly overestimated.
Revised estimate: 
Reduce the carb content of oatmeal slightly — use 10.5g carbs per 100g to better reflect the total energy.
248g (10.5 / 100) = 26.04g carbs.
Output: {"total_carbohydrates": 26.04}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 2g carbs.
1 toast has 13g carbs.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "The following meal contains 383.8 kcal. Tonight's dinner is 250 grams of pasta with cream sauce and seafood."
Answer: Let's think step by step.
The meal consists of 250 grams of pasta with cream sauce and seafood. The total energy content is 383.8 kcal.
250 grams of pasta with cream sauce and seafood typically contains about 30g of carbohydrates per 100g.
Therefore, 250g would contain (30 * 2.5) = 75g of carbohydrates.
Consistency check: 
- 75g carbs * 4kcal/g = 300 kcal. That leaves only 83.8 kcal for all fat and protein in a 383.8 kcal meal, which is unlikely given the presence of cream sauce, seafood, and poultry — all rich in fat and protein. 
- So, the assumed carb density is likely too high. 
Revised estimate: 
A more realistic estimate would use 16.5g of carbs per 100g, accounting for more calories from fat and protein.
250 * 0.165 = 41.25g carbs. This leaves 383.8 kcal - (41.25g * 4kcal/g) = 218.3 kcal coming from fat and protein which is reasonable.
Output: {"total_carbohydrates": 41.25}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 full sandwich has 50.6g carbs, so half has 50.6 * 0.5 = 25.3g carbs.
Output: {"total_carbohydrates": 25.3}'''


prompt_carb_cot_context_energy5= '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines (e.g., USDA).
2. If total energy (kcal) is provided, use it to consider multiple plausible forms of each item (e.g. thick vs. thin porridge, dry vs cooked oatmeal) and choose the one most consistent with the total kcal.
3. For each item, estimate its carbohydrate content in grams.
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
4. Add carbohydrate values and respond with total.
Total = 2 + 13 = 15
Output: {"total_carbohydrates": 15}

Query: "The following meal contains 106.6 kcal. I've got a drink made from 248 grams of oatmeal and water for breakfast."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- A drink made from 248g of oatmeal and water
The total energy content is 106.6 kcal.
2. Use total energy (106.6 kcal) to guide assumptions.
248g of dry oats would contain far more than 900 kcal, so this must refer to a very diluted mixture—either cooked oatmeal or an oat-based drink.
This energy density = 106.6 kcal / 248g ≈ 0.43 kcal/g, indicating a highly water-diluted product.
This is consistent with an oat drink.
3. Estimate carbohydrate content in grams.
Cooked oatmeal (or a thin oat drink) typically has around 10-12g carbs per 100g.
Using a conservative mid-range value of 10.5g/100g:
248g * 0.105 = 26.04g carbs
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 26.04}

Query: "The following meal contains 1470 kcal. For breakfast, I had 61.9g of red palm oil and 729.0g of ready-to-eat maize porridge."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 61.9g of red palm oil
- 729g of ready-to-eat maize porridge
The total energy content is 1470 kcal.
2. Use total energy (1470 kcal) to guide assumptions.
Palm oil has 0g carbohydrates and is almost entirely fat (~9 kcal/g), so:
61.9g * 9 = 557.1 kcal from fat
That leaves 1470 - 557.1 = 912.9 kcal from the porridge.
Energy density of the porridge = 912.9 kcal / 729g ≈ 1.25 kcal/g, indicating a thicker porridge (not watery).
3. Estimate carbohydrate content in grams.
Thicker maize porridges can have 25-30g of carbs per 100g.
Using 28.4g/100g as a realistic estimate:
729g * 0.284 = 207.04g carbs
4. Add carbohydrate values and respond with total.
Output: {"total_carbohydrates": 207.04}

Query: "The following meal contains 383.8 kcal. Tonight's dinner is 250 grams of pasta with cream sauce and seafood."
Answer: Let's think step by step.
1. Identify each food or beverage item and its serving size.
- 250 grams of pasta with cream sauce and seafood
The total energy content is 383.8 kcal.
2. Use total energy (383.8 kcal) to guide assumptions.
We're given that the total meal weighs 250g and provides 383.8 kcal, so:
Energy density = 383.8 kcal / 250g ≈ 1.535 kcal/g
This suggests a relatively low-fat, high-water content dish (e.g., light cream sauce, moderate seafood).
High-fat sauces or large seafood portions would push the kcal/g higher (above 2).
This energy density is consistent with a moderate portion of pasta, a bit of seafood (low fat), and a modest cream sauce.
3. Estimate carbohydrate content in grams.
Pasta is the main carbohydrate source here.
Seafood has minimal carbs. Cream sauce may have some (milk, flour, etc.), but also fat.
We'll estimate the dish as follows, based on weight proportions and typical recipes:
Pasta (cooked): ~65% of the weight makes up 250g * 0.65 = 162.5g
Cooked pasta has ~24g carbs per 100g so 162.5g * 0.24 = 39g carbs
Cream sauce: ~25% of the weight makes up 62.5g
Light cream sauce (mostly cream + milk + flour): ~5g carbs per 100g so 62.5g contains ~3.1g carbs
Seafood: ~10% of the weight makes up 25g
Minimal carbs ~0g
4. Add carbohydrate values and respond with total.
Estimated total carbs = 39 + 3.1 = 42.1 g.
Output: {"total_carbohydrates": 42.1}'''


prompt_carb_cot_context_energy6= '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams.
4. Add the carbohydrates from all items. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the meal}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.'''


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

prompt_carb_cot_context_energy8 = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Based on these items, classify the meal's regional context by determining whether it is best characterized as primarily Western (e.g., U.S./European) or non-Western/traditional (e.g., African, Asian, Latin American). Use USDA food composition data for Western meals, and FAO/WHO or local sources for non-Western/traditional meals.
2. If total energy (kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g., concentration, dry vs. wet vs. dehydrated). Repeat this step until a consistent energy estimate is obtained.
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
This is best characterized as a Western meal.
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
This is best characterized as a Western meal.
2. Estimate energy contributions per item.
248g of oatmeal: Approx. 70 kcal / 100g
248g * 0.7 kcal/g = 173.6 kcal. This is far above 106.6 kcal, so something is off.
Reconsider oatmeal: 248g may be referring to the entire drink, rather than the amount of dry oatmeal.
248g of oatmeal drink: Approx. 45 kcal / 100g
248 * 0.45 kcal/g = 111.6 kcal - acceptable.
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
This is best characterized as a non-Western/traditional meal, likely African in context, based on the ingredients (e.g., palm oil, maize porridge).
2. Estimate energy contributions per item.
Red palm oil: Energy content is about 884 kcal per 100g.
For 61.9g: 884 * 0.619 = 547 kcal
Ready-to-eat maize porridge: This depends heavily on its water content. A typical maize porridge (thin) has ~50-60 kcal per 100g.
729g * 0.55 = 400.95 kcal
Total estimated: 547 + 401 = 948 kcal. This is far below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali or polenta consistency).
Try 729 * 1.27 = 926.8 kcal (127 kcal/100g assumption)
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

prompt_carb_cot_context_energy9 = '''For the given query including a meal description, think step by step as follows:
1. Identify each food or beverage item and its serving size. If no size is given, assume one standard serving based on common guidelines. Use USDA references for meals typically associated with Western/U.S. diets. For other regional or traditional meals, prefer FAO/WHO food composition data.
2. If total energy (kcal) is provided, estimate the amount of energy each item contributes to the total. If the total energy estimated differs by more than 20%, reconsider the assumed form of every single relevant item (e.g. concentration, dry vs wet vs dehydrated). Repeat this step until a consistent energy estimate is obtained.
3. For each item settled on, estimate its carbohydrate content in grams based on its serving size.
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
729g * 0.55 = 400.95 kcal
Total estimated: 547 + 401 = 948 kcal. This is far below the stated 1470 kcal.
Adjustment needed: Either the maize porridge is more concentrated or oil quantity/energy is wrong.
Let's assume a thicker maize porridge or less water content. Dense cooked maize porridge can have up to 160 kcal/100g (closer to stiff ugali or polenta consistency).
Try 729 * 1.27 = 926.8 kcal (127 kcal/100g assumption)
New total: 547 (oil) + 927 (porridge) ≈ 1474 kcal, matching well. So, we'll assume maize porridge at 127 kcal/100g.
3. Estimate carbohydrate content in grams.
Red palm oil: 0g carbs (pure fat)
Thick maize porridge: Based on FAO data for stiff maize porridge (e.g. ugali or sadza), carb content is typically ~27-28g per 100g.
We'll use 28g carbs per 100g.
729g * 0.28 = 204.12g of carbohydrates
4. Add carbohydrate values and respond with total.
0g (oil) + 204.12g (porridge) = 204.12g
Output: {"total_carbohydrates": 204.12}'''



if __name__ == "__main__":
    # change these params
    nutrient="carb"
    prompt = prompt_carb_cot_context_energy7
    method="CoT"
    model = "gpt-4o-2024-08-06"
    # path="/data/lucasjia/projects/nutri/src/multi-nutrient/nb_v2_sub_laya.csv"
    # path = "/data/lucasjia/projects/nutri/src/multi-nutrient/sub4_metric.csv"
    path = "/data/lucasjia/projects/nutri/src/multi-nutrient/sub5_natural_language.csv"

    test_flag=True
    thresholds = {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0}
    # results_dir = "/data/lucasjia/projects/nutri/results/multi-nutrient/sub1w2/"
    # results_dir = "/data/lucasjia/projects/nutri/results/multi-nutrient/sub4/"
    results_dir = "/data/lucasjia/projects/nutri/results/multi-nutrient/sub5"

    temp=0.1
    top_p=0.1
    mbr=None
    n=1
    context = ["energy"]

    results = run_eval( prompt=prompt, 
                        nutrient=nutrient, 
                        method_name=method, 
                        model=model,
                        path=path,
                        temp=temp,
                        top_p=top_p,
                        mbr=mbr,
                        n=n,
                        test_flag=test_flag,
                        save_json=True,
                        thresholds = thresholds,
                        results_dir=results_dir,
                        context=context
                        )

    # results = run_combined_eval(prompt=prompt, 
    #                             method_name=method, 
    #                             model=model,
    #                             path=path,
    #                             temp=0.1,
    #                             top_p=0.1,
    #                             test_flag=test_flag,
    #                             save_json=True,
    #                             thresholds = thresholds,
    #                             results_dir=results_dir
    #                             )
