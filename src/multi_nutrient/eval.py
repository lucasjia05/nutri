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

from multi_nutrient.prompts import *

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
    path="/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_sub_laya.csv"
    ):
    nb_df = pd.read_csv(path)
    return nb_df.to_dict(orient='records')

def load_data0( # for unprocessed data
    path="/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_sub_laya.csv"
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
    # p = prompt + "\nQuery: " + query + "\nAnswer: "
    # messages = [{"role" : "user", "content" : p}]
    if "gpt-5" in model:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            top_p=top_p,
            n=n,
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
            path="/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_sub_laya.csv",
            test_flag=None,
            save_json=True,
            thresholds = {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0},
            # percentage will likely be more informative, probably pass as another argument later
            results_dir="/data/lucasjia/projects/nutri/results/multi_nutrient/",
            context=None,
            client=None
            ):
    if nutrient.lower() == "combined":
        return run_combined_eval(prompt=prompt, method_name=method_name, model=model, temp=temp, top_p=top_p, mbr=mbr, n=n, path=path, test_flag=test_flag, save_json=save_json, thresholds=thresholds, results_dir=results_dir)
    
    # OPENAI api
    if client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

    data = load_data(path=path)
    if test_flag==True:
        data = data[1:6]
    elif isinstance(test_flag, int) and not isinstance(test_flag, bool):
        k = max(1, min(test_flag, len(data)))
        rng = np.random.default_rng()
        idx = rng.choice(len(data), size=k, replace=False)
        data = [data[i] for i in idx]

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

    return {
        "eval_results" : eval_results,
        "samples_path": output_path if save_json else None
    }

# ------------------ RUN COMBINED EVAL ------------------
def run_combined_eval(
            prompt,
            method_name="cot",
            model="gpt-4o-2024-08-06",
            temp=0.1,
            top_p=0.1,
            mbr=None,
            n=1,
            path="/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_sub_laya.csv",
            test_flag=False,
            save_json=True,
            thresholds = {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0},
            # percentage will likely be more informative, probably pass as another argument later
            results_dir="/data/lucasjia/projects/nutri/results/multi_nutrient/",
            client = None
            ):
    
    # load OPENAI api + load data
    if client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
    data = load_data(path=path)
    if test_flag:
        data = data[1:6]
    elif isinstance(test_flag, int) and not isinstance(test_flag, bool):
        k = max(1, min(test_flag, len(data)))
        rng = np.random.default_rng()
        idx = rng.choice(len(data), size=k, replace=False)
        data = [data[i] for i in idx]
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
    
    return {
        "eval_results" : eval_results,
        "samples_path": output_path if save_json else None
    }


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
    base_context = "The meal contained "
    if len(context) == 1:
        key = context[0]
        value = doc[key]
        if not is_number(value):
            value = process_gt(value)
        value = round(value, 2)
        units = {"energy" : "kcal", 
                 "carb" : "grams of carbohydrates", 
                 "fat" : "grams of fat", 
                 "protein" : "grams of protein"}
        info = f"{value} {units[key]}"
    else:
        raise ValueError("Only 1 context nutrient is supported.")
    
    query = base_query + " " + base_context + info + "."
    return query

def query_add_context0(doc, context):
    base_query = doc['queries']
    base_context = "The following meal contains "
    if len(context) == 1:
        key = context[0]
        value = doc[key]
        if not is_number(value):
            value = process_gt(value)
        value = round(value)
        # value = round(value, 2)
        # unit = "kcal" if key == "energy" else "grams"
        # info = f"{value} {unit}"
        units = {"energy" : "kcal", 
                 "carb" : "grams of carbohydrates", 
                 "fat" : "grams of fat", 
                 "protein" : "grams of protein"}
        info = f"{value} {units[key]}"
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


if __name__ == "__main__":
    # change these params
    nutrient= "carb"
    prompt = prompt_carb_4o_mini2
    method="cot_4o_mini3"
    # model = "gpt-4o-2024-08-06"
    model = "gpt-4o-mini"
    path = "/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_test.csv"
    # path = "/data/lucasjia/projects/nutri/src/multi_nutrient/sub4_metric.csv"
    # path = "/data/lucasjia/projects/nutri/src/multi_nutrient/sub5_natural_language.csv"

    test_flag=False
    thresholds = {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0}
    results_dir = "/data/lucasjia/projects/nutri/results/multi-nutrient/sub1_gpt4_1/"

    temp=0
    top_p=1
    mbr=None
    n=1
    context = None
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
                        thresholds=thresholds,
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
