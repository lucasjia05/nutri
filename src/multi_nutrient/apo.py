from multi_nutrient.prompts import *
from multi_nutrient.eval import *
from multi_nutrient.analysis import *
from multi_nutrient.apo_prompts import *

# stage 1: eval + gradients
# starting prompt (also variations)
# evaluate
# parse highest error examples
# gradient prompt + (starting prompt, highest error examples)
# obtain gradients for each example?
# or obtain gradient for a batch of examples? which is cheaper?

# stage 2: prompt optimization
# optimization prompt + (starting prompt, highest error examples, gradients)
# for each variation prompt, rewrite to fix issues of gradients
# paraphrase prompt (paraphrase to diversify wording)
# beam search
# bandit selection



# ------------------ STAGE 1: EVAL + GRADIENTS ------------------
def get_topk_errors(path, nutrient="carb", k=8):
    df = pd.read_json(path, lines=True)
    doc_df = pd.json_normalize(df["doc"])
    result_df = pd.json_normalize(df["result"])
    flat = pd.concat([df.drop(columns=["doc", "result"]), doc_df, result_df], axis=1)
    top_k = flat.sort_values(by="mae", ascending=False).head(k)

    errors = []
    for _, row in top_k.iterrows():
        err = {
            "doc_id" : row["doc_id"],
            "query" : row["queries"],
            "food_items" : row["description"],
            "per_item_ground_truth" : row[nutrient], # list of nutrient values corresponding with food_items
            "response_text" : row["resps"][0], # list of 1 resp, indexed to first?
            "prediction" : round(row["pred"],2),
            "abs_err" : round(row["mae"],2)
        }
        errors.append(err)
    return errors


def get_gradients(errors, gradient_prompt, gradient_model="gpt-4o-mini"):
    pass


def apo_get_gradients(
        eval_params,
        num_errors,
        gradient_model="gpt-4o-mini", 
        ):
    # eval
    results = run_eval( prompt=eval_params["prompt"], 
                        nutrient=eval_params["nutrient"], 
                        method_name=eval_params["method"], 
                        model=eval_params["model"],
                        path=eval_params["path"],
                        temp=eval_params["temp"],
                        top_p=eval_params["top_p"],
                        test_flag=eval_params["test_flag"],
                        save_json=True,
                        thresholds=eval_params["threshold"],
                        results_dir=eval_params["results_dir"],
                        context=eval_params["context"]
                        )
    # filter errors



    # gradients
    gradients = []


    return gradients







if __name__ == "__main__":
    # models

    task_model = "gpt-4o-2024-08-06"
    gradient_model = "gpt-4o-mini"
    rewrite_model = "gpt-4o-2024-08-06"
    paraphrase_model = "gpt-4o-mini"


    test_flag = 32
    eval_params = {
        "nutrient" : "carb",
        "prompt" : prompt_carb_cot,
        "method" : "CoT",
        "model" : task_model,
        "path" : "/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_sub3_laya.csv",
        "test_flag" : test_flag,
        "thresholds" : {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0},
        "results_dir" : "/data/lucasjia/projects/nutri/results/APO_experiments/",
        "temp" : 0.1,
        "top_p" : 0.1,
        "context" : None,
    }

    # print(get_topk_errors(path="/data/lucasjia/projects/nutri/results/multi-nutrient/sub1_rotations/samples_carb_base_20250821_205205.jsonl"))
    # print(get_gradients())