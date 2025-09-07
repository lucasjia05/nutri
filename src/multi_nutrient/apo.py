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


def get_gradient(client, errors, base_prompt, num_feedbacks=1, gradient_model="gpt-4o-mini", temp=0.7, top_p=0.9):
    error_string = format_errors(errors)
    gradient_prompt = format_gradient_prompt(base_prompt, error_string, num_feedbacks)
    
    messages = [
        {"role": "system", "content": "You are an expert in prompt engineering and diagnosing prompt design flaws."},
        {"role": "user", "content": gradient_prompt}
    ]
    if "gpt-5" in gradient_model:
        response = client.chat.completions.create(
            model=gradient_model,
            messages=messages,
        )

    else:
        response = client.chat.completions.create(
            model=gradient_model,
            messages=messages,
            temperature=temp,
            top_p=top_p,
        )

    return response.choices[0].message.content


def apo_get_gradients(
        client,
        eval_params,
        top_k=24,
        group_size=2,
        num_feedbacks=1,
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
                        thresholds=eval_params["thresholds"],
                        results_dir=eval_params["results_dir"],
                        context=eval_params["context"],
                        client=client
                        )
    # filter errors
    errors = get_topk_errors(path=results["samples_path"], nutrient=eval_params["nutrient"], k=top_k)
    
    num_batches = (len(errors) + group_size - 1) // group_size or 1
    batches = [[] for _ in range(num_batches)]
    for i, err in enumerate(errors):
        batches[i % num_batches].append(err)

    # gradients
    gradients = []
    for batch in tqdm(batches):
        grad = get_gradient(
            client=client,
            errors=batch,
            base_prompt=eval_params["prompt"],
            num_feedbacks=num_feedbacks,
            gradient_model=gradient_model,
            temp=0.7,
            top_p=0.9,
        )
        gradients.append(grad)
    return gradients


# ------------------ HELPER FUNCTIONS ------------------
def format_gradient_prompt(base_prompt, error_string, num_feedbacks=1):
    user_text = (
        "I'm trying to write a zero-shot classifier prompt.\n"
        'My current prompt is:\n"{prompt}"\n'
        "But this prompt gets the following examples wrong:\n"
        "{error_string}\n"
        "give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.\n"
        "Wrap each reason with <START> and <END>"
    ).format(prompt=base_prompt, error_string=error_string, num_feedbacks=num_feedbacks)

    return user_text


def format_errors(errors):
    blocks = []
    for i, e in enumerate(errors, start=1):
            blocks.append(f"""Example {i} (doc_id: {e['doc_id']})
Query: "{e['query']}"
Items: {e['food_items']}
GT per item: {e['per_item_ground_truth']}
Model predicted total: {e['prediction']} | AbsErr: {e['abs_err']}
Resp: "{e['response_text']}" """.strip()
                    )
    return "\n\n".join(blocks)





if __name__ == "__main__":

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # models
    task_model = "gpt-4o-2024-08-06"
    gradient_model = "gpt-4o-mini"
    # rewrite_model = "gpt-4o-2024-08-06"
    # paraphrase_model = "gpt-4o-mini"

    base_prompt = prompt_carb_cot

    test_flag = False
    eval_params = {
        "nutrient" : "carb",
        "prompt" : prompt_carb_cot,
        "method" : "CoT",
        "model" : task_model,
        "path" : "/data/lucasjia/projects/nutri/src/multi_nutrient/nb_v2_sub_laya.csv",
        "test_flag" : test_flag,
        "thresholds" : {"carb" : 7.5, "protein" : 2.0, "fat" : 2.5, "energy" : 50.0},
        "results_dir" : "/data/lucasjia/projects/nutri/results/APO_experiments/",
        "temp" : 0.1,
        "top_p" : 0.1,
        "context" : None,
    }


    gradients = apo_get_gradients(
        client=client,
        eval_params=eval_params,
        top_k=8,
        group_size=2,
        num_feedbacks=1,
        gradient_model="gpt-4o-mini", 
        )
    print(gradients)
    # print(get_topk_errors(path="/data/lucasjia/projects/nutri/results/multi-nutrient/sub1_rotations/samples_carb_base_20250821_205205.jsonl"))
    # print(get_gradients())
    # print(format_gradient_prompt("base_prompt", "error_string", num_feedbacks=2))