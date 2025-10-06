import os
from tqdm import tqdm
import time
import json
import argparse

import scorers
import tasks
import predictors
import optimizers
import evaluators


def get_task_class(task_name):
    if task_name == 'nutribench':
        return tasks.NutrientTask
    else:
        raise Exception(f'Unsupported task: {task_name}')


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')



def get_scorer(scorer):
    if scorer == 'mae':
        return scorers.CachedMAEScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():

    # task parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='nutribench')
    parser.add_argument('--nutrient', default='carb')
    # parser.add_argument('--method', default='base')
    parser.add_argument('--data_dir', default='data/nutribench_v2')
    parser.add_argument('--prompts', default='prompts/nutri_base.md')
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--task_model', default="gpt-4o-mini", type=str)
    parser.add_argument('--gradient_model', default="gpt-4o-mini", type=str)
    parser.add_argument('--editing_model', default="gpt-4o-mini", type=str)
    parser.add_argument('--synonym_model', default="gpt-4o-mini", type=str)
    parser.add_argument('--n_test_exs', default=400, type=int)
    # parser.add_argument('--config', default='default.json')

    # hyperparameters
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--optimizer', default='nl-gradient')

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=4, type=int) # number of times to repeat the gradient generation process
    parser.add_argument('--errors_per_gradient', default=4, type=int) # errors per gradient round, included in error_string and provided to get_gradients
    parser.add_argument('--gradients_per_error', default=4, type=int) # number of gradients the model outputs in each call of get_gradients
    parser.add_argument('--steps_per_gradient', default=1, type=int) # number of new prompts to rewrite for each individual gradient
    parser.add_argument('--mc_samples_per_step', default=2, type=int) # number of synonyms to generate per rewritten prompt
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="mae", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    # parser.add_argument('--knn_k', default=2, type=int)
    # parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true') 
    
    # vestigial
    # parser.add_argument('--engine', default="chatgpt", type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.data_dir, args.nutrient, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.RegressionPredictor(config)

    # init optimizer
    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(',')]

    for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)
        
        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        round_time = time.time() - start
        print(f"Round {round} finished in {round_time:.2f}s")

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round} ========\n")
            outf.write(f"Wallclock time: {round_time:.2f}s\n")

            # summary view (just top scores, truncated prompt preview)
            outf.write("Top candidates (preview):\n")
            for i, (cand, score) in enumerate(zip(candidates, scores)):
                preview = cand[:200].replace("\n", " ")  # truncate long prompts
                outf.write(f"  {i+1:02d}: score={score:.4f}, preview=\"{preview}...\"\n")

            # full raw dump (for later parsing/repro)
            outf.write("\n-- RAW DATA --\n")
            outf.write(json.dumps({
                "candidates": candidates,
                "scores": scores
            }, indent=2))
            outf.write("\n")

        # evaluate each candidate on held-out set
        metrics = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            mae, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
            metrics.append({
                "rank": i + 1,
                "beam_score": score,
                "mae": mae
            })

        with open(args.out, 'a') as outf:
            outf.write("Evaluation results:\n")
            for m in metrics:
                outf.write(f"  Rank {m['rank']}: score={m['beam_score']:.4f}, MAE={m['mae']:.4f}\n")
            outf.write("\n")

    print("DONE!")
