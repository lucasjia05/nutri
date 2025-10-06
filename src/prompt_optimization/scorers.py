from collections import defaultdict
import numpy as np
from tqdm import tqdm
import concurrent.futures


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    try:
        pred = predictor.inference(ex, prompt)
    except Exception:
        # swallow API/network errors so the ProcessPool doesn't crash
        return prompt, ex, -1   # sentinel
    return prompt, ex, pred


NUTRIENT_WEIGHTS = {
    "carbs_g": 1.0,
    "calories_kcal": 1.0,
    "fat_g": 1.0,
    "protein_g": 1.0,
}

class CachedMAEScorer:
    def __init__(self):
        self.cache = {}
        self._pid = {}           # {full prompt : prompt id}
        self._next_pid = 0  

    def _prompt_id(self, prompt):
        pid = self._pid.get(prompt)
        if pid is None:
            pid = self._next_pid
            self._pid[prompt] = pid
            self._next_pid += 1
        return pid     

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        pids = {p: self._prompt_id(p) for p in prompts}
        
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(predict_on_example, inp) for inp in inputs]
                for i, future in tqdm(
                    enumerate(concurrent.futures.as_completed(futures)),
                    total=len(futures),
                    desc='4N MAE scorer'
                ):
                    prompt, ex, pred_str = future.result()

                    # --- Parse prediction JSON safely ---
                    try:
                        pred = json.loads(pred_str)
                    except Exception:
                        # if model output is invalid JSON → assign a big penalty
                        total_err = 1e6
                        key = (ex['id'], pids[prompt])
                        out_scores[key] = -total_err
                        continue

                    # --- Compute per-nutrient absolute error ---
                    total_err = 0.0
                    for nutrient, weight in NUTRIENT_WEIGHTS.items():
                        if nutrient in pred and nutrient in ex['y']:
                            try:
                                pred_val = float(pred[nutrient])
                                true_val = float(ex['y'][nutrient])
                                total_err += weight * abs(pred_val - true_val)
                            except Exception:
                                total_err += weight * 1e6   # penalty for bad type
                        else:
                            total_err += weight * 1e6       # penalty for missing field

                    key = (ex['id'], pids[prompt])
                    out_scores[key] = -total_err  # negative for "higher is better"

            return out_scores

        # --- Use cached scores where possible ---
        cached_scores = defaultdict(list)
        to_compute = []
        for ex in data:
            for prompt in prompts:
                key = (ex['id'], pids[prompt])
                if key in self.cache:
                    cached_scores[prompt].append(self.cache[key])
                else:
                    to_compute.append((prompt, ex))
        
        # --- Compute new scores for uncached pairs ---
        computed = compute_scores(to_compute)  
        for prompt, ex in to_compute:
            key = (ex['id'], pids[prompt])
            val = computed[key]
            self.cache[key] = val
            cached_scores[prompt].append(val)

        # --- Aggregate per prompt ---
        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)
