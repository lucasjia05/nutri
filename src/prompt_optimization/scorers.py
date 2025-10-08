from collections import defaultdict
import numpy as np
from tqdm import tqdm
import concurrent.futures


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    try:
        pred = predictor.inference(ex, prompt)
    except Exception as e:
        # swallow API/network errors so the ProcessPool doesn't crash
        print(e)
        return prompt, ex, -1   # sentinel
    return prompt, ex, pred


NUTRIENT_WEIGHTS = [1.0, 1.0, 1.0, 1.0]   # carb, energy, fat, protein

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
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                                      total=len(futures), desc='MAE scorer'):
                    prompt, ex, pred = future.result()     

                    # --- compute error for single carb value or combined nutrients ---
                    total_err = 1e6  # default penalty
                    try:
                        # single nutrient case
                        # print(pred, ex['y'])
                        if isinstance(ex['y'], (int, float)):
                            total_err = abs(float(pred) - float(ex['y']))
                        # combined nutrients case (list of 4)
                        elif isinstance(ex['y'], list) and isinstance(pred, list) and len(pred) == 4 and len(ex['y']) == 4:
                            total_err = 0.0
                            for idx, w in enumerate(NUTRIENT_WEIGHTS):
                                total_err += w * abs(float(pred[idx]) - float(ex['y'][idx]))
                        else:
                            total_err = 1e6
                    except Exception as e:
                        print(e)
                        total_err = 1e6

                    key = (ex['id'], pids[prompt])
                    out_scores[key] = -float(total_err)  # negative for "higher is better"

            return out_scores

        # --- cached scores ---
        cached_scores = defaultdict(list)
        to_compute = []
        for ex in data:
            for prompt in prompts:
                key = (ex['id'], pids[prompt])
                if key in self.cache:
                    cached_scores[prompt].append(self.cache[key])
                else:
                    to_compute.append((prompt, ex))
        
        # --- compute new scores ---
        computed = compute_scores(to_compute)  
        for prompt, ex in to_compute:
            key = (ex['id'], pids[prompt])
            val = computed[key]
            self.cache[key] = val
            cached_scores[prompt].append(val)

        # --- aggregate ---
        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unknown aggregation method: ' + agg)