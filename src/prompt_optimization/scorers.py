import utils
from collections import defaultdict
import numpy as np
from liquid import Template
from tqdm import tqdm
import concurrent.futures


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    pred = predictor.inference(ex, prompt)
    return prompt, ex, pred


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
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='MAE scorer'):
                    prompt, ex, pred = future.result()     
                    err = abs(float(pred) - float(ex['y'])) 
                    key = (ex['id'], pids[prompt])
                    out_scores[key] = -float(err) 
                    # negative to work with UCB/BF evaluators (higher is better)
            return out_scores

        cached_scores = defaultdict(list)
        to_compute = []
        for ex in data:
            for prompt in prompts:
                key = (ex['id'], pids[prompt])
                if key in self.cache:
                    cached_scores[prompt].append(self.cache[key])
                else:
                    to_compute.append((prompt, ex))
        
        computed = compute_scores(to_compute)  
        for prompt, ex in to_compute:
            key = (ex['id'], pids[prompt])
            val = computed[key]
            self.cache[key] = val
            cached_scores[prompt].append(val)

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)



class Cached01Scorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(predict_on_example, ex) for ex in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
                    prompt, ex, pred = future.result()            
                    if pred == ex['label']:
                        out_scores[f'{ex}-{prompt}'] = 1
                    else:
                        out_scores[f'{ex}-{prompt}'] = 0
            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)


def logprob_on_example(inputs):
    ex, predictor, base_prompt, prompt, temperature = inputs
    lps = utils.instructGPT_logprobs(prompt, temperature=temperature)
    # last log prob is the log prob of answer (assuming single token responses)
    return base_prompt, ex, lps[0]['logprobs']['token_logprobs'][-1]


class CachedLogLikelihoodScorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = []
            for prompt, ex in prompts_exs:
                inputs.append((
                    ex,
                    predictor,
                    prompt,
                    Template(
                        prompt + ' ' + predictor.categories[ex['label']]
                        ).render(text=ex['text']),
                            predictor.opt['temperature']
                ))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(logprob_on_example, input) for input in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)
                                                ), total=len(futures), desc='ll scorer'):
                    prompt, ex, pred = future.result()            
                    out_scores[f'{ex}-{prompt}'] = pred
            return out_scores


        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))

        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)
