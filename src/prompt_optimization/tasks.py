import requests
import concurrent.futures
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
import utils

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass

def process_example(ex, predictor, prompt):
    try:
        pred = predictor.inference(ex, prompt)
        if not isinstance(pred, list):
            pred = [pred]
    except Exception:
        pred = [-1] * len(ex['y'])
    return ex, pred


class RegressionTask(DataProcessor):
    def stringify_prediction(self, pred):
        try:
            return f"{float(pred):.2f}"
        except Exception:
            return str(pred)
    
    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        gt = []
        preds = []
        texts = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex['text'])
                gt.append(ex['y'])
                preds.append(pred)

        gt_arr = np.array(gt, dtype=float)
        pred_arr = np.array(preds, dtype=float)
        mae = float(np.mean(np.abs(gt_arr - pred_arr)))
        return mae, texts, gt, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                mae, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return mae, texts, labels, preds
    

class NutrientTask(RegressionTask):
    def __init__(self, data_dir, nutrient="carb", max_threads=1):
        super().__init__(data_dir, max_threads)
        self.nutrient = nutrient
        self.all_nutrients = ["carb", "energy", "fat", "protein"]

    def _process_row(self, row):
        if self.nutrient == "combined":
            # Return a list of [carb, energy, fat, protein]
            values = [utils.process_gt(row[n]) for n in self.all_nutrients]
        else:
            values = [utils.process_gt(row[self.nutrient])]
        return values


    def get_train_examples(self):
        df = pd.read_csv(self.data_dir + '/nb_v2_train.csv')
        exs = []
        for i, row in df.iterrows():
            y = self._process_row(row)
            exs.append({'id': f'train-{i}', 'text': row['queries'], 'y': y})
        return exs

    def get_test_examples(self):
        df = pd.read_csv(self.data_dir + '/nb_v2_test.csv')
        exs = []
        for i, row in df.iterrows():
            y = self._process_row(row)
            exs.append({'id': f'test-{i}', 'text': row['queries'], 'y': y})
        return exs

    # needs to be updated so that nutrients get labels
    def stringify_prediction(self, pred):
        try:
            vals = [f"{float(x):.2f}" for x in pred]
            return ", ".join(vals)
        except Exception:
            return str(pred)

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        gt = []
        preds = []
        texts = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex['text'])
                gt.append(ex['y'])
                preds.append(pred)

        gt_arr = np.array(gt, dtype=float)
        pred_arr = np.array(preds, dtype=float)
        mae = float(np.mean(np.abs(gt_arr - pred_arr)))

        return mae, texts, gt, preds