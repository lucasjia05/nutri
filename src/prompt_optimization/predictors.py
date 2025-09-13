from abc import ABC, abstractmethod
from liquid import Template
import utils

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass


class RegressionPredictor(GPT4Predictor):
    def inference(self, ex, prompt, nutrient="carb", method="base"):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=1024, n=1, timeout=30, 
            temperature=self.opt['temperature'])[0]
        pred = utils.clean_output(response, ex['text'], method, nutrient)
        return pred
