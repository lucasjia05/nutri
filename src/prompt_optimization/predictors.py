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
    def inference(self, ex, prompt, nutrient=None, method="base"):
        if nutrient is None:
            nutrient = "carb"
        # detect combined automatically
        if isinstance(ex['y'], list):
            nutrient = "combined"

        # fill in the template
        prompt_filled = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt_filled,
            model=self.opt['task_model'],
            max_tokens=2048,
            n=1,
            timeout=30,
            temperature=self.opt['temperature']
        )[0]

        if nutrient == "combined":
            # split the response into a list of 4 floats
            try:
                pred = [float(x.strip()) for x in response.split(',')]
            except:
                pred = [-1] * 4
        else:
            pred = utils.clean_output(response, ex['text'], method, nutrient)

        return pred
