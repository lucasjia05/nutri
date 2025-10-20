import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """
    def _sample_error_str(self, texts, labels, preds, task, n=4):
        """ Sample n highest absolute error strings from the given texts, labels, and preds"""

        errs = []
        for l, p in zip(labels, preds):
            err = np.mean([abs(float(x) - float(y)) for x, y in zip(l, p)])
            errs.append(err)
        top_idxs = sorted(range(len(errs)), key=lambda i: errs[i], reverse=True)[:n]

        error_string = ''
        for k, i in enumerate(top_idxs, 1):
            error_string += f'## Example {k}\n'
            error_string += f'Text: "{texts[i].strip()}"\n'
            error_string += f'Label: {task.stringify_prediction(labels[i])}\n'
            error_string += f'Prediction: {task.stringify_prediction(preds[i])}\n\n'
        return error_string.strip()


    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1, model="gpt-4o-mini"):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        I'm trying to write a few-shot nutrition estimation prompt.
    
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        res = utils.chatgpt(gradient_prompt, n=n, model=model)
        feedbacks = []
        for r in res:    
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1, model="gpt-4o-mini"):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a few-shot nutrition estimation prompt.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        res = utils.chatgpt(transformation_prompt, n=n, model=model)
        new_prompts = []
        for r in res:   
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3, model="gpt-4o-mini"):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = utils.chatgpt(rewriter_prompt, n=n, model=model)
        new_instructions = [x for x in new_instructions if x]
        return new_instructions

    def get_gradients(self, prompt, task_section, task, gpt4, texts, labels, preds, model="gpt-4o-mini"):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(
                task_section, error_string, self.opt['gradients_per_error'], n=1, model=model)
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, gpt4, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            #print("CURRENT PROMPT:", prompt)
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()
            # this part might be kept or might need to be cleaned later, remove all the remaining instances of task_section in previous functions, just edit the full prompt instead of just #task section
            # task_section = prompt

            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)

            # gradient-based rewrites
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, task_section, task, gpt4, texts, labels, preds, model=self.opt['gradient_model'])
                #print("gradient count:", len(gradients))
                new_task_sections = []
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'], model=self.opt['editing_model'])
                    new_task_sections += tmp
                #print("section:", new_task_sections)

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'], model=self.opt['synonym_model'])
                    mc_sampled_task_sections += mc_sects

            # combine gradient-based rewrites and generated synonym prompts
            new_sections = new_task_sections + mc_sampled_task_sections
            #print("NEW SECTIONS:", new_sections)
            new_sections = list(set(new_sections)) # dedup
            tmp_new_prompts = []
            sections = utils.parse_sectioned_prompt(prompt)

            for new_task_text in new_sections:
                # make a copy of all sections for this variant
                updated_sections = dict(sections)
                updated_sections['task'] = new_task_text.strip()

                # rebuild the full prompt from sections
                new_prompt = ""
                for header, content in updated_sections.items():
                    new_prompt += f"# {header.capitalize()}\n{content.strip()}\n\n"

                tmp_new_prompts.append(new_prompt.strip())
            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({'text': t, 'label': l})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    # speed up a little
                    tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                    error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads)
                    tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                else:
                    tmp_new_prompts = random.sample(tmp_new_prompts, 
                        k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        new_prompts = list(set(new_prompts)) # dedup
        #print("NEW PROMPTS:", new_prompts)
        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts, train_exs, task, gpt4,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval'],
            max_threads=self.max_threads
        )
        return evals
