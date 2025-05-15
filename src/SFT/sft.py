import os
import torch
import pandas as pd
import json
import random

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    logging
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, setup_chat_format, SFTConfig
import bitsandbytes as bnb
from datasets import Dataset


# helper
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    lora_module_names.discard('lm_head')  # Exclude lm_head for 16-bit
    return list(lora_module_names)


# tester
def test_model(model, tokenizer):
    print(model.device)
    print(torch.cuda.memory_allocated(0) / 1024 ** 2, "MiB allocated")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "How many carbohydrates are in a serving of white rice?"}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))




# -------------------------------- Load Model --------------------------------
def load_model(model_path = "/data/lucasjia/models/gemma-2-9b-it-tokenizer2"):    
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # modules = find_all_linear_names(model)
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules
    )

    tokenizer.chat_template = None

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)

    return model, tokenizer, peft_config




# -------------------------------- Load Training Data --------------------------------
def load_training_data(path="/data/lucasjia/projects/assignment1/src/SFT/train_v2.jsonl", seed=42, limit=None):
    dataset = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

    random.seed(seed)
    random.shuffle(dataset)

    if limit!=None:
        dataset = dataset[:limit]
    split_idx = int(len(dataset) * 0.8)
    train_data = Dataset.from_list(dataset[:split_idx])
    val_data = Dataset.from_list(dataset[split_idx:])

    

    return train_data, val_data




# # -------------------------------- Training arguments --------------------------------
# def get_train_args(output_dir="/data/lucasjia/models/gemma-2-9b-it-SFT"):
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=2,
#         gradient_accumulation_steps=2,
#         optim="paged_adamw_32bit",
#         num_train_epochs=3,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         eval_steps=200,  
#         save_steps=500,  
#         logging_steps=100,
#         warmup_steps=800,
#         logging_strategy="steps",
#         learning_rate=2e-4,
#         fp16=False,
#         bf16=True,
#         group_by_length=True,
#         load_best_model_at_end=True,
#         save_total_limit=3,
#     )
#     return training_args


# for testing
def get_train_args(output_dir="/data/lucasjia/models/gemma-2-9b-it-SFT-test"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,          
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,         
        optim="paged_adamw_32bit",
        num_train_epochs=1,                    
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,                          
        save_steps=100,
        logging_steps=10,                      
        warmup_steps=10,                        
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        group_by_length=True,
        load_best_model_at_end=False,          
        save_total_limit=1,    
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,               
    )
    return training_args


# -------------------------------- Full Workflow --------------------------------
def load_model_data_train(output_dir="/data/lucasjia/models/gemma-2-9b-it-SFT"):

    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    model, tokenizer, peft_config = load_model()
    model.config.use_cache = False
    model.train()  # Make sure model is in training mode

    # Optional: print device map for debugging
    if hasattr(model, 'hf_device_map'):
        print("Device map:", model.hf_device_map)


    train_data, val_data = load_training_data(limit=8000)
    training_args = get_train_args(output_dir)

    # def formatting_func(example):
    #     return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        args=training_args,
        # dataset_text_field="text", 
        # tokenizer=tokenizer,
        # packing=False,
        # formatting_func=formatting_func,
    )

    trainer.train()
    # model.config.use_cache = True

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)







if __name__ == "__main__":
    load_model_data_train()


    # model, tokenizer, peft_config = load_model()
    # test_model(model, tokenizer)
    # train_data, val_data = load_training_data()
    # print(val_data[0:2])

    # training_args = get_train_args()
    # print(training_args)

