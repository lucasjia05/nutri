from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated(0) / 1024 ** 2, "MiB allocated")


model_path = "/data/lucasjia/models/gemma-2-27b-it"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  
    device_map="auto" 
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the capital of France?"}],
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

# Decode final output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(model.device)
print(torch.cuda.memory_allocated(0) / 1024 ** 2, "MiB allocated")