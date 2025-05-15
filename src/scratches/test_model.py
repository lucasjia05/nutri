from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print(torch.cuda.device_count())


model_id = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)


prompt = "how many carbohydrates are in a serving of white rice?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
print(model.device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated(0) / 1024 ** 2, "MiB allocated")