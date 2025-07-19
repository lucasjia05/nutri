from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print(torch.cuda.device_count())

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "google/gemma-2-9b-it"
# model_id = "meta-llama/Llama-2-7b-hf"
model_id = "meta-llama/Llama-2-13b-hf"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# path = "/data/lucasjia/models/meta-llama-3-8b-instruct"
# path = "/data/lucasjia/models/gemma-2-9b-it"
# path = "/data/lucasjia/models/gemma-2-27b-it"
# path = "/data/lucasjia/models/llama-2-7b"
path = "/data/lucasjia/models/llama-2-13b"

tokenizer.save_pretrained(path)
model.save_pretrained(path)

print(torch.cuda.memory_allocated(0) / 1024 ** 2, "MiB allocated")