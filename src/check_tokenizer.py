from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/lucasjia/models/gemma-2-27b-it-tokenizer")

print(tokenizer.chat_template)