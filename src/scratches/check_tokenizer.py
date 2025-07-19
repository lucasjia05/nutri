from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/lucasjia/models/gemma-7b-it-tokenizer",
                                           local_files_only=True)

print(tokenizer.chat_template)