from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/lucasjia/models/gemma-2-9b-it-tokenizer2",
                                           local_files_only=True)

print(tokenizer.chat_template)