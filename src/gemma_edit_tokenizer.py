from transformers import AutoTokenizer

model_path = "/data/lucasjia/models/gemma-2-27b-it-tokenizer"
new_path = "/data/lucasjia/models/gemma-2-27b-it-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.chat_template = """{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
<|system|>{{ messages[0]['content'] }}<|end|>
{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif %}
{% for message in loop_messages %}
<|{{ message['role'] }}|>{{ message['content'] }}<|end|>
{% endfor %}
<|assistant|>"""

tokenizer.save_pretrained(new_path)
