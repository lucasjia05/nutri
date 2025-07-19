from transformers import AutoTokenizer

model_path = "/data/lucasjia/models/gemma-7b-it"
new_path = "/data/lucasjia/models/gemma-7b-it-tokenizer2"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.chat_template = """{{ bos_token }}
{% set system_message = "" %}
{% if messages[0]['role'] == 'system' %}
{% set system_message = messages[0]['content'] + "\\n\\n" %}
{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif %}
{% for message in loop_messages %}
<|{{ message['role'] }}|>{{ system_message if loop.index0 == 0 }}{{ message['content'] }}<|end|>
{% endfor %}
<|assistant|>"""

tokenizer.save_pretrained(new_path)
