import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/huawei/code/test/qwen_jax/model_cache/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float16
)

test_text = "你好，中国的首都是哪里"
inputs = tokenizer(test_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=" * 50)
print(response)