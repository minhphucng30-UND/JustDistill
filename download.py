from huggingface_hub import hf_hub_download, snapshot_download
import time
import torch

snapshot_download(repo_id="JetLM/SDAR-1.7B-Chat", local_dir="models/SDAR-1_7B-Chat")

# repo_name = "models/LLaDA-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
# # model = model.cuda().to(torch.bfloat16)

# history = []

# user_input = input("User: ").strip()
# history.append({"role": "user", "content": user_input})

# prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
# prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device='cuda')

# ## Chat in AR Mode
# gen_length = 256
# block_length = 32

# out_ids =  model.generate(prompt_ids,  gen_length=gen_length, block_length=block_length)
# tokenized_out = tokenizer.batch_decode(out_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]

# print(f"Model: {tokenized_out}")
