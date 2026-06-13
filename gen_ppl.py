import argparse
import json
import numpy as np
import torch.nn.functional as F
import time
from utils.generate import generate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo import sample, 
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/LLaDA-8B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='gsm8k')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output_path', type=str, default='outputs/ppl.txt')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--remasking', type=str, default='low_confidence')
    parser.add_argument('--use_cache', type=bool, default=False)

    device = 'cuda'
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ds = load_dataset(args.dataset_name, split='train')
    ds = ds.shuffle(seed=args.seed)
    ds = ds.select(range(args.n_samples))

    prompts = [[{"role": "user", "content": example["question"]}] for example in ds]
    prompt = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in prompts]

    input_ids = [tokenizer(prompt)['input_ids'] for prompt in prompts]
    input_ids = [torch.tensor(input_id).to(device).unsqueeze(0) for input_id in input_ids]


    all_outputs = []
    mask_id = 126336
    batch_size = 1

    start_time = time.time()
    for input_id in input_ids:
        with torch.no_grad():
            generated_ids = generate(model, input_id, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=args.temperature, remasking=args.remasking, use_cache=args.use_cache)
            prompt_ids = generated_ids[:, :input_id.shape[1]]
            completion_ids = generated_ids[:, input_id.shape[1]:]
            all_token_log_probs = 0.0
            for t in range(args.gen_length):
                # Construct input with AR masking (Past=Observed, Future=Masked)
                x = torch.cat([prompt_ids, completion_ids[:, :t],
                            torch.full((batch_size, args.gen_length - t), mask_id, device=device, dtype=generated_ids.dtype)], dim=1)

                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    logits = model(x).logits / args.temperature

                # Compute log probability of next token
                log_prob = F.log_softmax(logits[:, input_id.shape[1] + t, :].float(), dim=-1)
                token_log_prob = log_prob.gather(-1, completion_ids[:, t:t+1]).squeeze().item()
                all_token_log_probs += token_log_prob
            all_outputs.append(-all_token_log_probs / args.gen_length)
    
    gen_ppls = np.exp(np.mean(all_outputs)).item()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Output: {gen_ppls}")
    row_data = {
        'model': args.model_path,
        "gen_ppl": gen_ppls,
        'dataset': args.dataset_name,
        'seed': args.seed,
        'n_samples': args.n_samples,
        'steps': args.steps,
        'gen_length': args.gen_length,
        'block_length': args.block_length,
        'temperature': args.temperature,
        'remasking': args.remasking,
        'use_cache': args.use_cache,
    }
    with open(args.output_path, 'a') as f:
        f.write(json.dumps(row_data) + '\n')