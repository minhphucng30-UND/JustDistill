from datasets import load_dataset
import os
from tqdm import tqdm
import time
import argparse
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import torch
from torch.utils.data import Dataset
torch.backends.cuda.matmul.allow_tf32 = True

class TokenizedDataset(Dataset):
    def __init__(self, prompts, answers, tokenizer):
        self.prompts = prompts
        self.answers = answers
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        input_ids = []
        prompt_lengths = 0
        tokenized_prompt = self.tokenizer(self.prompts[idx], add_special_tokens=False)['input_ids']
        input_ids.extend(tokenized_prompt)
        prompt_lengths = len(tokenized_prompt)
        answer_ids = self.tokenizer(self.answers[idx], add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]
        input_ids.extend(answer_ids)
        return {
            'input_ids': input_ids,
            'prompt_lengths': prompt_lengths,
        }
    

def collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    all_prompt_lengths = []

    for item in batch:
        padding_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
        all_prompt_lengths.append(item["prompt_lengths"])
    return {
        "input_ids": torch.tensor(input_ids),
        "prompt_lengths": torch.tensor(all_prompt_lengths),
    }

def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/LLaDA-8B-Base")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="ckpts/LLaDA-8B-Base-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    
    args = parser.parse_args()
    ds = load_dataset(args.dataset, "main", split="train")
    questions = [[{"role": "user", "content": example["question"]}] for example in ds]
    answers = [example["answer"] for example in ds]
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    model = AutoModelForCausalLM.from_pretrained("models/LLaDA-8B-Base", **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained("models/LLaDA-8B-Base")

    prompts = [tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=True) for question in questions]

    dataset = TokenizedDataset(prompts, answers, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer), num_workers=8, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_update_steps_per_epoch = (len(dataloader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_update_steps_per_epoch * args.epochs, weight_decay=0.1)
    os.makedirs("ckpts", exist_ok=True)

    start_time = time.time()
    accumulation_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]
            noisy_batch, _, p_mask = forward_process(input_ids, args.eps)

            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            prompt_mask = prompt_mask.long()    
            answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

            masked_indices = (noisy_batch == 126336)

            logits = model(input_ids=noisy_batch).logits
                
            token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
            ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
            (ce_loss / args.gradient_accumulation_steps).backward()
            accumulation_step += 1
            if accumulation_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            progress_bar.set_postfix(loss=ce_loss.item())

        if accumulation_step % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        model.save_pretrained(args.output_dir + f"_epoch_{epoch+1}")
        tokenizer.save_pretrained(args.output_dir + f"_epoch_{epoch+1}")
    print(f"Training completed in {time.time() - start_time:.2f} seconds")