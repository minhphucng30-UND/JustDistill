import torch
import time
import numpy as np
import torch.nn.functional as F
import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens_simple(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_simple(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    assert prompt.shape[0] == 1 or (prompt[0] == prompt).all(), \
        "generate() requires all prompts in the batch to be identical (use prompt.repeat(n, 1)). " \
        "Variable-length prompts with mask_id padding are not supported."

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_simple(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x



def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.0, remasking='low_confidence_dynamic', mask_id=126336):
    model.eval()
    prompt_length = prompt.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = prompt.clone()
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
            )

            # Sampling strategy
            if remasking == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > 0.85
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking}")

            cur_x[transfer_index] = x0[transfer_index]
        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
    return x



def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if target in ['low_confidence', 'confidence']:
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values            # (b, l, 2)
        x0_p = top2[..., 0] - top2[..., 1]                # Δ(top1, top2)
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # –entropy
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(target)
    
    x0 = torch.where(mask_index, x0, x)
    
    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)  # (B, T)

        has_mask = mask_index.any(dim=-1)               # (B,)
        none_sel = (~selected.any(dim=-1)) & has_mask   # (B,)
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)     # (B,)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True

        return x0, selected

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., remasking='confidence', mask_id=126336, use_cache=True, unmask_threshold=0.9, attention_mask=None) -> torch.Tensor:
    B, L0 = prompt.shape
    x = torch.full((B, L0 + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :L0] = prompt
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask.to(device=prompt.device),
                torch.ones(B, gen_length, dtype=attention_mask.dtype, device=prompt.device),
            ],
            dim=1,
        )

    for blk in range(num_blocks):
        s, e = L0 + blk * block_length, L0 + (blk + 1) * block_length

        cur_steps = max(steps_per_block[blk], 1)
        num_transfer = get_num_transfer_tokens((x[:, s:e] == mask_id), cur_steps)

        out = None
        pkv = None
        if use_cache:
            out = model(x, use_cache=True)
            pkv = tuple(
                tuple(t[:, :, :s] for t in layer) for layer in out.past_key_values
            )

        step_i = 0
        max_iters = cur_steps + block_length
        while (x[:, s:e] == mask_id).any() and step_i < max_iters:
            mask_blk = (x[:, s:e] == mask_id)
            if attention_mask is not None:
                mask_blk &= attention_mask[:, s:e].bool()

            transfer_step = min(step_i, cur_steps - 1)

            if step_i == 0:
                if use_cache:
                    logits = out.logits[:, s:e]
                else:
                    logits = model(x, use_cache=False).logits[:, s:e]
            elif use_cache:
                logits = model(x[:, s:], past_key_values=pkv, use_cache=True).logits[:, :block_length]
            else:
                logits = model(x, use_cache=False).logits[:, s:e]

            x0, tr_idx = get_transfer_index(
                logits, temperature, remasking, mask_blk, x[:, s:e],
                num_transfer[:, transfer_step], unmask_threshold)
            x[:, s:e][tr_idx] = x0[tr_idx]
            step_i += 1

    return x


def main():
    device = 'cuda'

    model = AutoModelForCausalLM.from_pretrained('models/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('models/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    start_time = time.time()
    out = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence', use_cache=False)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Output: {tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}")

    # start_time = time.time()
    # out = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence')
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    # print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
