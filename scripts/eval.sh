# --task gsm8k \  # math500/humaneval/mbpp
torchrun --nproc-per-node=2 eval.py \
  --task gsm8k \
  --ckpt_path models/LLaDA-8B-Instruct \
  --gen_length 256 --steps 256 --block_length 32