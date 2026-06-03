accelerate launch --num_processes 2 --config_file configs/fsdp.yaml train.py \
  --dataset gsm8k \
  --grad_accum 8