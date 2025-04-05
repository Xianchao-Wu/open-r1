#########################################################################
# File Name: 1.sft.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Apr  1 01:56:55 2025
#########################################################################
#!/bin/bash

# Train via command line
#accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
#maxseqlen=16384
maxseqlen=1024 #16384

# TODO this packing is to combine several sample's text together
# and rerange them to a fixed sequence length...

# with --packing = all samples' text putting togeter and then use max_seq_length to seg again
# could have <pre.sample.ending> <current.sample.text> <next.sample.starting> problem!

# without --packing, each sample is truncated by max_seq_length, kept the indepenent of each 
# sample
#--packing \
CUDA_VISIBLE_DEVICES=7 python -m ipdb \
	src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length $maxseqlen \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill

# Train via YAML config
#accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
#    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
