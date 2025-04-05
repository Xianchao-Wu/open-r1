#########################################################################
# File Name: 2.2.grpo.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Apr  3 00:44:21 2025
#########################################################################
#!/bin/bash

export NCCL_DEBUG=WARN
#--vllm-server-port 'http://localhost:8001/v1' \

CUDA_VISIBLE_DEVICES=6 python -m ipdb src/open_r1/grpo.py \
	--dataset_prompt_column 'problem' \
	--vllm-server-port 8001 \
	--config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
