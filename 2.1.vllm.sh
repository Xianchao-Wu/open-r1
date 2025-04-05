#########################################################################
# File Name: 2.1.vllm.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Apr  1 11:30:06 2025
#########################################################################
#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export VLLM_ATTENTION_BACKEND=xformers

CUDA_VISIBLE_DEVICES=7 trl vllm-serve \
	--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
	--port 8001
	#--attn-backend xformers

