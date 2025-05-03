#!/bin/bash

# result_answer_target2000
# # MODEL="mistralai/Mistral-7B-v0.1"
# # BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# # MODEL="meta-llama/Llama-2-7b-hf"
# # MODEL="meta-llama/Llama-2-7b-chat-hf"
# BASE_MODEL="/home/qiaoan/data/llama-2-7b-chat/models--unsloth--llama-2-7b-chat/snapshots/a6d63d7c9ac31fd7e6d31e66ee0d1c784a489fcf"
# # MODEL_NAME="${MODEL//\//-}"

# # maxlen=5120
# mem=128
# r=128
# mean_compression_rate=5
# target_token=2000

# ICAE_MODEL_PATH="/home/qiaoan/data/icae/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"
# # ICAE_MODEL_PATH=$1  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

# CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench.py --mean_compression_rate $mean_compression_rate \
#                                                         --fixed_mem_size $mem \
#                                                         --lora_r $r \
#                                                         --output_dir $ICAE_MODEL_PATH \
#                                                         --model_name_or_path $BASE_MODEL \
#                                                         --bf16 \
#                                                         --train False \
#                                                         --load_origin_from /home/qiaoan/data/LongBench/data \
#                                                         --datasets all \
#                                                         --load_key context \
#                                                         --target_token $target_token \
#                                                         --save_path ./result_answer.json \

# result_answer_cr5_metallama
# # MODEL="mistralai/Mistral-7B-v0.1"
# # BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# # MODEL="meta-llama/Llama-2-7b-hf"
# # MODEL="meta-llama/Llama-2-7b-chat-hf"
# BASE_MODEL="/home/qiaoan/data/meta-llama-2-7b-chat-hf"
# # MODEL_NAME="${MODEL//\//-}"

# # maxlen=5120
# mem=128
# r=128
# mean_compression_rate=5
# target_token=0

# ICAE_MODEL_PATH="/home/qiaoan/data/icae/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"
# # ICAE_MODEL_PATH=$1  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

# CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench_v2.py --mean_compression_rate $mean_compression_rate \
#                                                         --fixed_mem_size $mem \
#                                                         --lora_r $r \
#                                                         --output_dir $ICAE_MODEL_PATH \
#                                                         --model_name_or_path $BASE_MODEL \
#                                                         --bf16 \
#                                                         --train False \
#                                                         --load_origin_from /home/qiaoan/data/LongBench/data \
#                                                         --datasets all \
#                                                         --load_key context \
#                                                         --target_token $target_token \
#                                                         --save_path ./output/result_answer_cr5_metallama.json \

# result_answer_cr5_metallama
# # MODEL="mistralai/Mistral-7B-v0.1"
# # BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# # MODEL="meta-llama/Llama-2-7b-hf"
# # MODEL="meta-llama/Llama-2-7b-chat-hf"
# BASE_MODEL="/home/qiaoan/data/meta-llama-2-7b-chat-hf"
# # MODEL_NAME="${MODEL//\//-}"

# # maxlen=5120
# mem=128
# r=128
# mean_compression_rate=5
# target_token=0

# ICAE_MODEL_PATH="/home/qiaoan/data/icae/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"
# # ICAE_MODEL_PATH=$1  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

# CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench_v2.py --mean_compression_rate $mean_compression_rate \
#                                                         --fixed_mem_size $mem \
#                                                         --lora_r $r \
#                                                         --output_dir $ICAE_MODEL_PATH \
#                                                         --model_name_or_path $BASE_MODEL \
#                                                         --bf16 \
#                                                         --train False \
#                                                         --load_origin_from /home/qiaoan/data/LongBench/data \
#                                                         --datasets all \
#                                                         --load_key context \
#                                                         --target_token $target_token \
#                                                         --save_path ./output/result_answer_cr5_metallama.json \

# # result_answer_target2000_mistral_e
# # MODEL="mistralai/Mistral-7B-v0.1"
# # BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# # MODEL="meta-llama/Llama-2-7b-hf"
# # MODEL="meta-llama/Llama-2-7b-chat-hf"
# BASE_MODEL="/home/qiaoan/data/Mistral-7B-Instruct-v0.2"
# # MODEL_NAME="${MODEL//\//-}"

# # maxlen=8000
# mem=128
# r=128
# mean_compression_rate=5
# target_token=2000

# ICAE_MODEL_PATH="/home/qiaoan/data/icae/mistral_7b_ft_icae.safetensors"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

# CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench_mistral_e.py --mean_compression_rate $mean_compression_rate \
#                                                         --fixed_mem_size $mem \
#                                                         --lora_r $r \
#                                                         --output_dir $ICAE_MODEL_PATH \
#                                                         --model_name_or_path $BASE_MODEL \
#                                                         --bf16 \
#                                                         --train False \
#                                                         --load_origin_from /home/qiaoan/data/LongBench/data \
#                                                         --datasets all \
#                                                         --load_key context \
#                                                         --target_token $target_token \
#                                                         --save_path ./output/result_answer_target2000_mistral_e.json \

# result_answer_cr5_mistral_e
# MODEL="mistralai/Mistral-7B-v0.1"
# BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="meta-llama/Llama-2-7b-chat-hf"
BASE_MODEL="/home/qiaoan/data/Mistral-7B-Instruct-v0.2"
# MODEL_NAME="${MODEL//\//-}"

# maxlen=8000
mem=128
r=128
mean_compression_rate=5
target_token=0

ICAE_MODEL_PATH="/home/qiaoan/data/icae/mistral_7b_ft_icae.safetensors"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench_mistral_e.py --mean_compression_rate $mean_compression_rate \
                                                        --fixed_mem_size $mem \
                                                        --lora_r $r \
                                                        --output_dir $ICAE_MODEL_PATH \
                                                        --model_name_or_path $BASE_MODEL \
                                                        --bf16 \
                                                        --train False \
                                                        --load_origin_from /home/qiaoan/data/LongBench/data \
                                                        --datasets all \
                                                        --load_key context \
                                                        --target_token $target_token \
                                                        --save_path ./output/result_answer_cr5_mistral_e.json \

# # result_answer_target2000_mistralai
# # MODEL="mistralai/Mistral-7B-v0.1"
# BASE_MODEL="/home/qiaoan/data/Mistral-7B-Instruct-v0.2"
# # MODEL="meta-llama/Llama-2-7b-hf"
# # MODEL="meta-llama/Llama-2-7b-chat-hf"
# # BASE_MODEL="/home/qiaoan/data/llama-2-7b-chat/models--unsloth--llama-2-7b-chat/snapshots/a6d63d7c9ac31fd7e6d31e66ee0d1c784a489fcf"
# # MODEL_NAME="${MODEL//\//-}"

# # maxlen=5120
# mem=128
# r=512
# mean_compression_rate=5
# target_token=2000

# ICAE_MODEL_PATH="/home/qiaoan/data/icae/mistral_7b_ft_icae.safetensors"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"
# # ICAE_MODEL_PATH=$1  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

# CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench_mistralai.py --mean_compression_rate $mean_compression_rate \
#                                                         --fixed_mem_size $mem \
#                                                         --lora_r $r \
#                                                         --output_dir $ICAE_MODEL_PATH \
#                                                         --model_name_or_path $BASE_MODEL \
#                                                         --bf16 \
#                                                         --train False \
#                                                         --load_origin_from /home/qiaoan/data/LongBench/data \
#                                                         --datasets all \
#                                                         --load_key context \
#                                                         --target_token $target_token \
#                                                         --save_path ./output/result_answer_target2000_mistralai.json \

# # result_answer_cr5_mistralai
# # MODEL="mistralai/Mistral-7B-v0.1"
# BASE_MODEL="/home/qiaoan/data/Mistral-7B-Instruct-v0.2"
# # MODEL="meta-llama/Llama-2-7b-hf"
# # MODEL="meta-llama/Llama-2-7b-chat-hf"
# # BASE_MODEL="/home/qiaoan/data/llama-2-7b-chat/models--unsloth--llama-2-7b-chat/snapshots/a6d63d7c9ac31fd7e6d31e66ee0d1c784a489fcf"
# # MODEL_NAME="${MODEL//\//-}"

# maxlen=5120
# mem=128
# r=512
# mean_compression_rate=4
# target_token=0

# ICAE_MODEL_PATH="/home/qiaoan/data/icae/mistral_7b_ft_icae.safetensors"  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"
# # ICAE_MODEL_PATH=$1  # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"

# CUDA_LAUNCH_BLOCKING=1 python test_icae_on_longbench_mistralai.py --mean_compression_rate $mean_compression_rate \
#                                                         --model_max_length $maxlen \
#                                                         --fixed_mem_size $mem \
#                                                         --lora_r $r \
#                                                         --output_dir $ICAE_MODEL_PATH \
#                                                         --model_name_or_path $BASE_MODEL \
#                                                         --bf16 \
#                                                         --train False \
#                                                         --load_origin_from /home/qiaoan/data/LongBench/data \
#                                                         --datasets all \
#                                                         --load_key context \
#                                                         --target_token $target_token \
#                                                         --save_path ./output/result_answer_cr5_mistralai.json \