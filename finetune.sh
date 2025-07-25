#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Use huggingface mirror to speed up downloads in China
export HF_ENDPOINT="https://hf-mirror.com"
# Set timeout for model downloads (in seconds)
export HF_HUB_DOWNLOAD_TIMEOUT=600

DIR=`pwd`

# It's recommended to download the model first:
# git lfs install
# git clone https://huggingface.co/MBZUAI/GeoPixel-7B
# Then set MODEL to local path like: export MODEL="./GeoPixel-7B"
export MODEL="MBZUAI/GeoPixel-7B"
# export DATA="path of data"
export DATA="data.txt"

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PYTHONWARNINGS="ignore" torchrun $DISTRIBUTED_ARGS train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --is_pretrained True \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler False \
    --use_lora True \
    --hd_num 1 \
    --output_dir output\
    --num_train_epochs 3\
    --batch_size 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --adam_beta2 0.95 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir "./logs" \
    --report_to "tensorboard" \
    --max_length 16384 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True