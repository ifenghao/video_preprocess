#!/bin/bash

USE_NUM=$HOST_GPU_NUM
INDEX=0
TOTAL=1
((part_total=USE_NUM * TOTAL))
((cur_index=USE_NUM * INDEX))

model_path=./models/Qwen3-32B
input_file=./data/qwen3vl_mpo_0121_infer.json
gt_caption_key=caption
pred_caption_key=qwen3vl
save_path=./data/caption_${pred_caption_key}

for ((i=0;i<USE_NUM;i++));
do
    {
        ((part_index = cur_index + i))
        echo ${cur_index}/${part_total}/process:${part_index}
        CUDA_VISIBLE_DEVICES=$i python3 reward_llm_model.py \
            --input_file ${input_file} \
            --save_path ${save_path} \
            --model_path ${model_path} \
            --gt_caption_key ${gt_caption_key} \
            --pred_caption_key ${pred_caption_key} \
            --is_cn \
            --verbose \
            --top_samples 1000 \
            --part_index="${part_index}" \
            --part_total="${part_total}" \
            > logs/reward_log_node${part_total}_${part_index}.txt 2>&1
    } &
done
wait

