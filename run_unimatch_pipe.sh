# start train: pssh -p 64 -i -t 0 -h cur.hosts bash ./run_unimatch.sh
# kill process: pssh -p 64 -i -t 0 -h cur.hosts 'ps -ef|grep "unimatch_flow.py"|grep -v grep|cut -c 9-16|xargs kill -9'

model_path=$1
input_path=$2
save_path=$3
video_root=$4
ext=$5
video_path_key=$6

USE_NUM=8
INDEX=$INDEX
TOTAL=$TAIJI_HOST_NUM
((part_total=USE_NUM * TOTAL))
((cur_index=USE_NUM * INDEX))

for ((i=0;i<USE_NUM;i++));
do
    {
        ((part_index = cur_index + i))
        echo ${cur_index}/${part_total}/process:${part_index}
        CUDA_VISIBLE_DEVICES=$i python src/unimatch_flow.py \
            --resume ${model_path} \
            --padding_factor 32 \
            --upsample_factor 4 \
            --num_scales 2 \
            --attn_splits_list 2 8 \
            --corr_radius_list -1 4 \
            --prop_radius_list -1 1 \
            --reg_refine \
            --num_reg_refine 6 \
            --input_file="${input_path}/${part_index}-${part_total}.${ext}" \
            --save_path="${save_path}" \
            --video_path_key="${video_path_key}" \
            --part_index="${part_index}" \
            --part_total="${part_total}" \
            --video_root="${video_root}" \
            > logs/data_log_node${TOTAL}_${INDEX}.txt 2>&1
    } &
done
wait