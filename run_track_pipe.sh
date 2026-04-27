# start train: pssh -p 64 -i -t 0 -h cur.hosts8 bash ./run_track.sh
# kill process: pssh -p 64 -i -t 0 -h cur.hosts8 'ps -ef|grep "track.py"|grep -v grep|cut -c 9-16|xargs kill -9'

input_file=$1
save_path=$2
model_path=$3
video_root=$4
is_3d=$5
video_path_key=$6

USE_NUM=$HOST_GPU_NUM
INDEX=$INDEX
TOTAL=$TAIJI_HOST_NUM
((part_total=USE_NUM * TOTAL))
((cur_index=USE_NUM * INDEX))

for ((i=0;i<USE_NUM;i++));
do
    {
        ((part_index = cur_index + i))
        echo ${cur_index}/${part_total}/process:${part_index}
        if [[ $input_file =~ '.json' ]] 
        then 
            file=$input_file
        else
            file=${input_file}/${part_index}-${part_total}.json
        fi
        CUDA_VISIBLE_DEVICES=$i python src/track.py \
            --input_file="${file}" \
            --save_path="${save_path}" \
            --model_path="${model_path}" \
            --video_path_key="${video_path_key}" \
            --is_3d="${is_3d}" \
            --part_index="${part_index}" \
            --part_total="${part_total}" \
            --video_root="${video_root}" \
            > logs/data_log_node${TOTAL}_${INDEX}.txt 2>&1
    } &
done
wait

