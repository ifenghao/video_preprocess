#!/bin/bash
set -e

video_root=./data/videos
version=v1
input_file=./data/file.json

python3 split_file.py \
    --input_file ${input_file} \
    --split_path ./data/${version}_part \
    --parts 16 \
    --force_json

python3 reshuffle.py \
    --input_files ${input_file} \
    --save_path ./data/${version}_rest_part \
    --exist_path ./motions/motions${version} \
    --nsplits 64

# motion
bash /etc/taiji/discover_hosts.sh > cur.hosts
pssh -p 64 -i -t 0 -h cur.hosts bash ./run_unimatch_pipe.sh \
    ./unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
    ./data/${version}_part \
    ./motions/motions${version} \
    ${video_root} \
    json \
    video_path

python3 post_process.py \
    --input_files ${input_file} \
    --save_path ./motions/motions${version} \
    --filtered_save_file ./data/anime_video_${version}_motion_filter.json \
    --nsplits 0 \
    --split_path ./data/${version}_filter_part \
    --task motion

# python3 reshuffle.py \
#     --input_files ./data/anime_video_${version}_motion_filter.json \
#     --save_path ./data/${version}_rest_part \
#     --exist_path ./tracks/tracks${version} \
#     --nsplits 64

# appear
pssh -p 64 -i -t 0 -h cur.hosts bash ./run_keye_pipe.sh \
    ./data/${version}_part \
    ./appears/appears${version} \
    ./models/Keye-VL-8B-Preview \
    ${video_root} \
    video_path

python3 post_process.py \
    --input_files ${input_file} \
    --save_path ./appears/appears${version} \
    --filtered_save_file ./data/anime_video_${version}_motion_appear_filter.json \
    --nsplits 24 \
    --split_path ./data/${version}_appear_part \
    --task appear

# track
pssh -p 64 -i -t 0 -h cur.hosts bash ./run_track_pipe.sh \
    ./data/${version}_appear_part \
    ./tracks/tracks${version} \
    ./models/pips2_weights.pth \
    ${video_root} \
    "1" \
    video_path

python3 post_process.py \
    --input_files ./data/anime_video_${version}_motion_appear_filter.json \
    --save_path ./tracks/tracks${version} \
    --filtered_save_file ./data/anime_video_${version}_motion_track_filter.json \
    --nsplits 0 \
    --split_path ./data/${version}_track_part \
    --task track

