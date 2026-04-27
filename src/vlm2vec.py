import sys
import argparse
import torch
import numpy as np
import pandas as pd
import cv2
import json
from tqdm import tqdm
import time
import os
import logging
import requests

def free_gpus():
    import subprocess
    return subprocess.call('ps -ef|grep "run.py"|grep -v grep|cut -c 9-16|xargs kill -9 2>/dev/null', shell=True)

def load_model():
    model_args = ModelArguments(
        model_name='./models/Qwen2-VL-2B-Instruct',
        checkpoint_path='./models/VLM2Vec-V2.0',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    data_args = DataArguments()

    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)
    model = model.to('cuda', dtype=torch.bfloat16)
    model.eval()
    return model, processor

def encode_video(video_path, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
        videos=video_inputs,
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
    qry_output = model(qry=inputs)["qry_reps"]
    return qry_output.detach().float().cpu().squeeze()

def encode_input(text, model, processor):
    inputs = processor(text=text, images=None, return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    tgt_output = model(tgt=inputs)["tgt_reps"]
    return tgt_output.detach().float().cpu().squeeze()

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def download_url(url, save_path):
    myfile = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(myfile.content)
    return save_path

def try_download(url, video_path, max_trial=3):
    trial = 0
    while (not os.path.exists(video_path)) or os.path.getsize(video_path) < 10000:
        download_url(url, video_path)
        trial += 1
        if trial > max_trial:
            return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="输入文件 json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./data/anime_video_clip_10w',
        help="如果需要保存的路径",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default='./data/download_videos',
    )
    parser.add_argument(
        "--part_index",
        type=int,
        default=0,
        help="图像文件索引编号",
    )
    parser.add_argument(
        "--part_total",
        type=int,
        default=1,
        help="图像文件索引总编号",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model, processor = load_model()
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    print('total', len(data))

    start, end = get_part_lines(len(data), args.part_index, args.part_total)
    free_gpus()
    for item in tqdm(data[start:end]):
        if os.path.exists(os.path.join(args.save_dir, item['source_id'] + '.pt')): continue
        try:
            item['video_path'] = os.path.join(args.video_root, item['source_id'])
            if not try_download(item['cos_signed_url'], item['video_path']):
                print('download fail', item['source_id'])
                fail_cnt += 1
                continue
            feat = encode_video(item['video_path'], model, processor) #  sedance_pro_480p_path
            torch.save(feat, os.path.join(args.save_dir, item['source_id'] + '.pt'))
        except Exception as e:
            logging.exception(e)
            print('fail', item)

def main_sim():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="输入文件 json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./data/anime_video_clip_10w',
        help="如果需要保存的路径",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default='./data/download_videos',
    )
    parser.add_argument(
        "--part_index",
        type=int,
        default=0,
        help="图像文件索引编号",
    )
    parser.add_argument(
        "--part_total",
        type=int,
        default=1,
        help="图像文件索引总编号",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model, processor = load_model()
    ext = args.input_file.split('.')[-1]
    if ext == 'json':
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    elif ext == 'csv':
        data = pd.read_csv(args.input_file, sep='\t').to_dict(orient='records')
    else:
        data = []
    
    print('total', len(data))

    file_list = os.listdir(args.save_dir)
    exist = {}
    for file in file_list:
        with open(os.path.join(args.save_dir, file), 'r') as f:
            part = f.readlines()
        for line in part:
            line = line.strip().split('\t')
            if len(line) != 2: continue
            source_id, appear_new_object = line
            exist[source_id] = appear_new_object

    start, end = get_part_lines(len(data), args.part_index, args.part_total)
    free_gpus()
    fail_cnt = 0
    with open(os.path.join(args.save_dir, f'{args.part_index}-{args.part_total}.txt'), 'a') as f:
        for item in tqdm(data[start:end]):
            if item['source_id'] in exist: continue
            try:
                # item['video_path'] = os.path.join(args.video_root, item['source_id'])
                if not os.path.exists(item['video_path']):
                    item['video_path'] = os.path.join(args.video_root, item['source_id'] + '.mp4')
                if not try_download(item['cos_signed_url'], item['video_path']):
                    print('download fail', item['source_id'])
                    fail_cnt += 1
                    continue
                feat = encode_video(item['video_path'], model, processor)
                feat2 = encode_input(item['training_caption'], model, processor)
                sim = torch.cosine_similarity(feat, feat2, dim=0)
                f.write('\t'.join([item['source_id'], str(sim.item())]) + '\n')
                f.flush()
            except Exception as e:
                fail_cnt += 1
                logging.exception(e)
                print('fail', item)
    print('finish', args.part_index, 'fail', fail_cnt)

def test():
    model, processor = load_model()
    video_path = 'test.mp4'
    text = '画面中间有一只小猫,有着大大的黑色眼睛和灰白相间的条纹皮毛,粉色耳朵内侧。它先是睁大眼睛抬头看并张开嘴巴,表情惊讶,动作缓慢,接着眯着眼睛坐下说话,神情严肃,动作幅度适中。场景设置在木质地板上。镜头抖动。'
    feat = encode_video(video_path, model, processor)
    feat2 = encode_input(text, model, processor)
    sim = torch.cosine_similarity(feat, feat2, dim=0)
    print(f"Similarity: {sim.item()}")
    

if __name__ == '__main__':
    main()
    # main_sim()