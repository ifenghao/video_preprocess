# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from dam import DescribeAnythingModel, disable_torch_init
import cv2
from lang_sam import LangSAM
import os
from tqdm import tqdm
import requests
import json
import logging
import pandas as pd

def free_gpus():
    import subprocess
    return subprocess.call('ps -ef|grep "run.py"|grep -v grep|cut -c 9-16|xargs kill -9 2>/dev/null', shell=True)

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

def load_video_frame(video_name):
    cap = cv2.VideoCapture(video_name)
    _, first_image = cap.read()
    cap.release()
    return first_image

def is_contained(mask1, mask2, threshold):
    intersection = np.sum(mask1 * mask2)
    sum1 = np.sum(mask1)
    sum2 = np.sum(mask2)
    if intersection / sum1 >= threshold:
        return 1
    if intersection / sum2 >= threshold:
        return 2
    return 0

def is_duplicated(mask1, mask2, threshold):
    intersection = np.sum(mask1 * mask2)
    sum1 = np.sum(mask1)
    sum2 = np.sum(mask2)
    ratio1 = intersection / sum1
    ratio2 = intersection / sum2
    if ratio1 >= threshold and ratio2 >= threshold:
        if ratio1 < ratio2:
            return 1 # mask1 > mask2
        else:
            return 2
    return 0

def get_masks_filter_flag(masks, threshold=0.9):
    flag_list = [0] * len(masks)
    if len(masks) == 1:
        return flag_list
    for i in range(len(masks)):
        if flag_list[i] == 1:
            continue
        for j in range(i+1, len(masks)):
            if flag_list[j] == 1:
                continue
            flag = is_duplicated(masks[i], masks[j], threshold)
            if flag == 1:
                flag_list[j] = 1
            elif flag == 2:
                flag_list[i] = 1
    return flag_list

def masks_filter_duplicated(masks_list, bbox_list=None):
    split_index = []
    merge_masks = []
    merge_bbox = []
    for i, masks in enumerate(masks_list):
        merge_masks.extend(masks)
        split_index.extend([i] * len(masks))
        if bbox_list is not None:
            merge_bbox.extend(bbox_list[i])
    flag_list = get_masks_filter_flag(merge_masks)
    res = [[] for _ in masks_list]
    if bbox_list is not None:
        res_bbox = [[] for _ in bbox_list]
    for i, flag in enumerate(flag_list):
        if flag == 0:
            res[split_index[i]].append(merge_masks[i])
            if bbox_list is not None:
                res_bbox[split_index[i]].append(merge_bbox[i])
    if bbox_list is not None:
        return res, res_bbox
    return res

def masks_filter_use_cate_mask(masks_list, cate_mask, threshold=0.9):
    res = []
    for masks in masks_list:
        masks_1 = []
        for mask in masks:
            flag = is_duplicated(cate_mask, mask, threshold)
            if flag == 0:
                masks_1.append(mask)
        res.append(masks_1)
    return res

def cate_filter_by_area(masks, boxes, threshold=0.1):
    area_list = []
    for mask in masks:
        area_list.append(np.sum(mask))
    max_area = max(area_list)
    res_masks = []
    res_boxes = []
    for i in range(len(area_list)):
        if area_list[i] / max_area > threshold:
            res_masks.append(masks[i])
            res_boxes.append(boxes[i])
    return res_masks, res_boxes

def main():
    parser = argparse.ArgumentParser(description="Describe Anything script")
    parser.add_argument(
        '--query', type=str, default='<image>\nDescribe the masked region in detail.', help='Prompt for the model')
    parser.add_argument('--model_path', type=str,
                        default='./models/DAM-3B-Video', help='Path to the model checkpoint')
    parser.add_argument('--sam_model_path', type=str,
                        default='./models/sam2/sam2.1_hiera_large.pt', help='Path to the model checkpoint')
    parser.add_argument('--gdino_model_path', type=str,
                        default='./models/grounding-dino-base', help='Path to the model checkpoint')
    parser.add_argument('--prompt_mode', type=str,
                        default='focal_prompt', help='Prompt mode')
    parser.add_argument('--conv_mode', type=str,
                        default='v1', help='Conversation mode')
    parser.add_argument('--temperature', type=float,
                        default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.5,
                        help='Top-p for sampling')
    parser.add_argument(
        "--input_file",
        type=str,
        default='',
        help="输入文件 json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default='./data/sdp_distill',
    )
    parser.add_argument(
        "--video_path_key",
        type=str,
        default='video_path',
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
    ext = args.input_file.split('.')[-1]
    if ext == 'json':
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    elif ext == 'csv':
        data = pd.read_csv(args.input_file, sep='\t')
        data = data.to_dict('records')
    print('total', len(data))
    if args.input_file.endswith(f'{args.part_index}-{args.part_total}.json'):
        start, end = 0, len(data)
    else:
        start, end = get_part_lines(len(data), args.part_index, args.part_total)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_frame_save_path = './data/langsam_first_frames'
    mask_save_path = './data/langsam_masks'
    os.makedirs(first_frame_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, f'{args.part_index}-{args.part_total}.txt')

    # Initialize DAM model
    disable_torch_init()
    prompt_modes = {
        "focal_prompt": "full+focal_crop",
    }
    dam = DescribeAnythingModel(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=prompt_modes.get(args.prompt_mode, args.prompt_mode),
    ).to(device)

    langsam = LangSAM(sam_type='sam2.1_hiera_large',
            sam_ckpt_path=args.sam_model_path,
            gdino_model_ckpt_path=args.gdino_model_path,
            gdino_processor_ckpt_path=args.gdino_model_path)
    cate_prompts = ["character", ]
    person_prompts = ["hair", "hair accessory", "eyebrows", "eyes", "nose", "mouth", "beard", "face", "clothing", "shoes", "earrings", "necklace", "bracelet"]

    file_list = os.listdir(args.save_path)
    exist = {}
    for file in file_list:
        with open(os.path.join(args.save_path, file), 'r') as f:
            part = f.readlines()
        for line in part:
            line = line.strip().split('\t')
            if len(line) != 3: continue
            source_id = line[0]
            exist[source_id] = 1
        print('load exist', len(exist))
    print(len(data), start, end)
    free_gpus()
    fail_cnt = 0
    # data = os.listdir('./test_images')
    with open(save_file, 'a') as f:
        for item in tqdm(data[start:end]):
            if item['source_id'] in exist: continue
            try:
                image_name = item['source_id'] + '.png'
                image_path = f"{first_frame_save_path}/{image_name}"
                if args.video_path_key not in item:
                    video_url = item['cos_signed_url']
                    video_path = f"{args.video_root}/{item['source_id']}.mp4"
                    if not try_download(video_url, video_path):
                        print(f"Failed to download {video_url}")
                        continue
                    item['video_path'] = video_path
                    image = load_video_frame(video_path)
                    cv2.imwrite(image_path, image)
                else:
                    image = load_video_frame(item[args.video_path_key])
                    cv2.imwrite(image_path, image)
                # image_path = f"./test_images/{item}"
                # item = {'source_id': item.split('.')[0]}

                image_pil = Image.open(image_path).convert("RGB")
                # dir_name = f'{mask_save_path}/{item["source_id"]}'
                # os.makedirs(dir_name, exist_ok=True)
                desc_dict = {}
                for cate_prompt in cate_prompts:
                    desc_dict[cate_prompt] = []
                    results = langsam.predict([image_pil], [cate_prompt])
                    masks = results[0]['masks']
                    bboxes = results[0]['boxes']
                    if len(masks) == 0:
                        continue

                    masks, bboxes = cate_filter_by_area(masks, bboxes)
                    
                    person_mask_list = []
                    for person_prompt in person_prompts:
                        res = langsam.predict([image_pil], [person_prompt])
                        person_mask_list.append(res[0]['masks'])

                    cate_masks, cate_bboxes = masks_filter_duplicated([masks], [bboxes])
                    cate_masks, cate_bboxes = cate_masks[0], cate_bboxes[0]
                    person_masks_list = masks_filter_duplicated(person_mask_list)
                    for i, cate_mask in enumerate(cate_masks):
                        person_masks_filter_list = masks_filter_use_cate_mask(person_masks_list, cate_mask)
                        cate_desc = {'bounding_box': str(cate_bboxes[i].tolist())}

                        cate_mask = cate_mask.astype(np.uint8)
                        if np.sum(cate_mask) == 0:
                            desc_dict[cate_prompt].append(cate_desc)
                            continue
                        cate_mask_pil = Image.fromarray(cate_mask * 255)
                        # cate_mask_save = Image.fromarray(cate_mask[:,:,None] * np.array(image_pil) + (1 - cate_mask[:,:,None]) * 255)
                        # cate_mask_path = f'{dir_name}/{i}_{cate_prompt}.png'
                        # cate_mask_save.save(cate_mask_path)

                        cate_out = dam.get_description(
                            image_pil,
                            cate_mask_pil,
                            args.query,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=1,
                            max_new_tokens=1024,
                        )
                        cate_desc["overall_description"] = cate_out
                        cate_desc["regional_description"] = {}
                        for j, person_masks in enumerate(person_masks_filter_list):
                            if len(person_masks) == 0:
                                cate_desc["regional_description"][person_prompts[j]] = None
                                continue
                            combine_mask = np.zeros_like(cate_mask)
                            for person_mask in person_masks:
                                combine_mask += cate_mask * person_mask.astype(np.uint8)

                            combine_mask = (combine_mask > 0).astype(np.uint8)
                            if np.sum(combine_mask) == 0:
                                cate_desc["regional_description"][person_prompts[j]] = None
                                continue
                            person_mask_pil = Image.fromarray(combine_mask * 255)
                            # person_mask_save = Image.fromarray(combine_mask[:,:,None] * np.array(image_pil) + (1 - combine_mask[:,:,None]) * 255)
                            # person_mask_path = f'{dir_name}/{i}_{cate_prompt}_{person_prompts[j]}.png'
                            # person_mask_save.save(person_mask_path)
                            person_out = dam.get_description(
                                image_pil,
                                person_mask_pil,
                                args.query,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=1,
                                max_new_tokens=1024,
                            )
                            cate_desc["regional_description"][person_prompts[j]] = person_out
                            print(person_prompts[j], person_out)
                        desc_dict[cate_prompt].append(cate_desc)

                line = '\t'.join([item['source_id'], image_path, json.dumps(desc_dict, ensure_ascii=False)])
                f.write(line + '\n')
                f.flush()
            except Exception as e:
                fail_cnt += 1
                logging.exception(e)
                print('fail', item)
    print('finish', args.part_index, 'failed', fail_cnt)

def save_final_result(version='v5_3w'):
    video_root = './data/sdp_distill'
    save_path = f'./captions/caption_{version}'
    file_list = os.listdir(save_path)
    res = {}
    for file in file_list:
        with open(os.path.join(save_path, file), 'r') as f:
            part = f.readlines()
        for line in part:
            source_id, image_path, desc_dict = line.strip().split('\t')
            res[source_id] = json.loads(desc_dict)
    
    input_file = './data/caption_v5_3w_trans.json'
    with open(input_file, 'r') as f:
        data = json.load(f)

    save_list = []
    for line in data:
        video_path = f"{video_root}/{line['source_id']}.mp4"
        key = line['source_id']
        if key not in res: continue
        desc_dict = res[key]
        line['video_path'] = video_path
        line['detailed_description'] = desc_dict['character']
        save_list.append(line)
    print(len(save_list), len(data))
    with open(f'./data/caption_{version}.json', 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
    # save_final_result()

