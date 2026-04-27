from transformers import AutoModel, AutoTokenizer, AutoProcessor
from keye_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info as process_vision_info_qwen
import torch
import pandas as pd
import json
from tqdm import tqdm
import argparse
import os
import logging
import requests
import cv2
import numpy as np
import imageio

def extract_video(video_name, force_fps=None, max_duration=5): # fps 固定
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), f'Failed to load video file {video_name}'
    # get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(np.ceil(fps))
    if force_fps is None:
        force_fps = fps
    interval = (fps - 1) // force_fps + 1
    max_frames = max_duration * force_fps if max_duration is not None else 1e10 
    imgs = []
    frame_count = 0
    while cap.isOpened():
        # get frames
        flag, img = cap.read()
        if not flag:
            break
        if frame_count % interval != 0:
            frame_count += 1
            continue
        
        # to rgb format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        frame_count += 1

        if len(imgs) >= max_frames:
            break
    cap.release()
    return imgs, fps

class KeyeModel:
    def __init__(self, model_path):
        # default: Load the model on the available device(s)
        # model_path = "Kwai-Keye/Keye-VL-8B-Preview"
        # model_path = "./models/Keye-VL-8B-Preview"  # pip install keye-vl-utils==1.0.0 same as qwen_vl_utils
        # model_path = "./models/Keye-VL-1_5-8B"  # pip install keye-vl-utils==1.5.2
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        # default processer
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def infer(self, video_path, prompt):
        # Messages containing a local video path and a text query
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
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Messages containing a video url and a text query
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": "http://s2-11508.kwimgs.com/kos/nlav11508/MLLM/videos_caption/98312843263.mp4",
        #             },
        #             {"type": "text", "text": "Have there been any objects that do not exist in the first frame of the video? Only answer Yes or No."},
        #         ],
        #     }
        # ]

        #In Keye-VL, frame rate information is also input into the model to align with absolute time.
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True) # return_video_kwargs=True
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

class QwenVLModel:
    def __init__(self, model_path):
        # default: Load the model on the available device(s)
        # model_path = "./models/Qwen2.5-VL-32B-Instruct"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        # default processer
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def infer(self, video_path, prompt):
        # Messages containing a local video path and a text query
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
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Messages containing a video url and a text query
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": "http://s2-11508.kwimgs.com/kos/nlav11508/MLLM/videos_caption/98312843263.mp4",
        #             },
        #             {"type": "text", "text": "Have there been any objects that do not exist in the first frame of the video? Only answer Yes or No."},
        #         ],
        #     }
        # ]

        #In Keye-VL, frame rate information is also input into the model to align with absolute time.
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info_qwen(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

def process_text(output_text):
    if '</analysis>' in output_text:
        output_text = output_text.split('</analysis>')[-1]
    if '</think>' in output_text:
        output_text = output_text.split('</think>')[-1]
    if '<answer>' in output_text:
        output_text = output_text.split('<answer>')[-1]
        output_text = output_text.split('</answer>')[0]
    if 'the answer is ' in output_text:
        output_text = output_text.split('the answer is ')[-1]
    if r'\boxed{' in output_text:
        output_text = output_text.split(r'\boxed{')[1]

    output_text = output_text.strip('"')
    if output_text.startswith('Yes'):
        return True
    elif output_text.startswith('No'):
        return False
    else:
        print('illegal', output_text)
        return

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def free_gpus():
    import subprocess
    return subprocess.call('ps -ef|grep "run.py"|grep -v grep|cut -c 9-16|xargs kill -9 2>/dev/null', shell=True)

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
        help="如果需要保存的路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default='./data/Netflix_Nyaa_790w_download_videos',
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

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    print('total', len(data))
    if args.input_file.endswith(f'{args.part_index}-{args.part_total}.json'):
        start, end = 0, len(data)
    else:
        start, end = get_part_lines(len(data), args.part_index, args.part_total)
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

    free_gpus()
    model = KeyeModel(args.model_path)
    res = {}
    fail_cnt = 0
    prompt_ = 'Have there been any objects in last few frames that do not exist in the first frame of the video? Only answer Yes or No./no_think'
    prompt = 'Have there been any objects in last few frames that do not exist in the first frame? Only answer Yes or No. For example, if one or more persons or characters appear in the last few frames, the answer should be Yes. If the objects that appear in the last few frames are attached to the human bodies or environments, the answer should be No./no_think'
    with open(os.path.join(args.save_dir, f'{args.part_index}-{args.part_total}.txt'), 'a') as f:
        for item in tqdm(data[start:end]):
            if item['source_id'] in exist: continue
            try:
                if args.video_path_key in item:
                    # video_path = item[args.video_path_key].replace('apdcephfs_gy2', 'apdcephfs_toc_gy2')
                    video_path = item[args.video_path_key]
                else:
                    video_path = os.path.join(args.video_root, f'{item["source_id"]}.mp4')
                if not try_download(item['cos_signed_url'], video_path):
                    fail_cnt += 1
                    continue
                
                # clip to 5s
                # images, fps = extract_video(video_path, max_duration=5)
                # out_path = os.path.join('./data/clip5s', f'{item["source_id"]}.mp4')
                # out = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=6)
                # for image in images:
                #     out.append_data(image)
                # out.close()
                # video_path = out_path

                output_text = model.infer(video_path=video_path, prompt=prompt)
                out = process_text(output_text)
                f.write('\t'.join([item['source_id'], '1' if out else '0']) + '\n')
                f.flush()
            except Exception as e:
                fail_cnt += 1
                logging.exception(e)
                print('fail', item)
    print('finish', args.part_index, 'fail', fail_cnt)

def save_final_result():
    save_path = './appears/appears3d'
    file_list = os.listdir(save_path)
    res = {}
    for file in file_list:
        with open(os.path.join(save_path, file), 'r') as f:
            # part = json.load(f)
            part = f.readlines()
        for line in part:
            source_id, appear_new_object = line.strip().split('\t')
            res[source_id] = int(appear_new_object)
    
    input_file_list = ['/apdcephfs_cq10/share_1367250/ryanyuwang/tarsier/AniCaption/data/anime_video_3D_motion_filter_21w.json']
    data = []
    for input_file in input_file_list:
        with open(input_file, 'r') as f:
            data.extend(json.load(f))
    print('total', len(data))

    save_list = []
    for line in tqdm(data):
        key = line['source_id']
        if key not in res: continue
        appear_new_object = res[key]
        if appear_new_object == 0:
            save_list.append(line)
    print('filtered', len(save_list), 'ratio', len(save_list) / len(data))
    save_file = '/apdcephfs_cq10/share_1367250/ryanyuwang/tarsier/AniCaption/data/anime_video_3D_motion_appear_filter.json'
    with open(save_file, 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

def revise():
    video_path = './demom'
    data = os.listdir(video_path)
    print(len(data))
    prompt = 'Have there been any objects in last few frames that do not exist in the first frame? Only answer Yes or No. For example, if one or more persons or characters appear in the last few frames, the answer should be Yes. If the objects that appear in the last few frames are attached to the human bodies or environments, the answer should be No./no_think'
    model = KeyeModel('./models/Keye-VL-1_5-8B')
    save_list = []
    total = 0
    for item in tqdm(data):
        output_text = model.infer(video_path=os.path.join(video_path, item),
                prompt=prompt)
        origin = output_text
        if '</analysis>' in output_text:
            output_text = output_text.split('</analysis>')[-1]
        if '<answer>' in output_text:
            output_text = output_text.split('<answer>')[-1]
            output_text = output_text.split('</answer>')[0]
        if 'the answer is ' in output_text:
            output_text = output_text.split('the answer is ')[-1]
        if r'\boxed{' in output_text:
            output_text = output_text.split(r'\boxed{')[1]

        if output_text.startswith('Yes'):
            res = 1
        elif output_text.startswith('No'):
            res = 0
        else:
            res = 0
            print('not support')
        total += res
        print(item, origin, res)
    print(total, len(data), total / len(data))

def revise_qwenvl():
    video_path = './demom'
    data = os.listdir(video_path)
    print(len(data))
    prompt = 'Have there been any objects in last few frames that do not exist in the first frame of the video? Only answer Yes or No.'
    model = QwenVLModel('./models/Qwen2.5-VL-32B-Instruct')
    save_list = []
    total = 0
    for item in tqdm(data):
        output_text = model.infer(video_path=os.path.join(video_path, item),
                prompt=prompt)
        origin = output_text
        # if '</analysis>' in output_text:
        #     output_text = output_text.split('</analysis>')[-1]
        # if '<answer>' in output_text:
        #     output_text = output_text.split('<answer>')[-1]
        #     output_text = output_text.split('</answer>')[0]
        # if 'the answer is ' in output_text:
        #     output_text = output_text.split('the answer is ')[-1]
        # if r'\boxed{' in output_text:
        #     output_text = output_text.split(r'\boxed{')[1]

        if output_text.startswith('Yes'):
            res = 1
        elif output_text.startswith('No'):
            res = 0
        else:
            res = 0
            print('not support')
        total += res
        print(item, origin, res)
    print(total, len(data), total / len(data))

if __name__ == '__main__':
    main()
