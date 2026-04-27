import os
import json
import logging
from tqdm import tqdm
import argparse
import pandas as pd
import random

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def save_final_result(save_path, input_file_list, save_file, nsplits, split_path):
    file_list = os.listdir(save_path)
    res = {}
    for file in file_list:
        with open(os.path.join(save_path, file), 'r') as f:
            part = f.readlines()
        for line in part:
            source_id, motion_stength_affine, motion_stength_origin = line.strip().split('\t')
            res[source_id] = [float(motion_stength_affine), float(motion_stength_origin)]
    
    data = []
    for input_file in input_file_list:
        ext = input_file.split('.')[-1]
        if ext == 'json':
            with open(input_file, 'r') as f:
                part = json.load(f)
        elif ext == 'csv':
            part = pd.read_csv(input_file, sep='\t')
            part = part.to_dict('records')
        else:
            part = []
        data.extend(part)

    motion_list = []
    for line in tqdm(data):
        key = line['source_id']
        if key not in res: continue
        motion_stength_affine, motion_stength_origin = res[key]
        line['motion_stength_affine'] = motion_stength_affine
        line['motion_stength_origin'] = motion_stength_origin
        motion_list.append(line)
    print('total', len(motion_list))

    save_list = []
    for line in tqdm(motion_list):
        if is_retained(line):
            save_list.append(line)
    print('filtered', len(save_list), 'ratio', len(save_list) / len(motion_list))
    with open(save_file, 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

    if nsplits > 0:
        os.makedirs(split_path, exist_ok=True)
        for i in range(nsplits):
            start, end = get_part_lines(len(save_list), i, nsplits)
            with open(f'{split_path}/{i}-{nsplits}.json', 'w') as f:
                json.dump(save_list[start:end], f, indent=4, ensure_ascii=False)
            print(f'finish {i}/{nsplits} length {end - start}')

def _is_retained(item):
    if item['motion_stength_affine'] < 1.7: return False
    if item['motion_stength_affine'] * 1.5 > item['motion_stength_origin']: return True
    # if item['motion_stength_affine'] > 5 and item['motion_stength_affine'] * 2 < item['motion_stength_origin']: return False
    if item['motion_stength_affine'] * 5 < item['motion_stength_origin']: return False
    return True

def _is_retained2(item):
    if item['motion_stength_affine'] < 8: return False
    if item['motion_stength_affine'] * 1.5 > item['motion_stength_origin']: return True
    if item['motion_stength_affine'] * 10 < item['motion_stength_origin']: 
        return False
    else:
        if item['motion_stength_origin'] > 50: return False
    return True

def is_retained(item):
    if item['motion_stength_affine'] < 5: return False  # anime: 5 movie: 8
    if item['motion_stength_affine'] * 1.5 > item['motion_stength_origin']: return True
    if item['motion_stength_affine'] * 10 < item['motion_stength_origin']: 
        return False
    return True

def save_track_result(save_path, input_file_list, save_file, nsplits, split_path):
    file_list = os.listdir(save_path)
    res = {}
    for file in file_list:
        with open(os.path.join(save_path, file), 'r') as f:
            part = f.readlines()
        for line in part:
            source_id, v, v_max, a, a_max, motion_ratio = line.strip().split('\t')
            res[source_id] = [float(v), float(v_max), float(a), float(a_max), float(motion_ratio)]
    
    data = []
    for input_file in input_file_list:
        ext = input_file.split('.')[-1]
        if ext == 'json':
            with open(input_file, 'r') as f:
                part = json.load(f)
        elif ext == 'csv':
            part = pd.read_csv(input_file, sep='\t')
            part = part.to_dict('records')
        else:
            part = []
        data.extend(part)

    motion_list = []
    for line in tqdm(data):
        key = line['source_id']
        if key not in res: continue
        v, v_max, a, a_max, motion_ratio = res[key]
        line['velocity'] = v
        line['velocity_max'] = v_max
        line['acceleration'] = a
        line['acceleration_max'] = a_max
        line['motion_ratio'] = motion_ratio
        motion_list.append(line)
    print('total', len(motion_list))

    save_list = []
    for line in tqdm(motion_list):
        # if is_retained_track(line):
        if judge_track(line):
            save_list.append(line)
    print('filtered', len(save_list), 'ratio', len(save_list) / len(motion_list))
    with open(save_file, 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

    if nsplits > 0:
        # random.shuffle(save_list)
        os.makedirs(split_path, exist_ok=True)
        for i in range(nsplits):
            start, end = get_part_lines(len(save_list), i, nsplits)
            with open(f'{split_path}/{i}-{nsplits}.json', 'w') as f:
                json.dump(save_list[start:end], f, indent=4, ensure_ascii=False)
            print(f'finish {i}/{nsplits} length {end - start}')

def is_retained_track(item):
    if item['acceleration'] < 1.5 and item['acceleration_max'] < 1.5: return False # anime: 2.5 movie: 1.5
    if item['motion_ratio'] < 0.04: return False
    return True

def judge_track(item):
    if (item['acceleration'] < 1.5 and item['acceleration_max'] < 1.5) or item['motion_ratio'] < 0.04:
        item['speed'] = 'slow'
    elif item['acceleration'] > 5 and item['acceleration_max'] > 8:
        item['speed'] = 'fast'
        # return False
    else:
        item['speed'] = 'medium'
    return True

def save_appear_result(save_path, input_file_list, save_file, nsplits, split_path):
    file_list = os.listdir(save_path)
    res = {}
    for file in file_list:
        with open(os.path.join(save_path, file), 'r') as f:
            part = f.readlines()
        for line in part:
            source_id, appear_new_object = line.strip().split('\t')
            res[source_id] = int(appear_new_object)
    
    data = []
    for input_file in input_file_list:
        ext = input_file.split('.')[-1]
        if ext == 'json':
            with open(input_file, 'r') as f:
                part = json.load(f)
        elif ext == 'csv':
            part = pd.read_csv(input_file, sep=',')
            part = part.to_dict('records')
        else:
            part = []
        data.extend(part)
    print('total', len(data))

    save_list = []
    for line in tqdm(data):
        key = line['source_id']
        if key not in res: continue
        appear_new_object = res[key]
        if appear_new_object == 0:
            save_list.append(line)
    print('filtered', len(save_list), 'ratio', len(save_list) / len(data))
    with open(save_file, 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

    if nsplits > 0:
        # random.shuffle(save_list)
        os.makedirs(split_path, exist_ok=True)
        for i in range(nsplits):
            start, end = get_part_lines(len(save_list), i, nsplits)
            with open(f'{split_path}/{i}-{nsplits}.json', 'w') as f:
                json.dump(save_list[start:end], f, indent=4, ensure_ascii=False)
            print(f'finish {i}/{nsplits} length {end - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="输入文件 json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='./motions',
        help="如果需要保存的路径",
    )
    parser.add_argument(
        "--filtered_save_file",
        type=str,
        help="图像文件索引编号",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=0,
        help="图像文件索引编号",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default='.',
        help="图像文件索引编号",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="任务",
    )
    args = parser.parse_args()
    if args.task == 'motion':
        save_final_result(
            save_path=args.save_path, 
            input_file_list=args.input_files, 
            save_file=args.filtered_save_file,
            nsplits=args.nsplits, 
            split_path=args.split_path)
    elif args.task == 'track':
        save_track_result(
            save_path=args.save_path, 
            input_file_list=args.input_files, 
            save_file=args.filtered_save_file,
            nsplits=args.nsplits, 
            split_path=args.split_path)
    elif args.task == 'appear':
        save_appear_result(
            save_path=args.save_path, 
            input_file_list=args.input_files, 
            save_file=args.filtered_save_file,
            nsplits=args.nsplits, 
            split_path=args.split_path)
