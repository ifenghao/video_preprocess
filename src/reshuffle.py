import glob
import json
import os
import pandas as pd
import random
import argparse

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def reshuffle(input_file_list, exist_path, save_path, parts):
    print('inputs', input_file_list)
    file_list = glob.glob(os.path.join(exist_path, '*.txt'))
    exist = {}
    for file in file_list:
        with open(os.path.join(file), 'r') as f:
            part = f.readlines()
        print('load', file, len(part))
        for line in part:
            if len(line.strip().split('\t')) <= 1:
                continue
            source_id = line.strip().split('\t')[0]
            exist[source_id] = 1
    print('exist', len(exist))
    
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
        print('load', input_file, len(part))
        data.extend(part)

    print('total', len(data))
    save_list = []
    for item in data:
        if item['source_id'] not in exist:
            save_list.append(item)

    print('rest', len(save_list))
    random.shuffle(save_list)
    os.makedirs(save_path, exist_ok=True)
    for i in range(parts):
        start, end = get_part_lines(len(save_list), i, parts)
        with open(f'{save_path}/{i}-{parts}.json', 'w') as f:
            json.dump(save_list[start:end], f, indent=4, ensure_ascii=False)
        print(i, end - start)

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
        "--exist_path",
        type=str,
        help="已存在的文件路径",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=0,
        help="图像文件索引编号",
    )
    args = parser.parse_args()
    reshuffle(args.input_files, args.exist_path, args.save_path, args.nsplits)