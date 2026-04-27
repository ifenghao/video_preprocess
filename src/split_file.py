import json
import os
import argparse
import pandas as pd
import numpy as np

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def split(args):
    ext = args.input_file.split('.')[-1]
    if ext == 'json':
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        np.random.shuffle(data)
    elif ext == 'csv':
        data = pd.read_csv(args.input_file, sep='\t')
        if args.force_json:
            data = data.to_dict('records')
            np.random.shuffle(data)
        else:
            data = data.sample(frac=1, random_state=np.random.randint(0, 10000))
    print('total', len(data))

    os.makedirs(args.split_path, exist_ok=True)
    for i in range(args.parts):
        start, end = get_part_lines(len(data), i, args.parts)
        if ext == 'json' or args.force_json:
            with open(f'{args.split_path}/{i}-{args.parts}.json', 'w') as f:
                json.dump(data[start:end], f, indent=4, ensure_ascii=False)
        elif ext == 'csv':
            data[start:end].to_csv(f'{args.split_path}/{i}-{args.parts}.csv', sep='\t', index=False)
        print(f'finish {i}/{args.parts} length {end - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="输入文件",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        help="输出路径",
    )
    parser.add_argument(
        "--parts",
        type=int,
        help="embedding 路径",
    )
    parser.add_argument(
        "--force_json",
        action='store_true',
        help="强制使用json格式",
    )
    args = parser.parse_args()
    split(args)