import sys
sys.path.append('./pips2')
import time
import numpy as np
from nets.pips2 import Pips
import utils.basic
import torch
import torch.nn.functional as F
import argparse
import cv2
import os
import json
from PIL import Image
import logging
from tqdm import tqdm
import requests
import random
import gc

def extract_video(video_name, force_fps=8, max_duration=5): # fps 固定
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), f'Failed to load video file {video_name}'
    # get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(np.ceil(fps))
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
    return imgs

def get_side(H, W, N):
    if H > W:
        width = N
        height = int(N * H / W)
    else:
        height = N
        width = int(N * W / H)
    return height, width

def run_model(model, rgbs, N=64, iters=16, sw=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # pick N points to track; we'll use a uniform grid
    # N_ = np.sqrt(N).round().astype(np.int32)
    N_H, N_W = get_side(H, W, N)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_H, N_W, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_H-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_W-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]
    return trajs_e

def load_model(model_path):
    model = Pips(stride=8).cuda()
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model

def infer(model, video_path, N=48, iters=16, max_resolution=512, force_fps=8):
    rgbs = extract_video(video_path, force_fps)
    rgb_seq = np.stack(rgbs, axis=0) # S,H,W,3
    _, height, width, C = rgb_seq.shape
    rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
    if max_resolution is not None:
        ratio = min(max_resolution / width, max_resolution / height, 1) # 较大尺寸进行缩小
        rgb_seq = F.interpolate(rgb_seq, (int(height * ratio), int(width * ratio)), mode='bilinear')
    rgb_seq = rgb_seq.unsqueeze(0) # 1,S,3,H,W

    with torch.no_grad():
        trajs_e = run_model(model, rgb_seq, N=N, iters=iters)
    return trajs_e

def calc_velocity(trajs_e):
    B,S,N,_ = trajs_e.shape
    trajs_e0 = trajs_e[:,:-1] # B,S-1,N,2
    trajs_e1 = trajs_e[:,1:] # B,S-1,N,2
    velocity = trajs_e1-trajs_e0 # B,S-1,N,2
    velocity = torch.cat([torch.zeros(B,1,N,2).cuda(), velocity], dim=1) # B,S,N,2
    return velocity

def calc_acceleration(velocity):
    B,S,N,_ = velocity.shape
    velocity0 = velocity[:, 1:-1]  # B,S-2,N,2
    velocity1 = velocity[:,2:] # B,S-2,N,2
    acceleration = velocity1-velocity0 # B,S-2,N,2
    acceleration = torch.cat([torch.zeros(B,1,N,2).cuda(), torch.zeros(B,1,N,2).cuda(), acceleration], dim=1) # B,S,N,2
    return acceleration

def calc_static_mask(trajs_e, mean_thres, std_thres):
    diff = trajs_e[:, 1:] - trajs_e[:, :1]  # B,S-1,N,2
    dist = (diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2) ** 0.5  # B,S-1,N
    std, mean = torch.std_mean(dist, dim=1, keepdim=False)  # B,N
    mask = torch.where(torch.logical_or(mean < mean_thres, std < std_thres), 0.0, 1.0)
    return mask

def calc_static_velocity_mask(velocity, mean_thres, std_thres):
    dist = (velocity[:, :, :, 0] ** 2 + velocity[:, :, :, 1] ** 2) ** 0.5  # B,S-1,N
    std, mean = torch.std_mean(dist, dim=1, keepdim=False)  # B,N
    mask = torch.where(torch.logical_or(mean < mean_thres, std < std_thres), 0.0, 1.0)
    return mask

def grid_search(trajs_e):
    res = []
    for mean in np.arange(0, 10, 0.5):
        row = []
        for std in np.arange(0, 10, 0.5):
            mask = calc_static_velocity_mask(trajs_e, mean, std)
            row.append(mask.squeeze().cpu().numpy().reshape([48, -1]) * 255)
        row = np.concatenate(row, axis=1)
        res.append(row)
    res = np.concatenate(res, axis=0)
    return res.astype(np.uint8)

def get_res(trajs_e, mean_thres=1., std_thres=1.):
    velocity = calc_velocity(trajs_e)
    mask = calc_static_velocity_mask(velocity, mean_thres, std_thres)
    motion_area = mask.sum()
    acceleration = calc_acceleration(velocity)

    v = (velocity[:,:,:,0]**2 + velocity[:,:,:,1]**2)**0.5
    v = v.mean([1])
    v_masked = v * mask
    res_v = v_masked.sum() / (motion_area + 1e-8)
    res_v_max = torch.quantile(v, 0.995)

    acc = (acceleration[:,:,:,0]**2 + acceleration[:,:,:,1]**2)**0.5
    acc = acc.mean([1])
    acc_masked = acc * mask
    res_acc = acc_masked.sum() / (motion_area + 1e-8)
    res_acc_max = torch.quantile(acc, 0.995)

    motion_ratio = motion_area / mask.numel()
    return res_v.cpu().numpy(), res_v_max.cpu().numpy(), res_acc.cpu().numpy(), res_acc_max.cpu().numpy(), motion_ratio.cpu().numpy()

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
        default='',
        help="输入文件 json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./models/pips2_weights.pth',
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
        default='./data/movie_download_videos',
    )
    parser.add_argument(
        "--video_path_key",
        type=str,
        default='video_path',
    )
    parser.add_argument(
        "--is_3d",
        type=int,
        default=0,
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
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    if args.input_file.endswith(f'{args.part_index}-{args.part_total}.json'):
        start, end = 0, len(data)
    else:
        start, end = get_part_lines(len(data), args.part_index, args.part_total)
    model = load_model(args.model_path)

    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, f'{args.part_index}-{args.part_total}.txt')

    file_list = os.listdir(args.save_path)
    exist = {}
    for file in file_list:
        with open(os.path.join(args.save_path, file), 'r') as f:
            part = f.readlines()
        for line in part:
            line = line.strip().split('\t')
            if len(line) != 6: continue
            source_id = line[0]
            exist[source_id] = 1
        print('load exist', len(exist))
    print(len(data), start, end)
    thres = 1.0 if args.is_3d else 2.5
    free_gpus()
    fail_cnt = 0
    with open(save_file, 'a') as f:
        for item in tqdm(data[start:end]):
            if item['source_id'] in exist: continue
            try:
                if args.video_path_key in item:
                    video_path = item[args.video_path_key]
                    if not os.path.exists(video_path):
                        video_path = os.path.join(args.video_root, item['source_id'] + '.mp4')
                else:
                    video_path = os.path.join(args.video_root, item['source_id'] + '.mp4')

                # if not try_download(item['cos_signed_url'], video_path):
                #     print('download fail', item['source_id'])
                #     fail_cnt += 1
                #     continue
                trajs_e = infer(model, video_path)
                v, v_max, acc, acc_max, motion_ratio = get_res(trajs_e, thres, thres)
                line = '\t'.join([item['source_id'], str(v), str(v_max), str(acc), str(acc_max), str(motion_ratio)])
                f.write(line + '\n')
                f.flush()
            except Exception as e:
                fail_cnt += 1
                logging.exception(e)
                print('fail', item)

            # gs_mask = grid_search(trajs_e)
            # Image.fromarray(gs_mask).save(f'./trajs/{os.path.splitext(file)[0]}.png')
            # trajs_e = trajs_e.cpu().numpy()
            # np.save(f'./trajs/{file}.npy', trajs_e)
            # print(file, trajs_e.shape, acc, acc_max, time.time() - start)
    print('finish', args.part_index, 'failed', fail_cnt)

if __name__ == '__main__':
    main()
