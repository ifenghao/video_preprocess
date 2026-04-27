import sys
sys.path.append('./unimatch')
from unimatch.unimatch import UniMatch
import torch
import argparse
import numpy as np
import os
import cv2
import json
import logging
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import requests
import imageio
from utils.flow2d_utils import iteratively_fit_affine_transform
from utils.flow_viz import flow_to_image

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation datasets')
    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding or resizing')

    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluation after training done')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')
    parser.add_argument('--val_things_clean_only', action='store_true')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed methic when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo if only visualization is needed')

    # inference on images or videos
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_video', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size for the input to the network')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--pred_bwd_flow', action='store_true',
                        help='predict backward flow only')
    parser.add_argument('--fwd_bwd_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--concat_flow_img', action='store_true')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # misc
    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time')

    parser.add_argument('--debug', action='store_true')

    # parallel
    parser.add_argument(
        "--input_file",
        type=str,
        default='',
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

    return parser

def load_model(args):
    args.distributed = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)

    if torch.cuda.device_count() > 1:
        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

        model_without_ddp = model.module
    else:
        model_without_ddp = model

    print('Load checkpoint: %s' % args.resume)

    loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.resume, map_location=loc)

    model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
    return model_without_ddp

def extract_video(video_name, max_resolution=None, force_fps=8, max_duration=5): # fps 固定
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), f'Failed to load video file {video_name}'
    # get video info
    width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(np.ceil(fps))
    interval = (fps - 1) // force_fps + 1
    # print('video size (hxw): %dx%d' % (height, width))
    # print('fps: %d' % fps, interval)
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
        
        if max_resolution is not None:
            ratio = min(max_resolution / width, max_resolution / height, 1) # 较大尺寸进行缩小
            img = cv2.resize(img, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LANCZOS4)
        # to rgb format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        frame_count += 1
        if len(imgs) >= max_frames:
            break
    return imgs, fps

@torch.no_grad()
def inference_flow(model,
                   inference_video,
                   padding_factor=8,
                   inference_size=None,
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   reverse_video=False,
                   max_resolution=768,
                   force_fps=8
                   ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fixed_inference_size = inference_size
    transpose_img = False

    filenames, fps = extract_video(inference_video, max_resolution, force_fps)  # list of [H, W, 3]

    # print('%d images found' % len(filenames))

    if reverse_video:
        filenames = filenames[::-1]

    batch_flow_preds = []

    for test_id in range(0, len(filenames) - 1):
        # if (test_id + 1) % 50 == 0:
        #     print('predicting %d/%d' % (test_id + 1, len(filenames)))

        if inference_video is not None:
            image1 = filenames[test_id]
            image2 = filenames[test_id + 1]
        else:
            image1 = frame_utils.read_gen(filenames[test_id])
            image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             pred_bidir_flow=pred_bidir_flow,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        batch_flow_preds.append(flow_pr.unsqueeze(1))
    
    batch_flow = torch.cat(batch_flow_preds, axis=1)  # [B, T, 2, H, W]
    return batch_flow

def get_motion_strength(flow_batch):
    flow_batch_ori = flow_batch.clone()
    B, T, _, H, W = flow_batch.size()
    ##### 解出来这 T 个帧间刚性变换
    fitted_results_stages = iteratively_fit_affine_transform(flow_batch, iter=2)
    motion_strength = fitted_results_stages[-1]["fitted_delta"]
    motion_strength = ((motion_strength[:,:,0]**2+motion_strength[:,:,1]**2)**0.5).mean([2,3])
    motion_strength= ((motion_strength.sum(1) - motion_strength.max(1)[0]) / (motion_strength.shape[1]-1)).detach()*50*(320*320)/(H*W)#(320*320)/(H*W)确保不同分辨率视频结果可比

    motion_strength_ori = ((flow_batch_ori[:,:,0]**2+flow_batch_ori[:,:,1]**2)**0.5).mean([2,3])
    motion_strength_ori = ((motion_strength_ori.sum(1) - motion_strength_ori.max(1)[0]) / (motion_strength_ori.shape[1]-1)).detach()*50*(320*320)/(H*W)

    motion_strength = motion_strength.cpu().numpy().astype(float)[0]
    motion_strength_ori = motion_strength_ori.cpu().numpy().astype(float)[0]
    return motion_strength, motion_strength_ori

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

def main(args):
    ext = args.input_file.split('.')[-1]
    if ext == 'json':
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    elif ext == 'csv':
        data = pd.read_csv(args.input_file, sep='\t')
        data = data.to_dict('records')
    print('total', len(data))
    if args.input_file.endswith(f'{args.part_index}-{args.part_total}.json') or args.input_file.endswith(f'{args.part_index}-{args.part_total}.csv'):
        start, end = 0, len(data)
    else:
        start, end = get_part_lines(len(data), args.part_index, args.part_total)
    model = load_model(args)
    model.eval()
    
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, f'{args.part_index}-{args.part_total}.txt')
    file_list = os.listdir(args.save_path)
    exist = {}
    for file in file_list:
        with open(os.path.join(args.save_path, file), 'r') as f:
            # part = json.load(f)
            part = f.readlines()
        for line in part:
            line = line.strip().split('\t')
            if len(line) != 3: continue
            source_id, motion_stength_affine, motion_stength_origin = line
            exist[source_id] = [float(motion_stength_affine), float(motion_stength_origin)]

    free_gpus()
    fail_cnt = 0
    with open(save_file, 'a') as f:
        for item in tqdm(data[start:end]):
            if item['source_id'] in exist: continue
            try:
                if args.video_path_key in item:
                    video_path = item[args.video_path_key]
                else:
                    video_path = os.path.join(args.video_root, item['source_id'])
                # if not try_download(item['cos_signed_url'], video_path):
                #     print('download fail', item['source_id'])
                #     fail_cnt += 1
                #     continue
                flow_batch = inference_flow(
                    model,
                    inference_video=video_path,
                    padding_factor=args.padding_factor,
                    inference_size=args.inference_size,
                    attn_type=args.attn_type,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list,
                    pred_bidir_flow=args.pred_bidir_flow,
                    pred_bwd_flow=args.pred_bwd_flow,
                    num_reg_refine=args.num_reg_refine,
                    reverse_video=True,
                    max_resolution=512,
                    force_fps=8,
                )
                motion_stength_affine, motion_stength_origin = get_motion_strength(flow_batch)
                line = '\t'.join([item['source_id'], str(motion_stength_affine), str(motion_stength_origin)])
                f.write(line + '\n')
                f.flush()
            except Exception as e:
                logging.exception(e)
                print('fail', item)
    print('finish', args.part_index, 'failed', fail_cnt)

def plot_flow(flow, video_path, save_path):  # B, T, 2, H, W
    flow = flow[0]
    flow = flow.permute(0, 2, 3, 1).cpu().numpy()  # T, H, W, 2
    images = []
    for i in range(len(flow)):
        images.append(flow_to_image(flow[i]))

    ori_imgs, fps = extract_video(video_path, 512, 8)
    ori_imgs = ori_imgs[:-1]
    results = []
    assert len(ori_imgs) == len(images)

    concat_axis = 0 if ori_imgs[0].shape[0] < ori_imgs[0].shape[1] else 1
    for img, flow in zip(ori_imgs, images):
        concat = np.concatenate((img, flow), axis=concat_axis)
        results.append(concat)
    
    imageio.mimwrite(save_path, results, fps=fps, quality=6)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)