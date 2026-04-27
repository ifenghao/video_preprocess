from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from moviepy import AudioFileClip, VideoFileClip
import os
import time
import torch
import glob
import random
from tqdm import tqdm
from multiprocessing import Pool

root_audio_path = './audios'
root_video_path = './video_clips'

pipeline = Pipeline.from_pretrained('./models/speaker-diarization-3.1', token="xxx")
pipeline.to(torch.device("cuda"))

def audio_segment(audio_path):

    with ProgressHook() as hook:
        output = pipeline(audio_path, hook=hook)
    print(output.speaker_diarization)
    res = []
    for turn, speaker in output.speaker_diarization:
        res.append((turn.start, turn.end))
    return res

def video2audio(video_path):
    audio_path = os.path.join(root_audio_path, os.path.basename(video_path.replace('.mp4', '.wav')))
    audio_clip = AudioFileClip(video_path)
    audio_clip.write_audiofile(audio_path)
    return audio_path

def cut_clip(video_clip, start, end, idx, video_name):
    save_path = os.path.join(root_video_path, video_name.replace('.mp4', f'_{idx}_{start:.1f}_{end:.1f}.mp4'))
    if os.path.exists(save_path):
        return save_path
    video_clip = video_clip.subclipped(start, end)
    video_clip.write_videofile(save_path)
    return save_path

def run(video_path):
    start = time.time()
    audio_path = video2audio(video_path)
    res = audio_segment(audio_path)
    video_clip = VideoFileClip(video_path)
    video_end = video_clip.end
    pre_start = None
    for idx, (start, end) in enumerate(res):
        if pre_start is not None:
            if end - pre_start > 5:
                cut_clip(video_clip, pre_start, min(end, video_end), idx, os.path.basename(video_path))
        else:
            if end - start < 5:
                pre_start = start
            elif end - start > 10:
                nums = (int(end - start) - 1) // 10 + 1
                clip_len = (end - start) / nums
                for i in range(nums):
                    cut_clip(video_clip, start + i * clip_len, min(start + (i + 1) * clip_len, video_end), idx, os.path.basename(video_path))
            else:
                cut_clip(video_clip, start, min(end, video_end), idx, os.path.basename(video_path))

    print(f'cost time: {time.time() - start}')
    return len(res)

def run_task(data, idx):
    for item in tqdm(data):
        run(item)
    print(f'finish {idx}')

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def run_parallel(data, n_parallels=30):
    pool = Pool(processes=n_parallels) # 创建4个进程
    results = []
    for i in range(n_parallels):
        start, end = get_part_lines(len(data), i, n_parallels)
        results.append(pool.apply_async(run_task, (data[start:end], i)))
    pool.close() # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join() # 等待进程池中的所有进程执行完毕

def main_parallel(path, sample_num=None):
    video_paths = glob.glob(path)
    print('total', len(video_paths))
    # if sample_num is not None:
    #     video_paths = random.sample(video_paths, sample_num)
    # run_parallel(video_paths)
    total = 0
    for video_path in video_paths:
        cnt = run(video_path)
        total += cnt
    print(total)

if __name__ == '__main__':
    main_parallel('data/*.mp4')