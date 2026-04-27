import numpy as np
import time
import os
import json
from tqdm import tqdm
import random
import argparse
import torch
from multiprocessing import Pool

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

def load_embed(data, path, url_key, caption_key):
    res = {}
    for item in tqdm(data):
        emb_path = os.path.join(path, item['source_id'] + '.pt')
        if not os.path.exists(emb_path): continue
        emb = torch.load(emb_path, weights_only=True)
        res[item['source_id']] = [emb.numpy(), item[url_key], item[caption_key]]
    return res

def parallel_load_data(data, path, url_key, caption_key, n_parallels=160):
    pool = Pool(processes=n_parallels) # 创建4个进程
    results = []
    for i in range(n_parallels):
        start, end = get_part_lines(len(data), i, n_parallels)
        results.append(pool.apply_async(load_embed, (data[start:end], path, url_key, caption_key)))
    pool.close() # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join() # 等待进程池中的所有进程执行完毕
    loaded_data = {}
    for res in results:       
        loaded_data.update(res.get())
    return loaded_data

def pipeline_parallel(tag, json_list, embed_path_list, url_key='cos_signed_url', caption_key='tarsier_long_motion_caption_zh'):
    if os.path.exists(f'embeds/embeds_{tag}.npy') and os.path.exists(f'embeds_name/embeds_name_{tag}.txt'): return
    assert len(json_list) == len(embed_path_list)
    file_map = dict(zip(json_list, embed_path_list))
    emb_list = []
    url_map = {}
    name_list = []
    for file, path in file_map.items():
        with open(file, 'r') as f:
            data = json.load(f)
        loaded_data = parallel_load_data(data, path, url_key, caption_key)
        for key, info in loaded_data.items():
            emb_list.append(info[0])
            url_map[key] = [info[1], info[2]]
            name_list.append(key)
    res = np.stack(emb_list, axis=0)
    np.save(f'embeds/embeds_{tag}.npy', res)

    with open(f'url_map/url_map_{tag}.json', 'w') as f:
        json.dump(url_map, f)

    with open(f'embeds_name/embeds_name_{tag}.txt', 'w') as f:
        f.write('\n'.join(name_list))

def pipeline(tag, json_list, embed_path_list, url_key='cos_signed_url', caption_key='tarsier_long_motion_caption_zh'):
    assert len(json_list) == len(embed_path_list)
    file_map = dict(zip(json_list, embed_path_list))
    emb_list = []
    url_map = {}
    name_list = []
    for file, path in file_map.items():
        with open(file, 'r') as f:
            data = json.load(f)
        for item in tqdm(data):
            emb_path = os.path.join(path, item['source_id'] + '.pt')
            if not os.path.exists(emb_path): continue
            emb = torch.load(emb_path, weights_only=True)
            emb_list.append(emb.numpy())

            url_map[item['source_id']] = [item[url_key], item[caption_key]]
            name_list.append(item['source_id'])
    res = np.stack(emb_list, axis=0)
    np.save(f'embeds/embeds_{tag}.npy', res)

    with open(f'url_map/url_map_{tag}.json', 'w') as f:
        json.dump(url_map, f)

    with open(f'embeds_name/embeds_name_{tag}.txt', 'w') as f:
        f.write('\n'.join(name_list))

def kmeans_centroid_sample_train_test(tag, feature_dim, num_clusters, n_sample_per_cluster, input_file_list):
    import faiss

    res = faiss.StandardGpuResources()
    kmeans = faiss.Clustering(feature_dim, num_clusters)
    kmeans.niter = 50
    kmeans.max_points_per_centroid = 1024
    kmeans.min_points_per_centroid = 128
    kmeans.spherical = True
    kmeans.verbose = True
    kmeans.seed = 1234
    kmeans.nredo = 5
    gpu_index = faiss.GpuIndexFlatIP(res, feature_dim)

    embeds = np.load(f'embeds/embeds_{tag}.npy')
    t1 = time.time()
    kmeans.train(embeds, gpu_index)
    print('kmeans', time.time() - t1)
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(num_clusters, feature_dim)
    index_cpu = faiss.IndexFlatIP(feature_dim)
    index_cpu.add(embeds)

    search_embeds = centroids
    distances, labels = index_cpu.search(search_embeds, n_sample_per_cluster)
    labels = labels.flatten()

    np.save(f'embeds/centroids_{tag}.npy', centroids)
    with open(f'embeds_name/embeds_name_{tag}.txt', 'r') as f:
        embed_list = f.readlines()
        embed_list = [embed.strip() for embed in embed_list]
    test_samples = set([embed_list[int(l)] for l in labels])

    data = []
    for input_file in input_file_list:
        with open(input_file, 'r') as f:
            data.extend(json.load(f))
    
    train, test = [], []
    dedup = set([])
    for item in data:
        if item['source_id'] in test_samples:
            if item['source_id'] not in dedup:
                test.append(item)
                dedup.add(item['source_id'])
            else:
                print('duplicates', item['source_id'])
        else:
            train.append(item)
    print('train', len(train), 'test', len(test))
    file_path, file_name = os.path.split(input_file_list[0])
    file_name, file_ext = os.path.splitext(file_name)
    with open(os.path.join(file_path, file_name + '_train' + file_ext), 'w') as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(os.path.join(file_path, file_name + '_test' + file_ext), 'w') as f:
        json.dump(test, f, indent=4, ensure_ascii=False)

def make_sample(tag, feature_dim, num_clusters, n_sample_per_cluster, input_file_list):
    import faiss

    res = faiss.StandardGpuResources()
    kmeans = faiss.Clustering(feature_dim, num_clusters)
    kmeans.niter = 50
    kmeans.max_points_per_centroid = 10240
    kmeans.min_points_per_centroid = 128
    kmeans.spherical = True
    kmeans.verbose = True
    kmeans.seed = 1234
    kmeans.nredo = 5
    gpu_index = faiss.GpuIndexFlatIP(res, feature_dim)

    embeds = np.load(f'embeds/embeds_{tag}.npy')
    t1 = time.time()
    kmeans.train(embeds, gpu_index)
    print('kmeans', time.time() - t1)
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(num_clusters, feature_dim)
    index_cpu = faiss.IndexFlatIP(feature_dim)
    index_cpu.add(centroids)

    search_embeds = embeds
    distances, labels = index_cpu.search(search_embeds, 1)
    labels = labels.flatten()
    np.save(f'labels/labels_{tag}_kmeans.npy', labels)

    lcnt = dict.fromkeys(np.unique(labels), 0)
    for l in labels:
        lcnt[l] += 1
    print(tag, num_clusters, len(lcnt), lcnt)

    labels = labels.astype(int).tolist()
    with open(f'embeds_name/embeds_name_{tag}.txt', 'r') as f:
        embed_list = f.readlines()
        embed_list = [embed.strip() for embed in embed_list]
    assert len(embed_list) == len(labels)
    embed_map = dict(zip(embed_list, labels))

    data = []
    for input_file in input_file_list:
        with open(input_file, 'r') as f:
            data.extend(json.load(f))
    print('total', len(data))
    cluster_map = {}
    for item in data:
        key = item['source_id']
        if key not in embed_map: continue
        cluster_id = embed_map[key]
        item['cluster_id'] = cluster_id
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(item)

    save_list = []
    for cluster_id, cluster_list in cluster_map.items():
        if len(cluster_list) < 10:
            print('ignore', cluster_id, len(cluster_list))
            continue
        cluster_samples = random.sample(cluster_list, n_sample_per_cluster)
        save_list.extend(cluster_samples)
    print(len(save_list))
    file_path, file_name = os.path.split(input_file_list[0])
    file_name, file_ext = os.path.splitext(file_name)
    with open(os.path.join(file_path, file_name + '_sample' + file_ext), 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        help="任务标记",
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="输入文件 json",
    )
    parser.add_argument(
        "--embed_paths",
        type=str,
        nargs="+",
        help="embedding 路径",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536,
        help="特征维度",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=100,
        help="聚类数量",
    )
    parser.add_argument(
        "--n_sample_per_cluster",
        type=int,
        default=1,
        help="单聚类采样数量，总采样数量=num_clusters*n_sample_per_cluster",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='sample',
        help="任务类型",
    )
    args = parser.parse_args()
    pipeline_parallel(args.tag, args.input_files, args.embed_paths)
    if args.task == 'train_test':
        kmeans_centroid_sample_train_test(args.tag, args.feature_dim, args.num_clusters, args.n_sample_per_cluster, args.input_files)
    elif args.task == 'sample':
        make_sample(args.tag, args.feature_dim, args.num_clusters, args.n_sample_per_cluster, args.input_files)
    else:
        print('not supported', args.task)
