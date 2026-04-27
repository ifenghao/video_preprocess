import sys
sys.path.append('./VLM2Vec')
from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
import json
import os
import random
import re
import time
import gradio as gr
import numpy as np
import faiss
import torch
import requests

tag = 'filter113'

with open(f'./url_map/url_map_{tag}.json', 'r') as f:
    url_map = json.load(f)

with open(f'./embeds_name/embeds_name_{tag}.txt', 'r') as f:
    embed_list = f.readlines()
    embed_list = [embed.strip() for embed in embed_list]

# faiss
feature_dim = 1536
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatIP(res, feature_dim)
embeds = np.load(f'./embeds/embeds_{tag}.npy')
gpu_index.add(embeds)
model_path = './models/'

# vlm2vec
model_args = ModelArguments(
    model_name=model_path + 'Qwen2-VL-2B-Instruct',
    checkpoint_path=model_path + 'VLM2Vec-V2.0',
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

def search_index(query_embeds, nums):
    distances, labels = gpu_index.search(query_embeds, nums)
    labels = np.squeeze(labels.astype(int))
    res = []
    for l in labels:
        embed_name = embed_list[l].split('.')[0]
        res.append(url_map[embed_name])
    return res

def search_index_ddp(query_embeds, ddp_nums, nums):
    distances, labels = gpu_index.search(query_embeds, ddp_nums)
    distances = np.squeeze(distances.astype(float))
    labels = np.squeeze(labels.astype(int))

    recall_embeds = embeds[labels]
    kernel_matrix = get_kernel_matrix(recall_embeds, distances)
    ddp_index = dpp(kernel_matrix, nums)
    ddp_labels = labels[ddp_index]
    res = []
    for l in ddp_labels:
        embed_name = embed_list[l].split('.')[0]
        res.append(url_map[embed_name])
    return res

def encode_input(text):
    inputs = processor(text=text, images=None, return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    tgt_output = model(tgt=inputs)["tgt_reps"]
    return tgt_output.detach().float().cpu().numpy()

def test():
    text = 'person is talking'
    query_embeds = encode_input(text)
    res = search_index(query_embeds, 5)
    print(res)

def run_search(text, ddp_length, length):
    query_embeds = encode_input(text)
    if ddp_length > 0:
        res = search_index_ddp(query_embeds, ddp_length, length)
    else:
        res = search_index(query_embeds, length)

    html_str = ''
    for each_result in res:
        if isinstance(each_result, list):
            html_str += '<table border="1"><tr><td><iframe width="512" height="512" src="{}" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td><td>{}</td></tr></table>'.format(each_result[0], each_result[1])
        else:
            html_str += '<div><iframe width="512" height="512" src="{}" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>'.format(each_result)

    # download
    # os.makedirs(f'./videos/', exist_ok=True)
    # path_list = []
    # for url in res:
    #     name = url.split('?')[0].split('/')[-1]
    #     path = f'./videos/{name}'
    #     myfile = requests.get(url)
    #     open(path, 'wb').write(myfile.content)
    #     path_list.append(path)
    return html_str

def dpp(kernel_matrix, max_length, epsilon=1e-10):
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = np.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def get_kernel_matrix(input_embeds, scores):
    n = input_embeds.shape[0]
    similarities = np.dot(input_embeds, input_embeds.T)
    kernel_matrix = scores.reshape((n, 1)) * similarities * scores.reshape((1, n))
    return kernel_matrix

def main(port):
    with gr.Blocks() as app:
        gr.Markdown(
            """
        # 欢迎体验文本视频检索工具！
        请上传一段文本，点击开始
        """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Textbox(
                        label="请输入文本：",
                        interactive=True,
                        max_lines=8,
                        autoscroll=False
                    )
                with gr.Row():
                    with gr.Column():
                        output_length = gr.Number(
                            label='返回结果数量',
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            precision=0,
                        )
                    with gr.Column():
                        ddp_length = gr.Number(
                            label='多样性召回数量(0表示不开启多样性)',
                            minimum=0,
                            maximum=2000,
                            value=1000,
                            step=100,
                            precision=0,
                        )
                with gr.Row():
                    search_btn = gr.Button(value="开始检索")
                
        with gr.Row():
            # output_result = gr.Gallery(elem_id="gallery", columns=6, object_fit="contain", interactive=True, label="检索结果", height="auto")
            output_result = gr.HTML(label="检索结果")

        
        search_btn.click(
            run_search,
            inputs=[text, ddp_length, output_length],
            outputs=output_result
        )

    # 启动应用程序 30.72.66.8:port
    app.queue(2)
    app.launch(server_name="0.0.0.0", server_port=int(port), share=True)

if __name__ == '__main__':
    main(8081)
    # test()