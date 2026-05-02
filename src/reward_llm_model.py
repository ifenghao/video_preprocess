import torch
import json
import re
import json_repair
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import random
import os

def free_gpus():
    import subprocess
    return subprocess.call('ps -ef|grep "run.py"|grep -v grep|cut -c 9-16|xargs kill -9 2>/dev/null', shell=True)

def get_part_lines(total, part_index, part_total):
    part_len = (total - 1) // part_total + 1
    start, end = part_index * part_len, (part_index + 1) * part_len
    end = min(total, end)
    return start, end

class Qwen3:
    def __init__(self, model_name):
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto"
        # )
        # self.model.eval()
        self.model = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            max_num_seqs=32,
            max_model_len=32768,
            tensor_parallel_size=1,
        )

    def infer(self, prompt, enable_thinking=False):
        # prepare the model input
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        ### transformers
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # conduct text completion
        # with torch.no_grad():
        #     generated_ids = self.model.generate(
        #         **model_inputs,
        #         max_new_tokens=32768
        #     )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # try:
        #     # rindex finding 151668 (</think>)
        #     index = len(output_ids) - output_ids[::-1].index(151668)
        # except ValueError:
        #     index = 0

        # # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        # content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        ### vllm
        sampling_params = SamplingParams(temperature=0.1, top_p=0.95, top_k=20, max_tokens=32768)
        outputs = self.model.generate([text], sampling_params)
        content = outputs[0].outputs[0].text
        return content

    def batch_infer(self, prompts, enable_thinking=False):
        # prepare the model input
        input_text_list = []
        for prompt in prompts:
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            input_text_list.append(text)
        sampling_params = SamplingParams(temperature=0.2, top_p=0.95, top_k=20, max_tokens=32768)
        outputs = self.model.generate(input_text_list, sampling_params)
        contents = [output.outputs[0].text for output in outputs]
        return contents

def qwen3vl(model_path):
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    # default: Load the model on the available device(s)
    # model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    #     model_path, dtype="auto", device_map="auto"
    # )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "./test.mp4",
                },
                {"type": "text", "text": "Describe the video in details./think"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    model_inputs = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        model_inputs[k] = v.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

prompt_extract_en = '''You are a video caption information extract expert, given a input video caption, extract the detailed information in JSON format with "characters", "events", "scene" and "camera_movement" as keys.
Requirements:
- If the input video caption contains one or more characters, you should extract each character's detailed information in JSON list. Detailed character information must contain "subject_name", which consists of brief location description and subject types, and can uniquely refer to a character. The pronouns in the position description should be replaced by the nouns they refer to. DO NOT use numbers. Then candidate keys are static features, such as "hair", "accessory", "eyebrows", "eyes", "mouth", "beard", "face", "clothing", etc. Not mentioned candidate key should not be output. All detailed information should be completely extracted without omission.
- If the input video caption contains one or more events, you should extract each event in list. An event must include an action, motion or movement (NOT STATIC INFOMATION). DO NOT repeat same events. Every event is represented by a brief sentence with in 10 words, with a subject, a predicate and optionally an object, avoid detailed information. Every event must be atomic, meaning that it cannot be further split into multiple events. Do not output camera related description. Substitute pronouns by the nouns they refer to.
- If the input video caption contains scene description, output them in JSON format. If not, do not output.
- If the input video caption contains camera movement, output them in JSON format. If not, do not output.
- If the input video caption contains special effects, such as emotional effects (sweat drops, anger veins, tears, nosebleeds, floating symbols, etc.), environmental effects (rain, snow, shooting stars, smoke, particles, falling leaves, etc.), action effects (speed lines, light beams, energy balls, electric currents, flames, explosions, magic, etc.), and background effects (bubble backgrounds, radial lines, etc.), output them in JSON format. If not, do not output.
Example:
Input: A red hair girl, wearing a green dress, is walking in a park filled with flowers. An old man, with white beard, is looking at her. Several petals are floating in the air. The camera remains stationary.
Output: {"characters": [{"subject_name": "red hair girl", "hair": "red", "clothing": "green dress"}, {"subject_name": "old man", "beard": "white"}], "events": ["A girl is walking", "An old man is looking at the girl"], "scene": "A park filled with flowers", "camera_movement": "stationary", "special_effect": "Several petals are floating in the air"}
'''
prompt_extract_cn = '''你是一位视频字幕信息提取专家，给定一段输入的视频字幕，请以JSON格式提取详细信息，并使用"characters"、"events"、"scene"和"camera_movement"作为键。
Requirements:
- 如果输入的视频字幕包含一个或多个角色，你应该在JSON列表中提取每个角色的详细信息。详细的角色信息必须包含"主体名称"，它由方位描述和主体类型构成，可以唯一地指代一个角色。方位描述中的代词用指代的名词替换。不要使用数字编号。候选键都是静态特征，例如：头发、发饰、眉毛、眼睛、耳朵、鼻子、嘴巴、胡须、脸、服装等。输出中不能包含未提及的候选键。所有详细信息都应被完整提取，不得遗漏。
- 如果输入的视频字幕包含一个或多个事件，你应该在列表中提取每个事件。一个事件必须包含一个动作、动态或移动（不是静态信息）。不要重复相同的事件。每个事件都用一个10个词以内的简短句子表示，包含主语、谓语和可选的宾语，避免详细信息。每个事件必须是原子的，意味着它不能被进一步拆分成多个事件。输出的事件中不能包含镜头相关描述。用代词所指代的名词替换代词。
- 如果输入的视频字幕包含场景描述，请以JSON格式输出。如果没有，则不输出。
- 如果输入的视频字幕包含镜头移动，请以JSON格式输出。如果没有，则不输出。
- 如果输入的视频字幕包含特效描述，例如情绪特效（汗滴、怒筋、眼泪、鼻血、悬浮符号等），环境特效（雨、雪、流星、烟雾、粒子、落叶等），动作特效（速度线、光束、能量球、电流、火焰、爆炸、魔法等）背景特效（泡泡背景、放射线条等），请以JSON格式输出。如果没有，则不输出。
Example:
Input: 一个穿着绿色裙子的红发女孩正在一个开满鲜花的公园里散步。一位有白胡子的老人正在看着她。空中漂浮着几朵花瓣。镜头保持静止。
Output: {"characters": [{"主体名称": "红发女孩", "头发": "红色", "服装": "绿色裙子"}, {"主体名称": "老人", "胡子": "白色"}], "events": ["一个红发女孩在散步", "一个老人正在看着红发女孩"], "scene": "一个开满鲜花的公园", "camera_movement": "静止", "special_effect": "空中漂浮着几朵花瓣"}
'''
prompt_extract_end = '## Input:\n{text}\n## Output:\n'

# v1
prompt_judge_en = '''You are a video caption pair comparison expert, given a JSON format detailed description and a input video caption, compare the description of every JSON field with the input video caption and classify whether they are consistent, contradictory or absent. Output must contain the same JSON format as the detailed description, with the field value filled by "consistent" or "absent" or "contradictory".
Requirements for the JSON input:
- The "characters" key contains all characters information in list format, each item describes whether the character exists and the localized feature of the character. If one item is entailed or there are similar expressions with it in the input video caption, output "consistent", and if the item are not mentioned in the input video caption, output "absent". Otherwise output "contradictory". The ouput list length is the same with the input.
- The "events" key contains all events including an action, motion or movement in list format. If one event is entailed or there are similar expressions with it in the input video caption, output "consistent", and if the event is not mentioned in the input video caption, output "absent". Otherwise output "contradictory". The ouput list length is the same with the input.
- The "scene" key contains the scene description. If scene is not mentioned in the input video caption, output "absent". If the scene descriptions exist and are consistent with the input video caption, output "consistent", otherwise output "contradictory".
- The "camera_movement" key contains the camera movement. If camera movement is not mentioned in the input video caption, output "absent".  If the camera movement exists and is consistent with the input video caption, output "consistent", otherwise output "contradictory".
Example:
JSON input: {"characters": ["There is the red hair girl.", "The eye of red hair girl is blue.", "The clothing of red hair girl is green dress.", "There is the old man.", "The beard of old man is white.", "There is the boy.", "The clothing of boy is blue shirt."], "events": ["A girl is walking", "An old man is sitting on the bench", "A boy is running"], "scene": "A park filled with flowers", "camera_movement": "stationary"}
Video caption input: A red hair girl, wearing a green dress, is walking in a park filled with flowers. An old man, with white beard, is looking at her. The camera remains stationary.
Output: {"characters": ["consistent", "contradictory", "consistent", "consistent", "consistent", "absent", "absent"], "events": ["consistent", "contradictory", "absent"], "scene": "consistent", "camera_movement": "consistent"}
'''
prompt_judge_cn = '''你是一位视频字幕对比专家，在给定一个JSON格式的详细描述和一个输入的视频字幕后，你需要比较每个JSON字段的描述与输入的视频字幕，并判断它们是一致、缺失还是矛盾。输出的分类字段值应填充为“consistent”、“absent”或“contradictory”。
Requirements for the JSON input:
- “characters”键以列表格式包含所有角色信息，其中每一项描述了角色主体是否存在以及该角色主体的局部特征。如果某一项在输入的视频字幕中被包含或有相似表述，则输出“consistent”；如果未在视频字幕中提及，则输出“absent”；否则输出“contradictory”。输出列表长度和输入一样。
- “events”键以列表格式包含所有包含动作、动态或移动的事件。如果某个事件在输入的视频字幕中被蕴含或有相似表述，则输出“consistent”；如果未在视频字幕中提及，则输出“absent”；否则输出“contradictory”。输出列表长度和输入一样。
- “scene”键包含场景描述。如果场景在输入的视频字幕中未被提及，则输出“absent”；如果场景描述存在且与视频字幕一致，则输出“consistent”；否则输出“contradictory”。
- “camera_movement”键包含镜头移动。如果镜头移动在输入的视频字幕中未被提及，则输出“absent”；如果镜头移动存在且与视频字幕一致，则输出“consistent”；否则输出“contradictory”。
Example:
JSON input: {"characters": ["画面存在红发女孩。", "红发女孩的眼睛是蓝色的。", "红发女孩的衣服是绿色的裙子。", "画面存在一个老人。", "老人的胡子是白色的。", "画面存在一个男孩。", "男孩的衣服是蓝色的衬衫。"], "events": ["一个女孩在走路", "一个老人坐在长椅上", "一个男孩在跑步"], "scene": "一个开满鲜花的公园", "camera_movement": "静止"}
Video caption input: 一个穿着绿色裙子的红发女孩正在一个开满鲜花的公园里散步。一位有白胡子的老人正在看着她。镜头保持静止。
Output: {"characters": ["consistent", "contradictory", "consistent", "consistent", "consistent", "absent", "absent"], "events": ["consistent", "contradictory", "absent"], "scene": "consistent", "camera_movement": "consistent"}
'''
prompt_judge_end = '## JSON input:\n{input_json}\n## Video caption input:\n{input_text}\n## Output:\n'

# v2
prompt_judge_event_en = '''You are a video caption pair comparison expert, given a input video caption and a list of events, compare each event with the input video caption and classify into three classes: consistent, contradictory or absent.
Definitions:
- "consistent" means that the event is entailed or there are similar expressions with it in the input video caption. Emotional descriptions can be ignored.
- "absent" means that the event is not mentioned in the input video caption.
- "contradictory" means that some detail of the event contradicts with the input video caption. 
Output a list in JSON format:
[{"event": "copy an event here", "class": "put class name here", "reason": "give a reason here"}, ... ]
Tips: The "event" field must copied from input without any modification. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON.
'''
prompt_judge_event_cn = '''你是一位视频字幕对比较专家，给定一个输入视频字幕和一个事件列表，请将每个事件与输入视频字幕进行比较，并将其分为三类："consistent", "absent", "contradictory"
定义：
- "consistent"意味着事件被包含，或者在输入视频字幕中有与之相似的表达。可以忽略情绪描述。
- "absent"意味着在输入视频字幕中没有提到该事件。
- "contradictory"意味着该事件的某些细节与输入视频字幕相矛盾。
输出列表的JSON格式：
[{"event": "在此处复制事件", "class": "在此处填写类别名称", "reason": "在此处给出原因"}, ... ]
注意：“event”字段必须从输入中复制而不做任何修改。不要提供任何其他输出文本或解释。只输出JSON。
'''
prompt_judge_list_end = '## Video caption input:\n{input_text}\n## List input:\n{input_json}\n## Output:\n'

# v3
prompt_judge_split_event_en = '''You are a video caption pair comparison expert, given a input video caption and an event, compare the event with the input video caption and classify into three classes: consistent, contradictory or absent.
Definitions:
- "consistent" means that the event is entailed or there are similar expressions with it in the input video caption. If the modifiers are different but the keywords are the same, "consistent" can also be output. If only the emotional descriptions are different, "consistent" can be output. If only the position descriptions of the subject are different, "consistent" can be output.
- "absent" means that the event is not mentioned in the input video caption. Do not speculate on events not explicitly mentioned in the video caption, just output "absent" directly. If the subject in the event is mentioned, it is not "absent".
- "contradictory" means that some detail of the event contradicts with the input video caption. If the subject in the event is not mentioned, it should be "absent" instead of "contradictory".
Output JSON format:
{"class": "put class name here", "reason": "give a reason here"}
Tips: DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON.
'''
prompt_judge_split_event_cn = '''你是一位视频字幕对比较专家，给定一个输入视频字幕和一个事件，请将该事件与输入视频字幕进行比较，并将其分为三类："consistent", "absent", "contradictory"
定义：
- "consistent"意味着事件被包含，或者在输入视频字幕中有与之相似的表达。如果修饰词不同而关键词相同也可以输出"consistent"。类似的颜色可以输出"consistent"。只有情绪描述不同也可以输出"consistent"。只有主体的位置描述不同也可以输出"consistent"。
- "absent"意味着在输入视频字幕中没有提到该事件。视频字幕中没有明确提到的事件不要推测，直接输出"absent"。
- "contradictory"意味着该事件的某些细节与输入视频字幕相矛盾。如果事件中的主体没有被提及，就不是"contradictory"而应该是"absent"。
输出列表的JSON格式：
{"class": "在此处填写类别名称", "reason": "在此处给出原因"}
注意：不要提供任何其他输出文本或解释。只输出JSON。
'''
prompt_judge_list_end = '## Video caption input:\n{input_text}\n## Event input:\n{input_json}\n## Output:\n'

def clean_output(input_text):
    match = re.search(r'```json(?P<content>[\S\s]+)```', input_text)
    text = match.group('content') if match else input_text
    try:
        return json_repair.loads(text)
    except:
        return {}

def convert_json_en(input_json):
    res = {'characters': []}
    for item in input_json.get('characters', []):
        if 'subject_name' not in item: continue
        subject_name = item['subject_name']
        for key, value in item.items():
            if key == 'subject_name':
                text = f'There is the {subject_name}.'
            else:
                text = f'The {key} of {subject_name} is {value}.'
            res['characters'].append(text)
    if 'events' in input_json:
        res['events'] = input_json['events']
    if 'scene' in input_json:
        res['scene'] = f'Scene description: {input_json["scene"]}'
    if 'camera_movement' in input_json:
        res['camera_movement'] = f'Camera movement description: {input_json["camera_movement"]}'
    if 'special_effect' in input_json:
        if isinstance(input_json['special_effect'], list):
            res['special_effect'] = f'Special effect description: {", ".join(input_json["special_effect"])}'
        else:
            res['special_effect'] = f'Special effect description: {input_json["special_effect"]}'
    return res

def convert_json_cn(input_json):
    res = {'characters': []}
    for item in input_json.get('characters', []):
        if '主体名称' not in item: continue
        subject_name = item['主体名称']
        for key, value in item.items():
            if key == '主体名称':
                text = f'画面存在{subject_name}.'
            else:
                text = f'{subject_name}的{key}是{value}.'
            res['characters'].append(text)
    if 'events' in input_json:
        res['events'] = input_json['events']
    if 'scene' in input_json:
        res['scene'] = f'场景描述: {input_json["scene"]}'
    if 'camera_movement' in input_json:
        res['camera_movement'] = f'镜头运动描述: {input_json["camera_movement"]}'
    if 'special_effect' in input_json:
        if isinstance(input_json['special_effect'], list):
            res['special_effect'] = f'特效描述: {"，".join(input_json["special_effect"])}'
        else:
            res['special_effect'] = f'特效描述: {input_json["special_effect"]}'
    return res

def get_recall_score(res_json):  # gt -> json, pred -> text
    res = {}
    if 'characters' in res_json:
        characters_res_list = res_json['characters']
        pos = 0
        for item in characters_res_list:
            if isinstance(item, dict):
                item = item.get('class')
            if item == 'consistent':
                pos += 1
        res['characters'] = 0. if len(characters_res_list) == 0 else pos / len(characters_res_list)
    if 'events' in res_json:
        events_res_list = res_json['events']
        pos = 0
        for item in events_res_list:
            if isinstance(item, dict):
                item = item.get('class')
            if item == 'consistent':
                pos += 1
        res['events'] = 0. if len(events_res_list) == 0 else pos / len(events_res_list)
    if 'scene' in res_json:
        if isinstance(res_json['scene'], dict):
            scene_res = res_json['scene'].get('class')
        else:
            scene_res = res_json['scene']
        res['scene'] = 1. if scene_res == 'consistent' else 0.
    if 'camera_movement' in res_json:
        if isinstance(res_json['camera_movement'], dict):
            camera_movement_res = res_json['camera_movement'].get('class')
        else:
            camera_movement_res = res_json['camera_movement']
        res['camera_movement'] = 1. if camera_movement_res == 'consistent' else 0.
    if 'special_effect' in res_json:
        if isinstance(res_json['special_effect'], dict):
            special_effect_res = res_json['special_effect'].get('class')
        else:
            special_effect_res = res_json['special_effect']
        res['special_effect'] = 1. if special_effect_res == 'consistent' else 0.
    return res

def get_precision_score(res_json):  # pred -> json, gt -> text
    res = {}
    if 'characters' in res_json:
        characters_res_list = res_json['characters']
        pos = 0
        total = 0
        for item in characters_res_list:
            if isinstance(item, dict):
                item = item.get('class')
            if item == 'consistent':
                pos += 1
            if item != 'absent':
                total += 1
        res['characters'] = 0. if total == 0 else pos / total
    if 'events' in res_json:
        events_res_list = res_json['events']
        pos = 0
        total = 0
        for item in events_res_list:
            if isinstance(item, dict):
                item = item.get('class')
            if item == 'consistent':
                pos += 1
            if item != 'absent':
                total += 1
        res['events'] = 0. if total == 0 else pos / total
    if 'scene' in res_json:
        if isinstance(res_json['scene'], dict):
            scene_res = res_json['scene'].get('class')
        else:
            scene_res = res_json['scene']
        res['scene'] = 1. if scene_res == 'consistent' else 0.
    if 'camera_movement' in res_json:
        if isinstance(res_json['camera_movement'], dict):
            camera_movement_res = res_json['camera_movement'].get('class')
        else:
            camera_movement_res = res_json['camera_movement']
        res['camera_movement'] = 1. if camera_movement_res == 'consistent' else 0.
    if 'special_effect' in res_json:
        if isinstance(res_json['special_effect'], dict):
            special_effect_res = res_json['special_effect'].get('class')
        else:
            special_effect_res = res_json['special_effect']
        res['special_effect'] = 1. if special_effect_res == 'consistent' else 0.
    return res

def get_f1_score(precision, recall):
    res = {}
    for key in ['characters', 'events', 'scene', 'camera_movement', 'special_effect']:
        if key in precision and key in recall:
            res[key] = 2 * precision[key] * recall[key] / (precision[key] + recall[key]) if precision[key] + recall[key] != 0 else 0.
    return res

def judge_pipeline(model, input_json, input_caption, prompt_judge_event):
    res = {}
    if 'characters' in input_json:
        characters_res = model.infer(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps(input_json['characters'], ensure_ascii=False), input_text=input_caption))
        res['characters'] = clean_output(characters_res)
    if 'events' in input_json:
        events_res = model.infer(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps(input_json['events'], ensure_ascii=False), input_text=input_caption))
        res['events'] = clean_output(events_res)
    if 'scene' in input_json:
        scene_res = model.infer(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps([input_json['scene']], ensure_ascii=False), input_text=input_caption))
        scene_res = clean_output(scene_res)
        if len(scene_res) > 0:
            res['scene'] = scene_res[0]
    if 'camera_movement' in input_json:
        camera_movement_res = model.infer(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps([input_json['camera_movement']], ensure_ascii=False), input_text=input_caption))
        camera_movement_res = clean_output(camera_movement_res)
        if len(camera_movement_res) > 0:
            res['camera_movement'] = camera_movement_res[0]
    if 'special_effect' in input_json:
        special_effect_res = model.infer(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps([input_json['special_effect']], ensure_ascii=False), input_text=input_caption))
        special_effect_res = clean_output(special_effect_res)
        if len(special_effect_res) > 0:
            res['special_effect'] = special_effect_res[0]
    return res

def batch_judge_pipeline(model, input_json, input_caption, prompt_judge_event):
    prompts = []
    key_list = []
    if 'characters' in input_json:
        key_list.append('characters')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps(input_json['characters'], ensure_ascii=False), input_text=input_caption))
    if 'events' in input_json:
        key_list.append('events')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps(input_json['events'], ensure_ascii=False), input_text=input_caption))
    if 'scene' in input_json:
        key_list.append('scene')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps([input_json['scene']], ensure_ascii=False), input_text=input_caption))
    if 'camera_movement' in input_json:
        key_list.append('camera_movement')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps([input_json['camera_movement']], ensure_ascii=False), input_text=input_caption))
    if 'special_effect' in input_json:
        key_list.append('special_effect')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=json.dumps([input_json['special_effect']], ensure_ascii=False), input_text=input_caption))
    batch_output = model.batch_infer(prompts)
    res = {}
    for key, output in zip(key_list, batch_output):
        output = clean_output(output)
        if key == 'scene' or key == 'camera_movement' or key == 'special_effect':
            if len(output) > 0:
                res[key] = output[0]
        else:
            res[key] = output
    return res

def batch_judge_split_pipeline(model, input_json, input_caption, prompt_judge_event):
    prompts = []
    key_list = []
    text_list = []
    if 'characters' in input_json:
        for text in input_json['characters']:
            text_list.append(text)
            key_list.append('characters')
            prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=text, input_text=input_caption))
    if 'events' in input_json:
        for text in input_json['events']:
            text_list.append(text)
            key_list.append('events')
            prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=text, input_text=input_caption))
    if 'scene' in input_json:
        text_list.append(input_json['scene'])
        key_list.append('scene')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=input_json['scene'], input_text=input_caption))
    if 'camera_movement' in input_json:
        text_list.append(input_json['camera_movement'])
        key_list.append('camera_movement')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=input_json['camera_movement'], input_text=input_caption))
    if 'special_effect' in input_json:
        text_list.append(input_json['special_effect'])
        key_list.append('special_effect')
        prompts.append(prompt_judge_event + prompt_judge_list_end.format(input_json=input_json['special_effect'], input_text=input_caption))
    batch_output = model.batch_infer(prompts)
    res = {}
    for key, text, output in zip(key_list, text_list, batch_output):
        output = clean_output(output)
        if not isinstance(output, dict): continue
        output['event'] = text
        if key == 'scene' or key == 'camera_movement' or key == 'special_effect':
            res[key] = output
        else:
            if key not in res:
                res[key] = []
            res[key].append(output)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="输入文件 json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="如果需要保存的路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='/apdcephfs_toc_gy/share_302617628/francofhzhu/models/Qwen3-32B',
    )
    parser.add_argument(
        "--gt_caption_key",
        type=str,
    )
    parser.add_argument(
        "--pred_caption_key",
        type=str,
    )
    parser.add_argument(
        "--is_cn",
        action='store_true',
        help="是否是中文",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
    )
    parser.add_argument(
        "--top_samples",
        type=int,
        default=100,
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

    data_path = args.input_file
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(len(data))
    data = data[:args.top_samples]
    random.shuffle(data)
    start, end = get_part_lines(len(data), args.part_index, args.part_total)

    os.makedirs(args.save_path, exist_ok=True)
    file_list = os.listdir(args.save_path)
    res = {}
    # for file in file_list:
    #     with open(os.path.join(args.save_path, file), 'r') as f:
    #         part = json.load(f)
    #     for item in part:
    #         res[item['source_id']] = item
    print('load exist', len(res))
    prompt_extract = prompt_extract_cn if args.is_cn else prompt_extract_en
    prompt_judge = prompt_judge_cn if args.is_cn else prompt_judge_en
    prompt_judge_event = prompt_judge_event_cn if args.is_cn else prompt_judge_event_en
    prompt_judge_split_event = prompt_judge_split_event_cn if args.is_cn else prompt_judge_split_event_en
    # model = Qwen3("/apdcephfs_toc_gy/share_302617628/francofhzhu/models/Qwen3-235B-A22B-Instruct-2507-FP8")
    model = Qwen3(args.model_path)
    free_gpus()
    save_list = []
    for item in tqdm(data[start:end]):
        if item['source_id'] in res:
            save_list.append(res[item['source_id']])
            continue
        gt_caption, pred_caption = item[args.gt_caption_key], item[args.pred_caption_key]
        # recall
        format_text = model.infer(prompt_extract + prompt_extract_end.format(text=gt_caption))
        format_text = clean_output(format_text)
        format_text = convert_json_cn(format_text) if args.is_cn else convert_json_en(format_text)
        judge_text = batch_judge_split_pipeline(model, input_json=format_text, input_caption=pred_caption, prompt_judge_event=prompt_judge_split_event)
        if args.verbose:
            if 'characters' in judge_text:
                print('-' * 20, 'recall characters', '-' * 20)
                for judge in judge_text['characters']:
                    print(judge.get('class'), judge)
            if 'events' in judge_text:
                print('-' * 20, 'recall events', '-' * 20)
                for judge in judge_text['events']:
                    print(judge.get('class'), judge)
        item['format_recall'] = format_text
        item['judge_recall'] = judge_text
        item['recall'] = get_recall_score(judge_text)
        print('recall', item['recall'])
        # precision
        format_text = model.infer(prompt_extract + prompt_extract_end.format(text=pred_caption))
        format_text = clean_output(format_text)
        format_text = convert_json_cn(format_text) if args.is_cn else convert_json_en(format_text)
        judge_text = batch_judge_split_pipeline(model, input_json=format_text, input_caption=gt_caption, prompt_judge_event=prompt_judge_split_event)
        if args.verbose:
            if 'characters' in judge_text:
                print('-' * 20, 'precision characters', '-' * 20)
                for judge in judge_text['characters']:
                    print(judge.get('class'), judge)
            if 'events' in judge_text:
                print('-' * 20, 'precision events', '-' * 20)
                for judge in judge_text['events']:
                    print(judge.get('class'), judge)
        item['format_precision'] = format_text
        item['judge_precision'] = judge_text
        item['precision'] = get_precision_score(judge_text)
        print('precision', item['precision'])
        item['f1'] = get_f1_score(item['precision'], item['recall'])
        print('f1', item['f1'])
        save_list.append(item)

    save_file = os.path.join(args.save_path, f'{args.part_index}-{args.part_total}.json')
    with open(save_file, 'w') as f:
        json.dump(save_list, f, indent=4, ensure_ascii=False)

    print('finish compare', args.gt_caption_key, args.pred_caption_key, ', write result to', save_file)
    if args.part_total == 1:
        get_avg_score(save_list, 'f1')
        get_avg_score(save_list, 'recall')
        get_avg_score(save_list, 'precision')

def get_avg_score(data, score_key):
    res = {}
    for item in data:
        for key, value in item[score_key].items():
            if key not in res:
                res[key] = []
            res[key].append(value)
    for key, value in res.items():
        res[key] = sum(value) / len(value)
    print(score_key, res)

if __name__ == '__main__':
    main()