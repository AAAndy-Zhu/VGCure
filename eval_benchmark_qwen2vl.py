import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.generation import GenerationConfig
import torch
import json
from tqdm import tqdm
import re
import random
torch.manual_seed(1234)


def inference(args):
    model_path = args.model_path
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)


    query_dict = {
        'directed':{
            'qa': 'The given image shows a graph where circles represent nodes, with the content inside indicating the node names. The arrowed lines connecting two nodes represent edges, and the content in the middle of the edges indicates the edge names. Answer the given questions based on the graph in the image.\nQuestion: ',
            'fc': 'The given image shows a graph where circles represent nodes, with the content inside indicating the node names. The arrowed lines connecting two nodes represent edges, and the content in the middle of the edges indicates the edge names. Verify the truth of the given claim against the graph in the image.\nClaim: '
        },
        'undirected':{
            'qa': 'The given image shows a graph where circles represent nodes, with the content inside being the node names. The lines connecting two nodes represent edges, and the content in the middle of the edges represents the edge names. Answer the given questions based on the graph in the image.\nQuestion: ',
            'fc': 'The given image shows a graph where circles represent nodes, with the content inside being the node names. The lines connecting two nodes represent edges, and the content in the middle of the edges represents the edge names. Verify the truth of the given claim against the graph in the image.\nClaim: '
        }
    }

    directed = args.directed

    eval_file_path = args.eval_file_path
    img_path = args.img_path
    answers_file = args.output_file_path

    test_data = [json.loads(line) for line in open(eval_file_path)]
    ans_file = open(answers_file, "w")
    for data in tqdm(test_data):
        result = dict()
        task = dict()
        result = {'id': data['id'], 'graph_image': data['graph_image'], 'task': task}
        for kt, vt in data['task'].items():

            task[kt] = vt
            for k, v in vt.items():
                if not v:
                    pass
                else:
                    if 'qa' in k:
                        question = v['question']
                        qs = query_dict[directed]['qa'] + question + '\n'
                    elif 'fc' in k:
                        claim = v['claim']
                        qs = query_dict[directed]['fc'] + claim + '\n'
                    image_file = os.path.join(img_path, data['graph_image'])
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_file,
                                },
                                {"type": "text", "text": qs},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    # print(text)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    task[kt][k]['prediction'] = output_text
        ans_file.write(json.dumps(result) + '\n')
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--model_path", type=str, required=True)
    parsers.add_argument("--eval_file_path", type=str, required=True)
    parsers.add_argument("--img_path", type=str, required=True)
    parsers.add_argument("--output_file_path", type=str, required=True)
    parsers.add_argument("--directed", type=str, required=True, choices=['directed, undirected'])

    args = parsers.parse_args()
    inference(args)