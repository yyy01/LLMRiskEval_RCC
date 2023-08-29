import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json, os
from format import construct_sample_body, construct_sub_qa_body, construct_option, INCORRECT_OPTION_NAME, GENERAL_QUESTION
from typing import List, Dict
from tqdm import tqdm
op_num_dict = {}
IS_USING_GENERAL_QUESTION = False


def format_single_sample(context:str, answer:str, option_list:List[str], question:str) -> Dict:
    sample = construct_sample_body(context=context)
    sub_qa = construct_sub_qa_body(question, answer)
    option = construct_option("GroundTruth", answer)
    sub_qa["options"].append(option)
    for op in option_list:
        if op != answer:
            sub_qa["options"].append(construct_option(INCORRECT_OPTION_NAME, op))
    sub_qa["option-number"] = len(sub_qa["options"])
    sample["sub-qa"].append(sub_qa)
    
    op_num_dict[sub_qa["option-number"]] = 1 if sub_qa["option-number"] not in op_num_dict else op_num_dict[sub_qa["option-number"]]+1

    return sample


def main(file_path:str, save_path:str) -> None:
    with open(file_path, 'rb') as f:
        data_list = [json.loads(line) for line in f]

    data_dict = {}
    for i, item in enumerate(tqdm(data_list)):
        sample = item
        ans = ''
        for c in item['question']['choices']:
            if c['label'] == item['answerKey']:
                ans = c['text']
        if IS_USING_GENERAL_QUESTION:
            sample = format_single_sample(context = item['question']['stem'], question=GENERAL_QUESTION,
                                        answer = ans, 
                                        option_list = [c['text'] for c in item['question']['choices']])
        else:
            sample = format_single_sample(context = "", 
                                          question=item['question']['stem'],
                                        answer = ans, 
                                        option_list = [c['text'] for c in item['question']['choices']])
        data_dict[str(i)] = sample
    if save_path == None : return data_dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

def process(file_path, save_path) :
    global IS_USING_GENERAL_QUESTION
    IS_USING_GENERAL_QUESTION = False
    return main(file_path, save_path)

    