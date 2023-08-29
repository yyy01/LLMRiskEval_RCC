import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from copy import deepcopy
import json, os
from format import construct_sample_body, construct_sub_qa_body, construct_option, INCORRECT_OPTION_NAME, GENERAL_QUESTION
from typing import List, Dict
from tqdm import tqdm
IS_USING_GENERAL_QUESTION = False

manual_incorrect_option = {'The person was in the drawing room, but wanted to take his draft to be completed, where did he go?':'store',
                           'Southern is the opposite of what?': 'north pole',
                           'John needed a straight wire.  Unfortunately, this one had endured some abuse and had become what?': 'loose',
                           'They would only allow one more in, he was unfortunately a what?': 'no way',
                           "Greg couldn't pay the cash advance fee, and thus was dinged for what?": 'flee',
                           'What grows from all mammal skin?': 'tooth',
                           'Sam was planning to return, but his sister was still angry that he was doing what?': 'prank',
                           'What covers the largest percentage of the pacific northwest?': 'water',
                           "Someone who doesn't believe in the divine could be called what?": 'priest',
                           'Where is one likely to find poker chips?': 'home'
                           }


def format_single_sample(context:str, answer:str, option_list:List[str], question:str) -> Dict:
    sample = construct_sample_body(context=context)
    sub_qa = construct_sub_qa_body(question, answer)
    option = construct_option("GroundTruth", answer)
    sub_qa["options"].append(option)
    for op in option_list:
        if op != answer:
            sub_qa["options"].append(construct_option(INCORRECT_OPTION_NAME, op))

    if len(sub_qa["options"]) != 5:
        if question == GENERAL_QUESTION:
            sub_qa["options"].append(construct_option(INCORRECT_OPTION_NAME, manual_incorrect_option[context]))
        else:
            sub_qa["options"].append(construct_option(INCORRECT_OPTION_NAME, manual_incorrect_option[question]))

    if len(sub_qa["options"]) != 5:
        print("error")

    sub_qa["option-number"] = len(sub_qa["options"])
    sample["sub-qa"].append(sub_qa)

    return sample


def main(file_path:str, save_path:str) -> None:
    data_df = pd.read_csv(file_path)
    data_dict = {}
    for i in tqdm(range(len(data_df))):
        sample = data_df.loc[i, ['q_text', 'q_ans', 'q_op1', 'q_op2', 'q_op3', 'q_op4', 'q_op5']].to_dict()
        if IS_USING_GENERAL_QUESTION:
            sample = format_single_sample(context = sample['q_text'], question=GENERAL_QUESTION, 
                                      answer = sample['q_ans'], option_list=[sample['q_op'+str(i)] for i in range(1, 6)])
        else:
            sample = format_single_sample(context = "", question = sample['q_text'], 
                                      answer = sample['q_ans'], option_list=[sample['q_op'+str(i)] for i in range(1, 6)])
        data_dict[str(i)] = sample
    if save_path == None : return data_dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

def process(file_path, save_path) :
    global IS_USING_GENERAL_QUESTION
    IS_USING_GENERAL_QUESTION = False
    return main(file_path, save_path)
