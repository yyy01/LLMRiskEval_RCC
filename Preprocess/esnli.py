import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json, os
from format import construct_sample_body, construct_sub_qa_body, construct_option, INCORRECT_OPTION_NAME
from typing import List, Dict
from tqdm import tqdm

def format_single_sample(question:str, answer:str, option_list:List[str], context:str="") -> Dict:
    sample = construct_sample_body(question, answer, context)
    sub_qa = construct_sub_qa_body(question, answer)
    option = construct_option("GroundTruth", answer)
    sub_qa["options"].append(option)
    for op in option_list:
        if op != answer:
            sub_qa["options"].append(construct_option(INCORRECT_OPTION_NAME, op))
    if len(sub_qa["options"]) != 3:
        print("error")
    sub_qa["option-number"] = len(sub_qa["options"])
    sample["sub-qa"].append(sub_qa)

    return sample


OPTIONS = ['neutral', 'entailment', 'contradiction']
QUESTION = "What is the logical relationship between premise and hypothesis?"


def main(file_path:str, save_path:str) -> None:
    data_df = pd.read_csv(file_path)
    data_dict = {}
    for i in tqdm(range(len(data_df))):
        sample = data_df.loc[i, ['Sentence1', 'Sentence2', 'gold_label']].to_dict()
        sample = format_single_sample(QUESTION, sample['gold_label'], 
                                      option_list=OPTIONS, 
                                      context="Premise: {} \n Hypothesis: {}".format(sample['Sentence1'], sample['Sentence2']))
        data_dict[str(i)] = sample
    if save_path == None : return data_dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

def process(file_path, save_path) :
    return main(file_path, save_path)