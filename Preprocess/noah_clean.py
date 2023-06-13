import json
import re
import os
import copy


def extract_answer_num(answer_text):
    pattern = r"-?\d+((/?\d+)|((\.)?\d+)|(:?\d+)|((\%)?))"
    match_iter = re.finditer(pattern, answer_text)
    num_list = []
    for x in match_iter:
        num_list.append(x.group())

    if len(num_list) == 0:
        return None
    elif len(num_list) == 1:
        return num_list[0]
    else:
        print(answer_text)


def add_zero(match):
    if match:
        return match.group(1) + '0' + match.group(2)
    else:
        return ''

def reformulate_sub_qa(sub_qa_list):
    new_qa_list = []
    for i, qa in enumerate(sub_qa_list):
        curr_qa = copy.deepcopy(qa)
        extract_answer_num(curr_qa["answer"])
        curr_qa["options"].pop()
        new_qa_list.append(curr_qa)

    return new_qa_list
    

def clean_dataset(file_path, save_path):
    with open(file_path, 'rb') as f:
        sample_list = json.load(f)
    results = {}
    for i, sample in enumerate(sample_list):
        sample = sample[str(i)]
        results[str(i)] = sample
        sub_qa_list = reformulate_sub_qa(sample["sub-qa"])
        results[str(i)]["sub-qa"] = sub_qa_list
    if save_path == None : return results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    return results

def process(file_path, save_path) :
    return clean_dataset(file_path, save_path)