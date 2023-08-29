import json
import re
import os
import copy


def replace_last_match(text, pattern, replacement):
    # Find the last match
    last_match = None
    for match in re.finditer(pattern, text):
        last_match = match

    # Replace the last match
    if last_match:
        start, end = last_match.span()
        text = text[:start] + replacement + text[end:]

    return text


def map_formulat2answer(answer_text, answer_formula):
    pattern1 = r"\d+\.?\d*"
    match = re.findall(pattern1, answer_formula)
    answer_description = None

    if len(match):
        pattern2 = ""
        for index in range(0, len(match)-1 if len(match)-1 < 2 else 2):
            if match[index] == '0.01':
                continue
            else:
                pattern2 = pattern2 + match[index] + r'[^1-9]*'
        pattern2 = pattern2 + r'.*=[^0-9]*'+match[-1]
        answer_description = re.sub(pattern2, "<<>>", answer_text)
        if answer_description == answer_text:
            if len(match) >= 3:
                reverse_pattern = r''+match[1]+ r'[^1-9]*'+match[0]+ r'[^1-9]*'+ r'.*=[^0-9]*'+match[-1]
            else:
                reverse_pattern = r'x{100}'
            answer_description = re.sub(reverse_pattern, "<<>>", answer_text)
            if answer_description == answer_text:
                if len(match) == 2 and match[0] == match[1]:
                    answer_description = replace_last_match(answer_description, match[-1], "<<>>")
                else:
                    answer_description = None

    if answer_description:
        answer_description = re.sub(r"(\d/)?<<>>", answer_formula.strip("<<>>"), answer_description)
    
    return answer_description


def add_zero(match):
    if match:
        return match.group(1) + '0' + match.group(2)
    else:
        return ''


def remove_non_digits(text):
    pattern = r"[^0-9=+\-*/\.]"
    return re.sub(pattern, "", text)


def unique_letter_count(text):
    letters = set()
    for char in text:
        if char.isalpha():
            letters.add(char.lower())
    return len(letters)


fail_count = 0
def extract_formula_from_answer_text(answer_text):
    # pattern = r"((\d+((/?\d+)|((\.)?\d+)|((\%)?)))(/W+))+" # =\D*(\-?\d+(\.\d+)?)
    pattern = r'(?P<first_operand>\d+(\.\d+)?\D*)(?P<operator>\s*[\+\-\*/]\s*\D?\d+(\.\d+)?\D*)+\s*=\s*\D*(?P<result>\d+(\.\d+)?)'
    match = re.search(pattern, answer_text)
    result = ""
    if match:
        init_formula = match.group()
        unique_letter_counts = unique_letter_count(init_formula)
        if unique_letter_counts == 0:
            # formula = re.sub(r'\s', '', init_formula)
            # has_alpha = any(c.isalpha() for c in formula)
            formula = remove_non_digits(init_formula)
            result = "<<"+formula+">>"
            answer_text = answer_text.replace(init_formula, formula)
        print(init_formula)
        print(result)
        global fail_count
        if result == "":
            fail_count += 1
    # else:
    #     print(answer_text)
    return answer_text, result


def reformulate_sub_qa(sub_qa_list):
    new_qa_list = []
    for i, qa in enumerate(sub_qa_list):
        curr_qa = copy.deepcopy(qa)
        
        if not len(curr_qa["answer_formula"]) or not len(curr_qa["answer"]) :
            if curr_qa["answer"] == "":
                curr_qa["answer_formula"] = ""
            else:
                curr_qa["answer"], curr_qa["answer_formula"] = extract_formula_from_answer_text(curr_qa["answer"])
        else:
            # 将 .20 转为 0.20
            text = re.sub(r'(\D)(\.\d)', add_zero,  curr_qa["answer"])
            if text != curr_qa["answer"]:
                curr_qa["answer"] = text
            # 将 10,000 中的,去掉 或者 10 000 中的空格
            text = re.sub(r'(?<=\d{1})[,|\s](?=\d{2})', '', curr_qa["answer"])
            if text != curr_qa["answer"]:
                curr_qa["answer"] = text
            # 将 times 和 plus equals 转化为 符号
            while True:
                text = re.sub(r"(\d+)\s+times\s+(\d+)", r"\1*\2", curr_qa["answer"])
                text = re.sub(r"(\d+)\s+plus\s+(\d+)", r"\1+\2", text)
                if text == curr_qa["answer"]:
                    break
                else:
                    curr_qa["answer"] = text
            text = re.sub(r"(\d+)\s+equals\s+(\d+)", r"\1=\2", curr_qa["answer"])
            if text != curr_qa["answer"]:
                curr_qa["answer"] = text
            # 对 answer_formula 处理，在 .02 这种浮点数前面添加0
            text = re.sub(r'(\D)(\.\d)', add_zero,  curr_qa["answer_formula"])
            if text != curr_qa["answer_formula"]:
                curr_qa["answer_formula"] = text
            
            answer = map_formulat2answer(curr_qa["answer"], curr_qa["answer_formula"])
            if answer is None:
                print()
                print("answer is None")
                print(curr_qa["answer"])
                print(curr_qa["answer_formula"])
                print()
            else:
                curr_qa["answer"] = answer
            
        curr_qa["options"].pop()
        new_qa_list.append(curr_qa)

    return new_qa_list
    

def clean_dataset(file_path, save_path = None):
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