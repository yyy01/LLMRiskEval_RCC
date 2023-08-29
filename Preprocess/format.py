from copy import deepcopy

SAMPLE_BODY_FORMAT = {
        "passage": "",
        # "answer": "",
        "sub-qa":[
        ]
    }

SUB_QA_BODY_FORMAT = {
                    "question": "",
                    "answer": "",
                    "answer_formula":"",
                    "option-number": 0,
                    "options":[]
                }

OPTION_FORMAT = {
                    "option_type": "",
                    "option_describe": ""
                }

INCORRECT_OPTION_NAME = "Answewr Error"

GENERAL_QUESTION = "Choose one of several candidate options that you think is correct."

def construct_sample_body(question:str="", answer:str="", context:str=""):
    sample = deepcopy(SAMPLE_BODY_FORMAT)
    # sample["question"] = question
    # sample["answer"] = answer
    sample["passage"] = context
    return sample


def construct_sub_qa_body(question:str, answer:str):
    sub_qa = deepcopy(SUB_QA_BODY_FORMAT)
    sub_qa["question"] = question
    sub_qa["answer"] = answer
    return sub_qa


def construct_option(option_type:str, option_describe:str):
    option = deepcopy(OPTION_FORMAT)
    option["option_type"] = option_type
    option["option_describe"] = option_describe
    return option