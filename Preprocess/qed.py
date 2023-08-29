import json
import jsonlines
import random
from DA import eda
from tqdm import tqdm
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

out={}
def process(file_path, save_path) :

    with open(file_path, "r+", encoding="utf8") as f1,open(save_path, 'w', encoding="utf8") as f2:
        for idx,item in enumerate(jsonlines.Reader(f1)):
            # print(item['question_text'])
            element=dict()
            element["passage"]=None
            element["sub-qa"]=[{
                "question":item["question_text"]+"?",
                "answer":item["original_nq_answers"][0][0]["string"],
                "option-number":4,
                "options":[
                    {
                        "option_type":"GroundTruth",
                        "option_describe":item["original_nq_answers"][0][0]["string"]
                    },
                    {
                        "option_type":"Irrelevant Chain",
                        "option_describe":""
                    },
                    {
                        "option_type":"Irrelevant Chain",
                        "option_describe":""
                    },
                    {
                        "option_type":"Text Augmentation",
                        "option_describe":""
                    }
                ]
        }]

            # print(item['original_nq_answerzs'][0][0]['string'])
            out[str(idx)]=element

        # random select 'Answer of another' from another question's ground_truth
        for key,item in tqdm(out.items()):
            for i in range(1,3):
                out[key]["sub-qa"][0]["options"][i]["option_describe"]=out[str(random.randint(0,len(out)-1))]["sub-qa"][0]["options"][0]["option_describe"]
                while out[key]["sub-qa"][0]["options"][i]["option_describe"]==out[key]["sub-qa"][0]["options"][0]["option_describe"]:
                    out[key]["sub-qa"][0]["options"][i]["option_describe"]=out[str(random.randint(0,len(out)-1))]["sub-qa"][0]["options"][0]["option_describe"]
            
            # print(out[key]["sub-qa"][0]["options"][0]["option_describe"])
            out[key]["sub-qa"][0]["options"][3]["option_describe"]=eda.eda(out[key]["sub-qa"][0]["options"][0]["option_describe"],alpha_sr=0, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1)[0]

        json.dump(out,f2, indent=4)

    return out