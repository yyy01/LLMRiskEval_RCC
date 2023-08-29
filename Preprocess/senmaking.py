import json
import jsonlines
from tqdm import tqdm

out={}
def process(file_path, save_path) :

    with open(file_path, "r+", encoding="utf8") as f1,open(save_path, 'w', encoding="utf8") as f2:
        data = list(jsonlines.Reader(f1))
        for idx,item in enumerate(tqdm(data)):
            element=dict()
            element["passage"]=f'Here are 2 statements,one of them is true and the other is false.\n Statement0: {item["sentence0"]}\n Statement1:{item["sentence1"]}\n '
            element["sub-qa"]=[{
                "question":f'Now we know Statement{item["false"]} is false,why?',
                "answer":item[item["reason"]],
                "option-number":3,
                "options":[
                    {
                        "option_type":"GroundTruth" if item["reason"]=="A" else "Wront Answer",
                        "option_describe":item["A"]
                    },
                    {
                        "option_type":"GroundTruth" if item["reason"]=="B" else "Wront Answer",
                        "option_describe":item["B"]
                    },
                    {
                        "option_type":"GroundTruth" if item["reason"]=="C" else "Wront Answer",
                        "option_describe":item["C"]
                    },
                ]
            }]
            out[str(idx)]=element
        json.dump(out,f2, indent=4)

    return out