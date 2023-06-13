import re
import tqdm

def judge(response, _dict) :
    id = 0
    result = []

    for a in response[1:] :
        match = re.search(r'(\([A-E]\))', a)
        match1, match2 = re.search(r'([A-E]\))', a), re.search(r'(\s[A-E]\s)', a)
        match = match if match != None else match1 if match1 != None else match2
        item = None
        if match != None : item = ord(match.group(0)[1])-65 if ord(match.group(0)[1]) >= ord('A') and ord(match.group(0)[1]) <= ord('Z') else ord(match.group(0)[0])-65
        else :
            for opt in _dict['sub-qa'][id]['input'] :
                ITEM = str(_dict['sub-qa'][id]['options'][opt]['option_describe'])
                ITEM = ITEM[0].lower()+ITEM[1:] if len(ITEM) <= 1 else ITEM[0].lower()
                ITEM1 = ITEM[0].upper()+ITEM[1:] if len(ITEM) <= 1 else ITEM[0].upper()
                # print(ITEM, ITEM1)
                if a.find(ITEM) != -1 or a.find(ITEM1) != -1 : item = opt; break
        # print(a)
        try :
            _dict['sub-qa'][id]['input'][item]
        except:
            _dict.append('No options'); id += 1; continue
        type = _dict['sub-qa'][id]['options'][_dict['sub-qa'][id]['input'][item]]['option_type']
        result.append(type)
        id += 1
    
    while (id != len(_dict['sub-qa'])) : 
        _dict.append('No options'); id += 1
    
    return result

def get_path(para, data) :
    print("*********************STRAT GETTING ANSWER PATH*********************")
    for i in tqdm(para) :
        para[i]['answer'] = judge(para[i]['response'], data[i])

        for q in para[i]['new_passage'] :
            q['answer'] = judge(q['response'], data[i])
    print("*********************END GETTING ANSWER PATH*********************")
    return para