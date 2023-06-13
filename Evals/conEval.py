def get_score(para, TYPE = 'prompt') :
    gt_result, changed_result, SUM = {}, {}, 0
    gt_result['ori'] = {'score' : 0}

    for id in para : 
        SUM += len(para[id]['answer'])

        for _ in range(len(para[id]['answer'])) :
            if para[id]['answer'][_] == 'GroundTruth' : gt_result['ori']['score'] += 1
        for index, opt in enumerate(para[id]['new_passage']) :
            if str(index+1) not in gt_result : 
                gt_result[str(index+1)], changed_result[str(index+1)] = {'score' : 0}, {'score' : 0}
            for __, answer in enumerate(opt['answer']) :
                if answer == 'GroundTruth' : gt_result[str(index+1)]['score'] += 1
                if para[id]['answer'][__] != answer : changed_result[str(index+1)]['score'] += 1
    
    if TYPE == 'prompt' :
        for key in gt_result :
            if key == 'ori' : continue
            gt_result[key]['prompt'] = changed_result[key]['prompt'] = para['0']['new_prompt'][int(key)-1]['prompt']
    for key in gt_result :
        gt_result[key]['score'] = gt_result[key]['score']/SUM*100
        if key == 'ori' : continue
        changed_result[key]['score'] = changed_result[key]['score']/SUM*100
    
    return gt_result, changed_result, SUM

if __name__ == '__name__' :
    pass