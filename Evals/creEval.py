import matplotlib.pyplot as plt
import numpy as np
import math

def get_rate(para) :
    rate_list = {'word_level_ri' : [], 'word_level_rd' : [], 'word_level_rp' : []}
    
    for id in para :
        answer = para[id]['answer']
        _dict = []

        for _, type in enumerate(['word_level_ri', 'word_level_rd', 'word_level_rp']) :
            for __ in range(len(answer)) : 
                _dict[type] = 0
                for ID in range(10*_, 10*_+10) :
                    if para[id]['new_passage'][ID]['AnswerPath'][__] != answer[__] :
                        _dict[type] = para[id]['new_passage'][ID]['rate']
                        break
                if _dict[type] == 0 : _dict[type] = 1.0
                rate_list[type].append(_dict[type])
    
    for _, type in enumerate(['word_level_ri', 'word_level_rd', 'word_level_rp']) :
            plt.subplot(1, 3, _+1)
            draw_table(rate_list[type], type)

    return rate_list

def draw_table(rate_list : list, type : str) :
    width = 0.35
    plt.rcParams.update({'font.size': 10})
    key, dict = [], {}
    for _ in rate_list : 
        if _ not in key : key.append(_); dict[_] = 1
        else : dict[_] += 1
    key = sorted(key)
    x = np.array(rate_list)
    y = np.array([1 for _ in rate_list])

    plt.bar(np.arange(len(key)), tuple(dict[i] for i in key), width, label="Changed", color="#87CEFA")
    
    plt.xlabel('Rate' )
    plt.ylabel('State')
    num = max([dict[i] for i in key])

    plt.title(type)
    plt.xticks(np.arange(len(key)), key)
    plt.yticks(np.arange(0, num, math.ceil(num/10)))

if __name__ == '__main__' :
    pass 