import random
import json
import math
import numpy as np
random.seed(0)
np.random.seed(0)


SYMBOL_LIST = ['#', '@', '～', '$', '%', '^', '&', '*', '-', '+', '=', '}', '{', '<', '>']

class ViualTemplate(object):
    def __init__(self, file:str='DA\\visual_letter_map.json', topn:int=20, is_equal_prob:bool=False) -> None:
        with open(file, 'r') as f:
            self.template_dict = json.load(f)
        # the hyperparameters are set according to https://github.com/UKPLab/naacl2019-like-humans-visual-attacks 
        self.topn = topn
        self.is_equal_prob = is_equal_prob
    
    def replace(self, c:str) -> str:
        if c in self.template_dict:
            chars = self.template_dict[c]['char'][0:self.topn]
            if self.is_equal_prob:
                new_c = np.random.choice(chars, 1, replace=True)[0]
            else:
                probs = np.array(self.template_dict[c]['similarity'][0:self.topn])
                probs /= np.sum(probs)
                new_c = np.random.choice(chars, 1, replace=True, p=probs)[0]
            return new_c
        else:
            return c

VISUAL_LETTER_TEMPLATE = ViualTemplate()

def attack_word(word, attack_type= 0, op_times = 1, **kwargs):

    def delete_chars(text, deleted_length):
        if deleted_length >= len(text):
            return text
        start = random.randint(0, len(text) - deleted_length)
        return text[:start] + text[start + deleted_length:]
    
    def repeat_char(text, repeat_num):
        index = random.randint(0, len(text)-1)  
        char = text[index] 
        new_text = text[:index] + char*(repeat_num+1) + text[index+1:] 
        return new_text
    
    def insert_char(text, insert_num, insert_char):
        for i in range(insert_num):
            index = random.randint(1, len(text))
            text = text[:index] + insert_char + text[index:]
        return text
    
    def repalce_char(text, ratio):
        length = math.ceil(len(text) * ratio)
        indices = set(random.sample(range(len(text)), length))
        new_text = ''
        for i in range(len(text)):
            if i in indices:
                new_text += VISUAL_LETTER_TEMPLATE.replace(text[i])
            else:
                new_text += text[i]
        return new_text
    
    attacked_word = None
    if attack_type == 0:
        attacked_word = delete_chars(word, op_times)
    elif attack_type == 1:
        attacked_word = repeat_char(word, op_times)
    elif attack_type == 2:
        attacked_word = insert_char(word, op_times, kwargs["insert_char"]) 
    elif attack_type == 3:
        attacked_word = repalce_char(word, kwargs["replace_ratio"])
    else:
        raise Exception("No such attack type {}".format(attack_type))
    
    return attacked_word


def attack(sentence, attack_rate, attack_type, **kwargs):
    words = sentence.split(' ')
    indices = [i for i,w in enumerate(words) if len(w) > 1]
    attack_word_count = math.ceil(len(indices)*attack_rate)
    select_indices = set(random.sample(indices, attack_word_count))
    attacked_words_list = []
    for i in range(len(words)):
        if i in select_indices:
            attacked_words_list.append(attack_word(words[i], attack_type, **kwargs))
        else:
            attacked_words_list.append(words[i])
    
    attacked_sentence = ' '.join(attacked_words_list)
    return attacked_sentence


if __name__ == '__main__':
    sentence = "Was Lil Jon's top ranked Billboard song a collaboration with a member of The Lox?"
    s1 = attack(sentence, attack_rate=0.1, attack_type=0, op_times=1)
    s2 = attack(sentence, attack_rate=0.1, attack_type=1, op_times=2)
    s3 = attack(sentence, attack_rate=0.1, attack_type=2, op_times=1, insert_char=SYMBOL_LIST[0])
    s4 = attack(sentence, attack_rate=0.4, attack_type=3, replace_ratio=0.5)
    print(s4)

