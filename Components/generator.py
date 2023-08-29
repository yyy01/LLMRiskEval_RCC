import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import argparse, os
from tqdm import tqdm
import json
import hanlp
from DA.eda import *
from textblob import TextBlob
import numpy as np
from fractions import Fraction

from Preprocess import ecqa, esnli, qasc, senmaking, qed

class Generator(object) :
    def __init__(
        self
    ):
        self.data = {}
    
    def assign(self, dict) :
        self.data = dict

    def generator_tf(self) :
        self.data['option-number'] = 3
        self.data['options'] = []
        _dict = self.data['options']
        self.data['answer'] = True if self.data['answer'] in [True] or isinstance(self.data['answer'], str) and self.data['answer'].lower() in ['true'] else False
        _dict.extend([{'option_type' : 'GroundTruth',
                       'option_describe' : self.data['answer']},
                       {'option_type' : 'False',
                        'option_describe' : not self.data['answer']},
                        {'option_type' : 'Undetermined',
                         'option_describe' : 'Unable to determine'}])
        
        return self.data
    
    @staticmethod
    def check(x, y) :
        word = TextBlob(y).words
        return word[0].singularize().lower() != x.lower() and word[0].pluralize().lower() != x.lower()
    
    def generator_word(self, HanLP, passage) :
        
        sentence = HanLP(passage.strip(' '))

        tok, pos = list([i.lower() for i in sentence['tok']]), list(sentence['pos'])

        self.data['option-number'] = 5
        self.data['options'] = []
        _dict = self.data['options']
        _dict.append({'option_type' : 'GroundTruth',
                      'option_describe' : self.data['answer']})
        answer, answer_temp = self.data['answer'], [self.data['answer']]
        
        for _ in range(4) :
            f = False
            
            try:
                position = pos[tok.index(answer)]
                # print(tok.index(answer[i]))
                for j in range(len(pos)) :
                    if pos[j] == position and tok[j] not in answer_temp and self.check(tok[j], answer):
                        answer_temp.append(tok[j]); f = True; break
            except : pass
            if not f :
                for j in range(len(pos)) :
                    if tok[j] not in answer_temp and self.check(tok[j], answer):
                        answer_temp.append(tok[j]); f = True; break
                    
            _dict.append({'option_type' : 'Answewr Error'+str(_),
                          'option_describe' : answer_temp[-1]})
        
        return self.data
    
    @staticmethod
    def Numb_type(element) :
        return 'int' if element.isdigit() else 'float'

    def generator_number(self) :
        self.data['option-number'] = 5
        self.data['options'] = []
        _dict = self.data['options']
        _dict.append({'option_type' : 'GroundTruth',
                      'option_describe' : self.data['answer']})
        answer = str(eval(str(self.data['answer'])))
        
        for _ in range(4) :
            def get_new_numb(numb) :
                if self.Numb_type(numb) == 'float' : new_numb = str(round(np.random.uniform(-3*eval(numb), 3*eval(numb)+2), 2))
                else : new_numb = str(np.random.randint(-3*eval(numb), 3*eval(numb)+2))
                return new_numb
            
            new_answer = answer
            while new_answer == answer : new_answer = get_new_numb(answer)
            _dict.append({'option_type' : 'Answewr Error'+str(_),
                          'option_describe' : new_answer})
        
        return self.data

class Generator_text(object) :
    def __init__(self) -> None:
        self.data = {}
    
    def eda_func(self, a, x) : 
        try : 
            return eda(a, alpha_sr = 0.3, alpha_ri = 0.3, alpha_rs = 0.3, p_rd = 0.3, num_aug = x)[0]
        except : return 'None'
    
    def assign(self, dict) :
        self.data = dict

    def gt_proc(self) :
        self.data['option-number'] = 1
        self.data['options'] = []
        _dict = self.data['options']
        _dict.append({'option_type' : 'GroundTruth',
                      'option_describe' : self.data['answer']})

    @staticmethod
    def pass_judge(Q, type) :
        for _ in Q['options'] : 
            if _['option_type'] == type : return False
        return True
    
    def eda_proc(self) :
        _dict = self.data['options']
        if not self.pass_judge(self.data, 'Text Augmentation') : return self.data

        a = self.data['answer']
        _dict.append({"option_type" : '', 'option_describe' : ''})
        c_opt = _dict[-1]
        c_opt['option_type'] = 'Text Augmentation'
        # print(a)
        c_opt['option_describe'] = self.eda_func(a, 1)
        while a.find(c_opt['option_describe']) != -1 : c_opt['option_describe'] = self.eda_func(a, 1)
        self.data['option-number'] += 1

        return self.data
    
    @staticmethod
    def formula_split(formula : str) -> list:
        formula = formula.strip(' ')
        result, pos = [], 0

        def get_type(ch) :
            return 'number' if ch.isdigit() or ch == '.' else 'other'
        
        def judge_type(ori, cur) :
            return get_type(ori) == get_type(cur)

        for id, ch in enumerate(formula) :
            if judge_type(ch, formula[pos]) : continue
            else : result.append(formula[pos:id]); pos = id
        result.append(formula[pos:])
        return result
    
    @staticmethod
    def search_formula(formula) :
        match_type = '0123456789+-*/%$.= ()\\'
        pos = formula.find('=')
        if pos == -1 : return None
        else :
            st, ed = pos, pos
            while st >= 0 and formula[st] in match_type : st -= 1
            while ed < len(formula) and formula[ed] in match_type : ed += 1
            st += 1
        return formula[st:ed]
    
    @staticmethod
    def get_numb(pos : int, s : str) -> int :
        POS = pos
        while POS < len(s) and s[POS].isdigit() : POS += 1
        return POS
    
    @staticmethod
    def tf_certain_probability(probability, sequence = [True, False]):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability : break
        return item

    def formula_error_proc(self) :
        self.data['options'].append({"option_type" : '', 'option_describe' : ''})
        c_opt = self.data['options'][-1]
        c_opt['option_type'] = 'Formula Error'
        c_opt['option_describe'] = self.data['answer']
        fl = False
        a = self.data['answer']

        while True :
            pos = 0
            new_formula = c_opt['option_describe']
            while pos < len(new_formula) :
                if new_formula[pos].isdigit() :
                    fl = True
                    ed = self.get_numb(pos, new_formula)
                    numb = new_formula[pos:ed]
                    if self.tf_certain_probability([0.8, 0.2]) :
                        new_numb = str(np.random.randint(0, min(1e9+7, 5*int(numb)+2)))
                        new_formula = new_formula[:max(pos, 1)]+new_numb+(new_formula[ed:] if ed < len(new_formula) else '')
                        pos += len(new_numb)
                    else : pos = ed
                else : pos += 1
            if not fl : break
            if c_opt['option_describe'] != new_formula : 
                c_opt['option_describe'] = new_formula; break
            
        if not fl :
            c_opt['option_describe'] = self.eda_func(a, 1)
            while a.find(c_opt['option_describe']) != -1 : c_opt['option_describe'] = self.eda_func(a, 1)
        
        self.data['option-number'] += 1
        return self.data
    
    def mask_proc(self) :
        _dict = self.data['options']
        _dict.append({"option_type" : 'Mask CorrectAnswer', 'option_describe' : 'None of other options is correct.'})
        self.data['option-number'] += 1
        return self.data
    
    @staticmethod
    def generate_number(up_bound = 500):
        return str(random.randint(0, up_bound))

    @staticmethod
    def generate_operator():
        operators = ['+', '-', '*', '/']
        return random.choice(operators)

    @staticmethod
    def generate_parenthesis():
        parentheses = ['(', ')']
        return random.choice(parentheses)

    def generate_expression(self, length) :
        if length <= 1:
            return self.generate_number()
        elif length == 2:
            return self.generate_number()+self.generate_operator()+self.generate_number()

        use_parenthesis = random.choice([True, False])

        if use_parenthesis:
            opening_parenthesis = self.generate_parenthesis()
            closing_parenthesis = self.generate_parenthesis()
            inner_length = length-2
            return opening_parenthesis+self.generate_expression(inner_length)+closing_parenthesis

        else:
            left_length = random.randint(1, length-2)
            right_length = length-left_length-1
            left_expression = self.generate_expression(left_length)
            right_expression = self.generate_expression(right_length)
            operator = self.generate_operator()
            return left_expression+operator+right_expression
        
    def irrelevant_formula_proc(self) :

        formula = self.search_formula(self.data['answer'])
        a = self.data['answer']
        self.data['options'].append({"option_type" : 'Irrelevant Formula', 'option_describe' : a})

        if formula != None :
            length = int(self.generate_number(10))
            expression = self.generate_expression(length)
            self.data['options'][-1]['option_describe'] = self.data['options'][-1]['option_describe'].replace(formula, ' '+expression+' ')
            self.data['option-number'] += 1
        else : 
            while a.find(self.data['options'][-1]['option_describe']) != -1 : self.data['options'][-1]['option_describe'] = self.eda_func(a, 1)
        return self.data

    def text_options_generator(self) :
        self.gt_proc()
        self.eda_proc()
        self.formula_error_proc()
        self.mask_proc()
        self.irrelevant_formula_proc()

class Options_generator(Generator, Generator_text) :
    def __init__(self) :
        Generator.__init__(self)
        Generator_text.__init__(self)
        self.dict = {}
    
    def __reinit(self) :
        self.dict = {}

    def read_dict(self, indir) :
        self.__reinit()
        self.dict = json.load(open(indir, encoding = 'utf-8'))
        return self.dict

    @staticmethod
    def opt_type(obj) :
        if isinstance(obj, bool) or (isinstance(obj, str) and obj.lower() in ['true', 'false', 'yes', 'no']) : 
            return "T/F"
        elif isinstance(obj, (int, float, Fraction)) or (isinstance(obj, str) and re.match(r'^[-+]?\d+(\.\d+)?$', obj)) : 
            return "Number"
        elif isinstance(obj, str) :
            if obj.isalpha() : return "Word"
            else : return "Text"
        else : raise TypeError    

    def confusion_proc(self, id : str, ID : int) :
        another_id = str(np.random.randint(0, len(self.dict)))
        while another_id == id or another_id not in self.dict : another_id = str(np.random.randint(0, len(self.dict)))
        _dict = self.dict[id]
        # pos, mx = 0, len(self.dict[another_id]['sub-qa'])
        # dir = [1, -1]; cur_dir = 0

        # def move_next(pos, cur_dir) -> tuple :
        #     if pos+dir[cur_dir] >= 0 and pos+dir[cur_dir] < mx : pos += dir[cur_dir]
        #     if pos == 0 or pos == mx-1 : cur_dir = (cur_dir+1)%2
        #     return pos, cur_dir
        _id = np.random.randint(0, len(self.dict[another_id]['sub-qa']))
        for _, Q in enumerate(_dict['sub-qa']) :
            if _ != ID : continue
            Q['option-number'] += 1
            Q['options'].append({"option_type" : '', 'option_describe' : ''}); c_opt = Q['options'][-1]
            c_opt['option_type'] = 'Irrelevant Chain'
            c_opt['option_describe'] = self.dict[another_id]['sub-qa'][_id]['answer']
    
    def process(self, indir = None, dict = None, outdir = None) :
        if indir != None : self.read_dict(indir)
        elif dict != None : self.dict = dict
        else : raise ValueError
        POP_list = []
        HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
        for id in tqdm(self.dict) :
            # if int(id) > 5 : POP_list.append(id); continue
            _dict = self.dict[id]
            f = False
            pop_list = []
            for ID, q in enumerate(_dict['sub-qa']) :
                if 'options' in q : f = True; continue
                if q['answer'] in ['', None] : 
                    pop_list.append(ID)
                    continue
                f = True
                type = self.opt_type(q['answer'])
                # print(type, q['answer'])
                if type == 'Number' :
                    Generator.assign(self, q)
                    Generator.generator_number(self)
                elif type == 'T/F' :
                    Generator.assign(self, q)
                    Generator.generator_tf(self)
                elif type == 'Word' :
                    Generator.assign(self, q)
                    Generator.generator_word(self, HanLP, _dict['passage'])
                else :
                    Generator_text.assign(self, q)
                    Generator_text.text_options_generator(self)

            if not f : 
                POP_list.append(id)
                continue
            if pop_list != [] :
                for ID in pop_list.reverse() : _dict['sub-qa'].pop(ID)
        
        for id in POP_list : self.dict.pop(id)

        # print('*************************************')
        for id in self.dict :
            # print(id)
            _dict = self.dict[id]
            for ID, q in enumerate(_dict['sub-qa']) :
                type = self.opt_type(q['answer'])
                if type == 'Text' : self.confusion_proc(id, ID)
        # print('*************************************')
        
        self.item_proc()
        # print('*************************************')
        if indir == None and outdir == None : return
        if outdir == None : outdir = 'Datasets/Generator_out/'+indir.split('/')[-1]
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        self.save(outdir)
        # print('The transformed dataset is stored at: '+'Datasets/Generator_out/')
        return self.dict
                
    def item_proc(self) :
        for id in self.dict :
            _dict = self.dict[id]

            for Q in _dict['sub-qa'] :
                q_list = [i for i in range(len(Q['options']))]
                q_list = list(np.random.choice(q_list, size = len(q_list), replace = False))
                Q['input'] = [int(i) for i in q_list]

    def save(self, outdir) :
        with open(outdir, 'w', encoding = 'utf-8') as f:
            json.dump(self.dict, f, indent = 4)
        f.close()

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type = str,
        nargs = "?",
        default = None,
        help = "input directory"
    )

    parser.add_argument(
        "--save_path",
        type = str,
        nargs = "?",
        default = None,
        help = "output directory"
    )

    opt = parser.parse_args()

    Generator = Options_generator()
    for file_name in ['GSM8K', 'NoahQA', 'AQuA', 'Creak', 'StrategyQA', 'bAbi15', 'bAbi16', 'e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED'] :
        file_path = 'Datasets/Converter_out/'+file_name+'.json'
        Generator.process(indir = file_path)
    # Generator.process(indir = opt.file_path, outdir = opt.save_path)

if __name__ == '__main__' :
    main()