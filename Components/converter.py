import re
import jsonlines
import argparse, os
from tqdm import tqdm
import json
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Preprocess import ecqa, esnli, qasc, senmaking, qed

class Converter(object) :
    def __init__(
        self
    ):
        self.result = {}
        self.data = {}

    def __reinit(self) :
        self.result = {}
        self.data = {}

    def read_8k(self, indir : str) :
        self.__reinit()
        id = 0
        for dict, _dict in zip(jsonlines.Reader(open(indir, encoding = 'utf-8')), jsonlines.Reader(open(indir[:-6]+'_socratic.jsonl', encoding = 'utf-8'))) :
            dict['answer'] = _dict['answer']
            self.data[str(id)] = dict
            id += 1
        return self.data
    
    @staticmethod
    def find(s:str, S:str) -> int :
        return S.lower().find(s.lower())
    
    def clean_8k(self) :
        Type = {}
        for id in tqdm(self.data) :
            dict = self.data[id]
            self.result[id] = {}
            q, a = dict['question'], dict['answer'].split('\n')
            _dict = self.result[str(id)]
            p, q, ans = q, '', a[-1]
            if '?' in p : 
                q = re.split('[,.;]', p)[-1]
                p = p[:-len(q)]
            else : q, p = p[self.find('calculate', p):], p[:self.find('calculate', p)]
            _dict['passage'], _dict['sub-qa'] = p, []
            if str(len(a)) not in Type : Type[str(len(a))] = 1
            else : Type[str(len(a))] += 1

            for i in range(len(a)-1) :
                question, a[i] = a[i].split('**')[0], a[i].split('**')[1]
                answer = a[i][:a[i].find('<<')]+a[i][a[i].find('>>')+2:]
                if answer == '' : continue
                _dict['sub-qa'].append({'question' : question,
                                        'answer' : answer})
                
            randint = np.random.randint(0, 2)
            _dict['sub-qa'].append({'question' : 'The answer to question \"'+q+('\" is '+'\"'+ans+'\"'+', is it right?' if randint == 0 else  '\" is not '+'\"'+ans+'\"'+', is it right?'),
                            'answer' : True if randint == 0 else False})

        return Type, sum([Type[key] for key in Type.keys()]), sum([int(key)*Type[key] for key in Type.keys()])
    
    def read_Noah(self, indir : str) :
        self.__reinit()
        self.data = json.load(open(indir, encoding = 'utf-8')) 
        return self.data
    
    @staticmethod
    def get_noah_c(passage : list) -> str :
        passage.sort(key = lambda x : x[0])
        question = ''
        for p in passage : question += p[1]
        return question
    
    def clean_Noah(self) :
        Type = {}
        for id, key in enumerate(tqdm(self.data.keys())) :
            dict = self.data[key]
            self.result[str(id)] = {}
            _dict = self.result[str(id)]
            p = self.get_noah_c(dict['passage'])
            count = str(int(dict['max_turn']))

            if count not in Type : Type[count] = 1
            else : Type[count] += 1

            _dict['passage'], q = p, dict['qa_pairs'][-1]['question']
            ans = dict['qa_pairs'][-1]['ans'] if 'ans' in dict['qa_pairs'][-1] else dict['qa_pairs'][-1]['answer']
            _dict['sub-qa'] = []

            for i in range(int(dict['max_turn'])-1) :
                answer = dict['qa_pairs'][i]['ans'] if 'ans' in dict['qa_pairs'][i] else dict['qa_pairs'][i]['answer']
                _dict['sub-qa'].append({'question' : dict['qa_pairs'][i]['question'],
                                        'answer' : answer})

            randint = np.random.randint(0, 2)
            _dict['sub-qa'].append({'question' : 'The answer to question \"'+q+('\" is '+'\"'+ans+'\"'+', is it right?' if randint == 0 else  '\" is not '+'\"'+ans+'\"'+', is it right?'),
                          'answer' : True if randint == 0 else False})

        return Type, sum([Type[key] for key in Type.keys()]), sum([int(key)*Type[key] for key in Type.keys()])

    def read_aq(self, indir : str) :
        self.__reinit()
        id = 0
        for dict in jsonlines.Reader(open(indir, encoding = 'utf-8')) :
            self.data[str(id)] = dict
            id += 1
        return self.data
    
    def clean_aq(self) :
        Type = {'1' : 0}
        for id in tqdm(self.data) :
            _dict = self.data[id]
            self.result[id] = {}
            dict = self.result[id]

            q, answer = _dict['question'], _dict['options'][ord(_dict['correct'])-ord('A')][2:]
            option = _dict['options'].copy()
            option = option[0:ord(_dict['correct'])-ord('A')]+option[ord(_dict['correct'])-ord('A')+1:len(option)]

            for key in ['question', 'options', 'rationale', 'correct'] : _dict.pop(key)
            dict['passage'] = q
            dict['sub-qa'] = []
            Type[str(1)] += 1

            dict['sub-qa'].append({'question' : 'Choose one of several candidate options that you think is correct.',
                                    'answer' : answer, 
                                    'option-number' : len(option)+1,
                                    'options' : [{'option_type' : 'GroundTruth',
                                                  'option_describe' : answer}]})
            dict = dict['sub-qa'][-1]['options']
            for ID, i in enumerate(option) :
                dict.append({'option_type' : 'Value Error'+str(ID),
                              'option_describe' : i[2:]})

        return Type, sum([Type[key] for key in Type.keys()]), sum([int(key)*Type[key] for key in Type.keys()])

    def read_strategyqa(self, indir : str) :
        self.__reinit()
        self.data = json.load(open(indir, encoding = 'utf-8'))
        return self.data

    def clean_strategyqa(self) :
        Type = {'1' : 0}
        for id, dict in enumerate(tqdm(self.data)) :
            self.result[str(id)] = {}
            _dict = self.result[str(id)]
            _dict['passage'] = ''.join(dict['facts'])
            _dict['sub-qa'] = []
            _dict = _dict['sub-qa']
            Type[str(1)] += 1
            _dict.append({'question' : dict['question'],
                          'answer' : dict['answer']})
        return Type, sum([Type[key] for key in Type.keys()]), sum([int(key)*Type[key] for key in Type.keys()])

    def read_creak(self, indir) :
        self.__reinit()
        id = 0
        for dict in jsonlines.Reader(open(indir, encoding = 'utf-8')) :
            self.data[str(id)] = dict
            id += 1
        return self.data
    
    def clean_creak(self) :
        Type = {'1' : 0}
        for id in tqdm(self.data) :
            dict = self.data[id]
            self.result[id] = {}
            _dict = self.result[id]
            _dict['passage'], _dict['sub-qa'] = dict['explanation'], []
            Type[str(1)] += 1
            _dict = _dict['sub-qa']

            _dict.append({'question' : dict['sentence'][:-1]+', is it right?',
                          'answer' : True if dict['label'] == 'true' else False})
        return Type, sum([Type[key] for key in Type.keys()]), sum([int(key)*Type[key] for key in Type.keys()])

    def read_babi(self, indir, type = 'deduction') :
        self.__reinit()

        with open(indir, 'r', encoding = 'utf-8') as f:
            context = f.read().splitlines()

        if type == 'deduction' :
            num = int(len(context)/12)
            for id, _ in enumerate(context) :
                if id%12 in range(9) : context[id] = context[id][2:]
                else : context[id] = context[id][3:]
            for i in range(num) :
                self.data[str(i)] = {}
                self.data[str(i)]['passage'], self.data[str(i)]['question'] = ' '.join(context[12*i:12*i+8]), context[12*i+8:12*i+12]

        else :
            num = int(len(context)/10)
            for id, _ in enumerate(context) :
                if id%10 in range(9) : context[id] = context[id][2:]
                else : context[id] = context[id][3:]
            for i in range(num) :
                self.data[str(i)] = {}
                self.data[str(i)]['passage'], self.data[str(i)]['question'] = ' '.join(context[10*i:10*i+9]), context[10*i+9:10*i+10]

        return self.data

    def clean_babi(self, type = 'deduction') :
        Type = {}
        for id in tqdm(self.data) :
            self.result[id] = {}
            _dict, dict = self.result[id], self.data[id]
            _dict['passage'], _dict['sub-qa'] = dict['passage'], []

            _dict = _dict['sub-qa']

            answer = []
            for q in dict['question'] : answer.append(re.split(r'\s+', q)[-3 if type == 'deduction' else -4])

            if str(len(answer)) not in Type : Type[str(len(answer))] = 1
            else : Type[str(len(answer))] += 1

            for i, q in enumerate(dict['question']):
                _dict.append({'question' : q[:q.find('?')+1],
                              'answer' : answer[i]})

        return Type, sum([Type[key] for key in Type.keys()]), sum([int(key)*Type[key] for key in Type.keys()])


class DataCleaner(Converter) :
    def save(self, outdir) :
        with open(outdir, 'w', encoding = 'utf-8') as f:
            json.dump(self.result, f, indent = 4)
        f.close()
    
    def process_8k(self, indir, outdir) :
        self.read_8k(indir)
        result, sum, qsum = self.clean_8k()
        if outdir != None : self.save(outdir)
        return result, sum, qsum

    def process_Noah(self, indir, outdir) :
        self.read_Noah(indir)
        result, sum, qsum = self.clean_Noah()
        if outdir != None : self.save(outdir)
        return result, sum, qsum
    
    def process_aq(self, indir, outdir) :
        self.read_aq(indir)
        result, sum, qsum = self.clean_aq()
        if outdir != None : self.save(outdir)
        return result, sum, qsum
    
    def process_strategyqa(self, indir, outdir) :
        self.read_strategyqa(indir)
        result, sum, qsum = self.clean_strategyqa()
        if outdir != None : self.save(outdir)
        return result, sum, qsum
    
    def process_creak(self, indir, outdir) :
        self.read_creak(indir)
        result, sum, qsum = self.clean_creak()
        if outdir != None : self.save(outdir)
        return result, sum, qsum
    
    def process_babi15(self, indir, outdir) :
        self.read_babi(indir, 'deduction')
        result, sum, qsum = self.clean_babi('deduction')
        if outdir != None : self.save(outdir)
        return result, sum, qsum
    
    def process_babi16(self, indir, outdir) :
        self.read_babi(indir, 'induction')
        result, sum, qsum = self.clean_babi('induction')
        if outdir != None : self.save(outdir)
        return result, sum, qsum
    
    def process(self, Label = ['GSM8K', 'NoahQA', 'AQuA', 'Creak', 'StrategyQA', 'bAbi15', 'bAbi16', 'e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED']) :
        key_dict = {'GSM8K' : '.jsonl', 'NoahQA': '.json', 'AQuA' : '.jsonl', 
                    'Creak' : '.jsonl', 'StrategyQA' : '.json', 'bAbi15' : '.txt', 
                    'bAbi16' : '.txt', 'e-SNLI' : '.csv', 'ECQA' : '.csv',
                    'QASC' : '.jsonl', 'SenMaking' : '.jsonl', 'QED' : '.jsonl'}
        for label in Label :
            indir = 'Datasets/Raw_Data/'+label+key_dict[label]
            outdir = 'Datasets/Converter_out/'+label+'.json'
            func_dict = {'GSM8K' : self.process_8k, 'NoahQA': self.process_Noah, 'AQuA' : self.process_aq, 'Creak' : self.process_creak, 
                        'StrategyQA' : self.process_strategyqa, 'bAbi15' : self.process_babi15, 'bAbi16' : self.process_babi16,
                        'e-SNLI' : esnli.process, 'ECQA' : ecqa.process, 'QASC' : qasc.process,
                        'SenMaking' : senmaking.process, 'QED' : qed.process}
            
            if outdir != None : os.makedirs(os.path.dirname(outdir), exist_ok=True)
            if label in ['e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED'] : self.result = func_dict[label](indir, outdir)
            else : func_dict[label](indir, outdir)
        # print('The transformed dataset is stored at: '+'Datasets/Converter_out/')
        return self.result

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type = str,
        nargs = "+",
        default = ['GSM8K', 'NoahQA', 'AQuA', 'Creak', 'StrategyQA', 'bAbi15', 'bAbi16', 'e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED'],
        help = "dataset name"
    )

    opt = parser.parse_args()

    converter = DataCleaner()
    converter.process(opt.dataset) 
    
    print('The transformed dataset is stored at: '+'Datasets/Converter_out/')
    
if __name__ == '__main__' :
    main()