import json
import argparse, os
from tqdm import tqdm
from DA.eda import *
from DA.token_level_attack import *

SYMBOL_LIST = ['#', '@', '～', '$', '%', '^', '&', '*', '-', '+', '=', '}', '{', '<', '>']

class Attacker(object) :
    def __init__(
        self
    ):
        self.data = {}
        self.result = {}
        self.eda_func = lambda a,x : eda(a, alpha_sr = 0.0, alpha_ri = 0.3, alpha_rs = 0.3, p_rd = 0.3, num_aug = x)[0]

    def __reinit(self) :
        self.data = {}

    def assign(self, dict) :
        self.data = dict

    def read_data(self, indir) :
        self.__reinit()
        self.data = json.load(open(indir, encoding = 'utf-8'))
        return self.data
    
    @staticmethod
    def word_level_attack(words, p, type) :
        num = round(len(words)*p)
        if type == 'ri' :
            return random_insertion(words, num)
        elif type == 'rd' :
            return random_deletion(words, p)
        elif type == 'rp' :
            return random_replacement(words, num)
        
    @staticmethod
    def tf_certain_probability(probability, sequence = [1, 2, 3]):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability : break
        return item
    
    def robustness_samples_generate(self, type : dict = {'character' : 4, 'word' : 2, 'visual' : 2}) :
        for id in tqdm(self.data) :
            self.result[id] = {}
            _dict = self.result[id]
            _dict['ori_passage'] = self.data[id]['passage']
            _dict['new_passage'] = []
            _dict = _dict['new_passage']
            sentence = self.data[id]['passage'] if self.data[id]['passage'] not in ['', None] else self.data[id]['sub-qa'][0]['question']

            words = sentence.split(' ')

            if 'character' in type :
                for _ in range(type['character']) :
                    p = np.random.randint(1, 11)/10
                    Type = np.random.randint(0, 3)
                    op_time = self.tf_certain_probability([0.4, 0.4, 0.2])
                    if Type != 2 : new_sentence = attack(sentence, attack_rate = p, attack_type = Type, op_times = op_time)
                    else : new_sentence = attack(sentence, attack_rate = p, attack_type = Type, op_times = 1, insert_char = np.random.choice(SYMBOL_LIST))
                    _dict.append({'passage' : new_sentence,
                                  'attack_type' : 'character_level'+'_'+['delete', 'repeat', 'insert'][Type],
                                  'rate' : p})

            if 'word' in type :
                # random delete
                words = [word for word in words if word != '']
                sig, words[-1] = words[-1][-1], words[-1][:-1]
                for Type in ['ri', 'rd', 'rp'] :
                    for _ in range(type['word']) :
                        p = np.random.randint(1, 11)/10
                        context = self.word_level_attack(words, p, Type)
                        _dict.append({'passage' : ' '.join(context)+sig,
                                      'attack_type' : 'word_level'+'_'+Type,
                                      'rate' : p})
                        
            if 'visual' in type :
                for Rate in [10, 50, 90] :
                    for _ in range(type['visual']) :
                        p = np.random.randint(1, 11)/10
                        _dict.append({'passage' : attack(sentence, attack_rate = p, attack_type = 3, replace_ratio = Rate/100)+'.',
                                      'attack_type' : 'visual_attack'+'_'+str(Rate)+'%',
                                      'rate' : p})
        return self.result
    
    def consistency_sample_generate(self, prompt) :
        for id in tqdm(self.data) :
            self.result[id] = {}
            _dict = self.result[id]
            _dict['ori_passage'] = self.data[id]['passage']
            _dict['new_passage'] = []
            _dict = _dict['new_passage']

            for p in prompt :
                _dict.append({'passage' : p})

        return self.result
    
    def credibility_sample_generate(self, type : list = ['ri', 'rd', 'rp']) :
        for id in tqdm(self.data) :
            self.result[id] = {}
            _dict = self.result[id]
            _dict['ori_passage'] = self.data[id]['passage']
            _dict['new_passage'] = []
            _dict = _dict['new_passage']
            sentence = self.data[id]['passage'] if self.data[id]['passage'] not in ['', None] else self.data[id]['sub-qa'][0]['question']

            words = sentence.split(' ')

            # random delete
            words = [word for word in words if word != '']
            # print(words)
            sig, words[-1] = words[-1][-1], words[-1][:-1]
            for Type in type :
                for _ in range(10) :
                    p = (_+1)/10
                    context = self.word_level_attack(words, p, Type)
                    _dict.append({'passage' : ' '.join(context)+sig,
                                  'attack_type' : 'word_level'+'_'+Type,
                                  'rate' : p})
        return self.result

    def save(self, outdir) :
        with open(outdir, 'w', encoding = 'utf-8') as f :
            json.dump(self.result, f, indent = 4)
        f.close()
    
    def process(self, type = 'robustness', indir = None, outdir = None, dict = None, p_dir = None,
                iter= {'character' : 4, 'word' : 2, 'visual' : 2}, cred : list = ['ri', 'rd', 'rp']) :
        if indir != None : self.read_data(indir)
        elif dict != None : self.assign(dict)

        if type == 'robustness' : self.robustness_samples_generate(iter)
        elif type == 'credibility' : self.credibility_sample_generate(cred)
        elif type == 'consistency' : self.consistency_sample_generate(json.load(open(p_dir, encoding = 'utf-8')))

        if indir == None and outdir == None : return self.result
        elif outdir == None : outdir = 'Datasets\Attacker_out\\'+type+'\\'+indir.split('\\')[-1][:-5]+'_attacked.json'
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        self.save(outdir)
        return self.result

def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type = str,
        nargs = "?",
        default = None,
        help = "input directory"
    )

    parser.add_argument(
        "--outdir",
        type = str,
        nargs = "?",
        default = None,
        help = "output directory"
    )

    parser.add_argument(
        "--type",
        type = str,
        nargs = "?",
        default = "robustness",
        help = "test type including robusteness, consistency and credibility"
    )

    parser.add_argument(
        "--iter",
        type = int,
        nargs = "+",
        default = [4, 2, 2],
        help = "iteration times of each level including character, word and visual"
    )

    parser.add_argument(
        "--cred_operation",
        type = str,
        nargs = "+",
        default = ['ri', 'rd', 'rp'],
        help = "operation types including random insertion for ri, random deletion for rd and random replacement for rp"
    )

    parser.add_argument(
        "--prompt_dir",
        type = str,
        nargs = "?",
        default = None
    )

    opt = parser.parse_args()
    iter = {'character' : opt.iter[0], 'word' : opt.iter[1], 'visual' : opt.iter[2]}
    generator = Attacker()
    generator.process(p_dir = opt.prompt_dir, type = opt.type, indir = opt.indir, outdir = opt.outdir, iter = iter, cred = opt.cred_operation)

if __name__ == '__main__' :
    main()