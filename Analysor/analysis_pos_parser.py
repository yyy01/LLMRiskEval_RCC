import numpy as np
import argparse
import json
import re
import math
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP

PASER_LABEL = {
    "ADJP":	"Adjective Phrase.",
    "ADVP": "Adverb Phrase.",
    "CONJP": "Conjunction Phrase.",
    "FRAG": "Fragment.",
    "INTJ": "Interjection. Corresponds approximately to the part-of-speech tag UH.",
    "LST": "List marker. Includes surrounding punctuation.",
    "NAC": "Not a Constituent; used to show the scope of certain prenominal modifiers within an NP.",
    "NP": "Noun Phrase.",
    "NX": "Used within certain complex NPs to mark the head of the NP. Corresponds very roughly to N-bar level but used quite differently.",
    "PP": "Prepositional Phrase.",
    "PRN": "Parenthetical.",
    "PRT": "Particle. Category for words that should be tagged RP.",
    "QP": "Quantifier Phrase (i.e. complex measure/amount phrase); used within NP.",
    "RRC": "Reduced Relative Clause.",
    "UCP": "Unlike Coordinated Phrase.",
    "VP": "Vereb Phrase.",
    "WHADJP": "Wh-adjective Phrase. Adjectival phrase containing a wh-adverb, as in how hot.",
    "WHAVP": "Wh-adverb Phrase. Introduces a clause with an NP gap. May be null (containing the 0 complementizer) or lexical, containing a wh-adverb such as how or why.",
    "WHNP": "Wh-noun Phrase. Introduces a clause with an NP gap. May be null (containing the 0 complementizer) or lexical, containing some wh-word, e.g. who, which book, whose daughter, none of which, or how many leopards.",
    "WHPP": "Wh-prepositional Phrase. Prepositional phrase containing a wh-noun phrase (such as of which or by whose authority) that either introduces a PP gap or is contained by a WHNP.",
    "X": "Unknown, uncertain, or unbracketable. X is often used for bracketing typos and in bracketing the...the-constructions."
}

class Para_analyse : 
    def __init__(
            self
    ) :
        self.para = {}
        self.result = {}

    def read_para(self, indir) :
        self.para = json.load(open(indir, encoding = 'utf-8'))
        self.result['dataset'] = indir.split('\\')[-2]
        return self.para
    
    @staticmethod
    def word_compare(ori_list, new_list, pos) -> bool :
        for i in range(pos-4,pos+5) :
            try :
                if new_list[i] == ori_list[pos] : return True
            except : continue
        return False

    def compare_para(self, ori_str : str, new_str : str, type : str, id : str) -> tuple :
        # print(ori_str, '\n', new_str)
        ori_str, new_str = ori_str.strip(), new_str.strip()
        ori_list, new_list = re.split(r'\s+', ori_str), re.split(r'\s+', new_str)
        if (int(id) < 22918) :
            ori_list[-1] = ori_list[-1][:-1]
            new_list[-1] = new_list[-1][:-1]
        ori_list = [word for word in ori_list if word != '']
        new_list = [word for word in new_list if word != '']
        result_list = []
        mx_index = None
        
        if type == 'word_level_ri' :
            flag = 0
            for i in range(len(ori_list)) : 
                # print(ori_list)
                try :
                    new_pos = new_list[flag:].index(ori_list[i])
                except :
                    print(ori_str, '\n', new_str)
                    raise 'YWT'
                result_list.extend([i]*(new_pos-flag))
                flag = new_pos+1
            result_list.extend([len(ori_list)]*(len(new_list)-flag))
            mx_index = len(ori_list)
        elif type == 'word_level_rd' :
            flag = 0
            result_list = list(range(len(ori_list)))
            # print(ori_list, '\n', new_list)
            for i in range(len(new_list)) : 
                try :
                    new_pos = ori_list[flag:].index(new_list[i])
                except:
                    print(ori_str, '\n', new_str)
                    raise 'error'
                # print(flag, new_list[i],new_pos+flag)
                result_list.remove(new_pos+flag)
                flag = new_pos+flag+1
        else :
            for i in range(len(ori_list)) :
                # if 'visual' in type or 'character' in type :
                #     if ori_list[i] != new_list[i] : result_list.append(i)
                # else :
                if not self.word_compare(ori_list, new_list, i) : result_list.append(i)
                else : continue
    
        if mx_index == None : mx_index = len(ori_list)-1
        
        return mx_index, result_list
    
    def para_pos_statistic(self, judge = 'GroundTruth') :
        length = min([len(re.split(r'\s+', self.para[id]['ori_context'])) for id in self.para])
        segment = 5
        TYPE = ['character_level_repeat', 'character_level_delete', 'character_level_insert',
                'word_level_ri', 'word_level_rd', 'word_level_rp',
                'visual_attack_10%', 'visual_attack_50%', 'visual_attack_90%']
        for _ in TYPE : self.result[_] = {'head' : 0, '1/5' : 0, '2/5' : 0, '3/5' : 0, '4/5' : 0, '5/5' : 0, 'tail' : 0}
        
        for id in self.para :
            print(id)
            ori_para = self.para[id]['ori_context']
            for _dict in self.para[id]['new_context'] :
                attack_type, new_para, rate = _dict['attack_type'], _dict['para'], _dict['rate']
                weight = 0
                for index in range(len(_dict['AnswerPath'])) :
                    if judge == 'GroundTruth' : 
                        if _dict['AnswerPath'][index] == judge : continue
                    else : 
                        if _dict['AnswerPath'][index] == self.para[id]['ori_answer'][index] : continue
                    weight += 1
                if weight > 0 :
                    mx_index, id_list = self.compare_para(ori_para, new_para, attack_type, id)
                    mn_index, count_times = 0, len(id_list)
                    gap = (mx_index-mn_index+1e-5)/segment
                    discrete_index = [1+int((_-mn_index)/gap) for _ in id_list if _ != mn_index and _ != mx_index]
                    # print(id_list, mx_index, discrete_index)
                    # return
                    pos = 0
                    for i in id_list :
                        if i == mn_index or i == mx_index :
                            if i == mn_index : self.result[attack_type]['head'] += weight*(1/count_times if mn_index != mx_index else 1/2/count_times)
                            if i == mx_index : self.result[attack_type]['tail'] += weight*(1/count_times if mn_index != mx_index else 1/2/count_times)
                            continue 
                        self.result[attack_type][str(discrete_index[pos])+'/5'] += weight*1/count_times
                        pos += 1

    @staticmethod
    def get_parse(nlp, sentence) :
        return nlp.parse(sentence)

    def para_parser_statistic(self, judge = 'GroundTruth') :
        nlp = StanfordCoreNLP(r'stanford-corenlp-4.5.4')
        TYPE = ['character_level_repeat', 'character_level_delete', 'character_level_insert',
                'word_level_ri', 'word_level_rd', 'word_level_rp',
                'visual_attack_10%', 'visual_attack_50%', 'visual_attack_90%']
        for _ in TYPE : self.result[_] = {key: 0 for key in PASER_LABEL.keys()}
        
        for id in self.para :
            print(id)
            ori_para = self.para[id]['ori_context']
            for _dict in self.para[id]['new_context'] :
                attack_type, new_para, rate = _dict['attack_type'], _dict['para'], _dict['rate']
                weight = 0
                for index in range(len(_dict['AnswerPath'])) :
                    if judge == 'GroundTruth' : 
                        if _dict['AnswerPath'][index] == judge : continue
                    else : 
                        if _dict['AnswerPath'][index] == self.para[id]['ori_answer'][index] : continue
                    weight += 1
                if weight > 0 :
                    ori_list = re.split(r'\s+', ori_para)
                    _, id_list = self.compare_para(ori_para, new_para, attack_type, id)
                    count_times = len(id_list)
                    parser_list = [i.strip() for i in self.get_parse(nlp, ori_para).split('\r\n')]
                    # print(self.get_parse(nlp, ori_para))
                    pos = 0
                    for i in id_list :
                        if i == len(ori_list) : continue
                        for __, parser in enumerate(parser_list[pos:]) : 
                            if parser.find(ori_list[i]+')') != -1 : 
                                # print(ori_list[i], parser)
                                pos = pos+__+1
                                try :
                                    self.result[attack_type][parser[1:parser.find(' ')]]  += weight*1/count_times
                                except : continue

    @staticmethod
    def draw_table(rate_result : dict, type : str) :
        width = 0.35
        keys = list(rate_result.keys())
        RATE = [rate_result[key] for key in keys]

        plt.bar(np.arange(len(rate_result)), tuple(rate_result[key] for key in keys), width, label="Importance", color=plt.cm.Blues(np.asarray(RATE)/np.array(max(RATE))))
        
        plt.xlabel('Type', fontsize = 25)
        plt.ylabel('Frequency',fontsize = 25)
        # plt.ylabel('Changed Rate')

        num = max([rate_result[key] for key in rate_result.keys()])
        if type == 'word_level_ri' : Type = 'word_level_insert'
        elif type == 'word_level_rd' : Type = 'word_level_delete'
        elif type == 'word_level_rp' : Type = 'word_level_replace'
        else : Type = type

        plt.title(Type, fontsize = 25)
        plt.xticks(np.arange(len(rate_result)), keys)
        plt.yticks(np.arange(0, num, math.ceil(num/10)))
        # plt.yticks(np.arange(0, 1, 0.1))
        
        # plt.rcParams.update({'font.size': 35})
        plt.legend(loc="upper right")

    def process_para(self, indir, type, TYPE = 'Changed') :
        self.read_para(indir)
        if type == 'pos' : self.para_pos_statistic(TYPE)
        if type == 'parser' : self.para_parser_statistic(TYPE)
        print(self.result)
        print(self.result)
        plt.figure(figsize=(10, 8), dpi=80)
        plt.rcParams.update({'font.size': 20})
        # plt.suptitle(self.result['dataset'], fontsize = 20)
        for id, type in enumerate(self.result) :
            if type == 'dataset' : continue
            plt.subplot(3, 3, id)
            self.draw_table(self.result[type], type)
        plt.tight_layout(pad = 1, h_pad = -0.5, w_pad = -2)
        plt.show()

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
    opt = parser.parse_args()

    solver = Para_analyse()
    solver.process_para(opt.indir, 'pos') # or 'parser'


if __name__ == '__main__' :
    index = 0
    PARA = {}

    main()