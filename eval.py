import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from Evals import get_path, conEval, robEval, creEval
import matplotlib.pyplot as plt

class Evaluator :
    def __init__(
        self
    ):
        self.data = {}
        self.para = {}
        self.label = ''
        self.result = {}
    
    def read_data(self, indir) :
        self.data = json.load(open(indir, encoding = 'utf-8'))
        return self.data
    
    def read_para(self, type = 'robustness') :
        indir = 'Datasets/Interpreter_out/'+type+'/'+self.label+'.json'
        self.para = json.load(open(indir, encoding = 'utf-8'))
    
    def get_answerpath(self) :
        self.para = get_path.get_path(self.para, self.data)
    
    def eval(self, type) :
        if type == 'robustness' :
            # plt.figure(figsize=(10, 6), dpi=80)
            # plt.suptitle(type+self.label, fontsize = 20)
            error_rate, changed_rate, SUM = robEval.get_score(self.para)
            self.result['ER score'][self.label] = error_rate
            self.result['ASR score'][self.label] = changed_rate
            self.result['sum'] = SUM
            # robEval.draw_table(error_rate, changed_rate, SUM)
            # plt.tight_layout()
            # plt.show()
            
        elif type == 'consistency' :
            # print(self.para)
            gt_result, changed_result, SUM = conEval.get_score(self.para)
            self.result['Groundtruth'][self.label] = gt_result
            self.result['ASR score'][self.label] = changed_result
            self.result['sum'] = SUM
        
        elif type == 'credibility' :
            # plt.figure(figsize=(10, 6), dpi=80)
            # plt.suptitle(type+self.label, fontsize = 20)
            # print(self.para)
            Rate_list = creEval.get_rate(self.para).copy()
            SUM = 0
            for key in Rate_list :
                Rate_list[key] = round(sum([i for i in Rate_list[key]])/len(Rate_list[key]), 3)
                SUM += Rate_list[key]
            self.result['RTI score'][self.label] = round(SUM/3, 3)
            # plt.show()
    
    def save(self, outdir) :
        with open(outdir, 'w', encoding = 'utf-8') as f :
            json.dump(self.result, f, indent = 4)
        f.close()
            
    def process(self, config) :
        for type in config['type'] :
            print('-'*40)
            print(type+':')
            if type == 'robustness' : self.result = {'ER score' : {}, 'ASR score' : {}, 'sum' : {}}
            elif type == 'consistency' : self.result = {'Groundtruth' : {}, 'ASR score' : {}, 'sum' : {}}
            elif type == 'credibility' : self.result = {'RTI score' : {}}
            for self.label in config['dataset'] :
                self.read_data(os.path.join(config['path'], self.label+'.json'))
                self.read_para(type)
                self.get_answerpath()
                self.eval(type)
            outdir = 'Datasets/Eval_out/'+type+'.json'
            os.makedirs(os.path.dirname(outdir), exist_ok=True)
            self.save(outdir)
            print(self.result)
            print('-'*40)

def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type = str,
        nargs = "+",
        default = None,
        help = "dataset labels"
    )

    parser.add_argument(
        "--file_path",
        type = str,
        nargs = "?",
        default = None,
        help = "input directory"
    )

    parser.add_argument(
        "--type",
        type = str,
        nargs = "+",
        default = ["robustness", "consistency", "credibility"],
        help = "test type including robusteness, consistency and credibility"
    )

    opt = parser.parse_args()

    config = {
        'type' : opt.type,
        'path' : opt.file_path,
        'dataset' : opt.dataset
    }

    evaluator = Evaluator()
    # path = 'Datasets/Generator_out'
    # config = {'type' : ['robustness', 'consistency', 'credibility'], 'path' : path, 
    #           'dataset' : ['GSM8K', 'NoahQA', 'AQuA', 'Creak', 'StrategyQA', 'bAbi15', 'bAbi16', 'e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED']}
    # evaluator.process(config)
    evaluator = Evaluator()
    evaluator.process(config)


if __name__ == '__main__' :
    main()