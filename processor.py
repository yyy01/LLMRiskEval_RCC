import argparse
import json
from Components import converter, generator, attacker, interpreter

class Processor :
    def __init__(
        self
    ):
        self.data = {}
        self.para = {}
        self.label = ''
    
    def read_data(self, indir) :
        self.data = json.load(open(indir, encoding = 'utf-8'))
        return self.data
    
    def options_generator(self) :
        self.generator = generator.Options_generator()
        outdir = 'Datasets/Generator_out/'+self.label+'.json'
        self.data = self.generator.process(dict = self.data, outdir = outdir)
        print('Date primitives form data is stored in \"'+outdir+'\"')
    
    def passage_attacker(self, config, type = 'robustness') :
        self.attacker = attacker.Attacker()
        outdir = 'Datasets/Attacker_out/'+type+'/'+self.label+'_attacked.json'
        self.para = self.attacker.process(type = type, dict = self.data, 
                                          outdir = outdir, p_dir = config['p_dir'], 
                                          iter = dict(zip(['character', 'word', 'visual'], config['iter'])), cred = config['cred'])
        print('Attacked form data is stored in \"'+outdir+'\"')
    
    def question_interpreter(self, config, type = 'robustness') :
        self.interpreter = interpreter.Interpreter()
        if not config['use_chatgpt'] : outdir = 'Datasets/Queation_out/'+type+'/'+self.label+'.json'
        else : 
            outdir = 'Datasets/Interpreter_out/'+type+'/'+self.label+'_question.json'
        para = self.interpreter.process(type = type, dict_para = self.para, dict_data = self.data,
                                        api_key = config['api_key'], num_thread = config['num_thread'], 
                                        use_chatgpt = config['use_chatgpt'],
                                        outdir = outdir)
        if para != None : self.para = para

        if not config['use_chatgpt'] : print('The question file is stored in \"'+outdir+'\"')
        else : print('The response file is stored in \"'+outdir+'\"')

    def process(self, config) :
        for self.label in config['dataset'] :
            print('-'*40)
            indir = 'Datasets/Converter_out/'+self.label+'.json'
            self.read_data(indir)
            self.options_generator()
            for type in config['type'] :
                self.passage_attacker(config, type)
                self.question_interpreter(config, type)
            print('-'*40)

def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type = str,
        nargs = "+",
        default = ['GSM8K', 'NoahQA', 'AQuA', 'Creak', 'StrategyQA', 'bAbi15', 'bAbi16', 'e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED'],
        help = "dataset name"
    )

    parser.add_argument(
        "--api_key",
        type = str,
        nargs = "+",
        default = ['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']
    )

    parser.add_argument(
        "--num_thread",
        type = int,
        nargs = "?",
        default = 10
    )

    parser.add_argument(
        "--useChatGPT",
        action='store_true',
        help="use model ChatGPT",
    )

    parser.add_argument(
        "--type",
        type = str,
        nargs = "+",
        default = ["robustness", "consistency", "credibility"],
        help = "test type including robustess, consistency and credibility"
    )

    parser.add_argument(
        "--prompt_dir",
        type = str,
        nargs = "?",
        default = 'Datasets/prompt_template.json'
    )

    parser.add_argument(
        "--cred_operation",
        type = str,
        nargs = "+",
        default = ['ri', 'rd', 'rp'],
        help = "operation types including random insertion for ri, random deletion for rd and random replacement for rp"
    )

    parser.add_argument(
        "--iter",
        type = int,
        nargs = "+",
        default = [4, 2, 2],
        help = "iteration times of each level including character, word and visual"
    )

    opt = parser.parse_args()

    config = {
        'type' : opt.type,
        'dataset' : opt.dataset,
        'p_dir' : opt.prompt_dir,
        'api_key' : opt.api_key,
        'num_thread' : opt.num_thread,
        'use_chatgpt' : opt.useChatGPT,
        'iter' : opt.iter,
        'cred' : opt.cred_operation
    }

    processor = Processor()
    processor.process(config)


if __name__ == '__main__' :
    main()