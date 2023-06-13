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
    
    def get_label(self, str) :
        str = str.split('\\')[-1]
        self.label = str[:str.find('.')]
    
    def form_converter(self, label = None, indir = None) :
        if label != None :
            self.converter = converter.DataCleaner()
            self.data = self.converter.process(label)
        else :
            self.read_data(indir)
    
    def options_generator(self) :
        self.generator = generator.Options_generator()
        outdir = 'Datasets\Generater_out\\'+self.label+'.json'
        self.data = self.generator.process(dict = self.data, outdir = outdir)
        print('Date primitives form data is stored in \''+outdir+'\'')
    
    def passage_attacker(self, config, type = 'robustness') :
        self.attacker = attacker.Attacker()
        outdir = 'Datasets\Attacker_out\\'+type+'\\'+self.label+'_attacked.json'
        self.para = self.attacker.process(type = config['type'], dict = self.data, 
                                          outdir = outdir, p_dir = config['p_dir'], 
                                          iter = config['iter'], cred = config['cred'])
        print('Attacked form data is stored in \''+outdir+'\'')
    
    def question_interpreter(self, config, type = 'robustness') :
        self.interpreter = interpreter.Interpreter()
        if not config['use_chatgpt'] : outdir = 'Datasets\Queation_out\\'+type+'\\'+self.label+'.json'
        else : outdir = 'Datasets\Interpreter_out\\'+type+'\\'+self.label+'_question.json'
        para = self.interpreter.process(type = type, dict_para = self.para, dict_data = self.data,
                                        indir_prompt = config['p_dir'], api_key = config['api_key'],
                                        num_thread = config['num_thread'], use_chatgpt = config['use_chatgpt'],
                                        outdir = outdir)
        if type(para) != None : self.para = para

        if not config['use_chatgpt'] : print('The question file is stored in \''+outdir+'\'')
        else : print('The response file is stored in \''+outdir+'\'')

    def process(self, config) :
        if config['dataset'] == None : 
            self.read_data(config['indir'])
            self.get_label(config['indir'])
        self.form_converter(config['dataset'])
        self.options_generator()
        self.passage_attacker(config, config['type'])
        self.question_interpreter(config, config['type'])

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
        "--dataset",
        type = str,
        nargs = "?",
        default = "GSM8K",
        help = "dataset name"
    )

    parser.add_argument(
        "--api_key",
        type = str,
        nargs = "+",
        default = None
    )

    parser.add_argument(
        "--num_thread",
        type = int,
        nargs = "?",
        default = 100
    )

    parser.add_argument(
        "--useChatGPT",
        action='store_true',
        help="use model ChatGPT",
    )

    parser.add_argument(
        "--type",
        type = str,
        nargs = "?",
        default = "robustness",
        help = "test type including robusteness, consistency and credibility"
    )

    parser.add_argument(
        "--prompt_dir",
        type = str,
        nargs = "?",
        default = None
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
        'indir' : opt.indir,
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