import argparse, os
import json
import time
from API import api
from tqdm import tqdm

class Interpreter(object) :
    def __init__(
        self
    ):
        self.data = {}
        self.para = {}
        self.result = {}
        self.question = []
    
    def read_data(self, indir) :
        self.data = json.load(open(indir, encoding = 'utf-8'))
        return self.data
    
    def read_para(self, indir) :
        self.para = json.load(open(indir, encoding = 'utf-8'))
        return self.para
    
    def get_single_question(self, _dict, prompt, para) :
        self.question.append([])
        if para == None : para = ''
        self.question[-1].append(prompt+para+'\"\n')
            
        for _, Q in enumerate(_dict['sub-qa']) :
            if _ == 0 : q = "The first question is \""
            else : q = "The next question is \""
            q = q+Q['question']+'\":\n'
            q_list = Q['input']
            for id, i in enumerate(q_list) :
                # print(i)
                q = q+'('+chr(65+id)+')'+str(Q['options'][i]['option_describe'])+'\n'
            self.question[-1].append(q)
    
    def get_question(self, type = 'robustness', 
                     prompt = 'Next, I will ask you a series of questions given a description, and you will have to choose one of several candidate options that you think is correct.  The description is \"') :
        self.question = []
        for id in tqdm(self.data) :
            _dict = self.data[id]
            if type in ['robustness', 'credibility'] :
                self.get_single_question(_dict, prompt, self.para[id]['ori_passage'])
                for p in self.para[id]['new_passage'] :
                    self.get_single_question(_dict, prompt, p['passage'])
            
            elif type == 'consistency' :
                for p in self.para[id]['new_passage'] :
                    self.get_single_question(_dict, p['passage'], self.para[id]['ori_passage'])

            else : raise AttributeError

        return self.question
    
    def get_chatgpt_answer(self, api_key : list[str], num_thread, type = 'robustness') :
        # print(self.question[0])
        chatbot = api.ChatbotWrapper(config={"api_key":api_key, "proxy":"http://127.0.0.1:1087"})
        start = time.time() 
        all_responses = chatbot.ask_batch(self.question, num_thread)
        end = time.time()
        i = 0
        for id in self.data :
            if type in ['robustness', 'credibility']:
                self.para[id]['response'] = all_responses[i]
                i += 1
            if 'new_passage' in self.para[id] :
                for _ in self.para[id]['new_passage'] :
                    _['response'] = all_responses[i]
                    i += 1
        total_time = end-start
        print("total time ", total_time)
        return self.para
    
    def get_llm_answer(self, para, res_dir) :
        res = json.load(open(res_dir, encoding = 'utf-8'))
        for id in para :
            self.para[id]['response'] = res[i]
            i += 1
            if 'new_passage' in para[id] :
                for _ in para[id]['new_passage'] :
                    _['response'] = res[i]
                    i += 1
        return para

    def save(self, dict, outdir) :
        with open(outdir, 'w', encoding = 'utf-8') as f :
            json.dump(dict, f, indent = 4)
        f.close()

    def process(self, dict_data = None, dict_para = None, indir_data = None, indir_para = None, outdir = None, 
                type = 'robustness', api_key : list[str] = None, num_thread = 10, use_chatgpt = True) :
        if indir_data != None : self.read_data(indir_data)
        elif dict_data != None : self.data = dict_data
        if indir_para != None : self.read_para(indir_para)
        elif dict_para != None : self.para = dict_para

        self.get_question(type = type)
        if use_chatgpt :
            self.get_chatgpt_answer(api_key = api_key, num_thread = num_thread, type = type)

        if outdir == None : outdir = 'Datasets/Interpreter_out/'+type+'/'+indir_data.split('/')[-1]
        if not use_chatgpt : outdir = outdir[:-5]+'_question.json'
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        self.save(self.para, outdir) if use_chatgpt else self.save(self.question, outdir)
        # print('The transformed dataset is stored at: '+'Datasets/Interpreter_out/')
        return self.para if use_chatgpt else None

def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_primitive_path",
        type = str,
        nargs = "?",
        default = None
    )

    parser.add_argument(
        "--attacked_path",
        type = str,
        nargs = "?",
        default = None
    )

    parser.add_argument(
        "--save_path",
        type = str,
        nargs = "?",
        default = None,
        help = "output directory"
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
        nargs = "?",
        default = "robustness",
        help = "test type including robustness, consistency and credibility"
    )

    opt = parser.parse_args()
    solver = Interpreter()

    for type in ['robustness', 'consistency', 'credibility'] :
        for file_name in ['GSM8K', 'NoahQA', 'AQuA', 'Creak', 'StrategyQA', 'bAbi15', 'bAbi16', 'e-SNLI', 'ECQA', 'QASC', 'SenMaking', 'QED'] :
            para_path = 'Datasets/Attacker_out/'+type+'/'+file_name+'_attacked.json'
            data_path = 'Datasets/Generator_out/'+file_name+'.json'
            solver.process(indir_data = data_path, indir_para = para_path, api_key = [''], type = type)
    # solver.process(indir_data = opt.data_primitive_indir, indir_para = opt.attacked_indir,  
    #                api_key = opt.api_key, use_chatgpt = opt.useChatGPT, type = opt.type)

if __name__ == '__main__' :
    main()