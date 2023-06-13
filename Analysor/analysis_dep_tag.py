import spacy
from spacy.tokens import Doc
from spacy.parts_of_speech import IDS as POS_IDS
import json
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import re
from collections import Counter

# https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
# https://github.com/explosion/spaCy/blob/b69d249a223fa4e633e11babc0830f3b68df57e2/spacy/glossary.py
DEPENDENCY_LABELS = {
    "acl": "clausal modifier of noun (adjectival clause)",
    "acomp": "adjectival complement",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "agent": "agent",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "attr": "attribute",
    "aux": "auxiliary",
    "auxpass": "auxiliary (passive)",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "complm": "complementizer",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "csubjpass": "clausal subject (passive)",
    "dative": "dative",
    "dep": "unclassified dependent",
    "det": "determiner",
    "discourse": "discourse element",
    "dislocated": "dislocated elements",
    "dobj": "direct object",
    "expl": "expletive",
    "fixed": "fixed multiword expression",
    "flat": "flat multiword expression",
    "goeswith": "goes with",
    "hmod": "modifier in hyphenation",
    "hyph": "hyphen",
    "infmod": "infinitival modifier",
    "intj": "interjection",
    "iobj": "indirect object",
    "list": "list",
    "mark": "marker",
    "meta": "meta modifier",
    "neg": "negation modifier",
    "nmod": "modifier of nominal",
    "nn": "noun compound modifier",
    "npadvmod": "noun phrase as adverbial modifier",
    "nsubj": "nominal subject",
    "nsubjpass": "nominal subject (passive)",
    "nounmod": "modifier of nominal",
    "npmod": "noun phrase as adverbial modifier",
    "num": "number modifier",
    "number": "number compound modifier",
    "nummod": "numeric modifier",
    "oprd": "object predicate",
    "obj": "object",
    "obl": "oblique nominal",
    "orphan": "orphan",
    "parataxis": "parataxis",
    "partmod": "participal modifier",
    "pcomp": "complement of preposition",
    "pobj": "object of preposition",
    "poss": "possession modifier",
    "possessive": "possessive modifier",
    "preconj": "pre-correlative conjunction",
    "predet" : "", # manual add
    "prep": "prepositional modifier",
    "prt": "particle",
    "punct": "punctuation",
    "quantmod": "modifier of quantifier",
    "rcmod": "relative clause modifier",
    "relcl": "relative clause modifier",
    "reparandum": "overridden disfluency",
    "root": "root",
    "ROOT": "root",
    "vocative": "vocative",
    "xcomp": "open clausal complement",
}


def find_attacked_word(ori_sent:str, attack_sent:str, attack_type:str) -> Tuple[List[str], List[int]]:
    ori_list = ori_sent.split(' ')
    attack_list = attack_sent.split(' ')
    if 'character_level' in attack_type or 'visual' in attack_type:
        if len(ori_list) != len(attack_list):
            print("="*20+ "error")
            print(ori_list)
            print(attack_list)
        attacked_words, attacked_indices = [], []
        for i, word in enumerate(ori_list):
            if word != attack_list[i]:
                attacked_words.append(word)
                attacked_indices.append(i)
        return attacked_words, attacked_indices
    else:
        attacked_words, attacked_indices = [], []
        attack_dict = Counter(attack_list)
        if attack_type == 'word_level_rd' or attack_type == 'word_level_rp':
            for i, word in enumerate(ori_list):
                if word not in attack_dict or attack_dict[word] <= 0:
                    attacked_words.append(word)
                    attacked_indices.append(i)
                    # 逻辑待完善
                if word in attack_dict: 
                    attack_dict[word] -= 1
            return attacked_words, attacked_indices
        else:
            index = 0
            for i, word in enumerate(ori_list):
                if len(word) == 0:
                    continue
                is_attacked = False
                while index < len(attack_list):
                    if attack_list[index] == word:
                        if (i == 0 and index != 0) \
                            or (i != 0 and index != 0 and ori_list[i-1] != attack_list[index-1]) \
                            or (i == len(ori_list)-1 and index != len(attack_list)-1) \
                            or (i != len(ori_list)-1 and index != len(attack_list)-1 and ori_list[i+1] != attack_list[index+1]):
                            is_attacked = True
                        index += 1
                        break
                    index += 1
                if is_attacked:
                    attacked_words.append(word)
                    attacked_indices.append(i)       

        return attacked_words, attacked_indices


def get_attacked_word_tag_info(ori_sent_doc:Doc, attacked_words:List[str]) -> List[int]:
    def remove_non_letters(input_str):
        return re.sub(r'[^a-zA-Z]', '', input_str)
    aw_set = set(attacked_words)
    aw_set.add(remove_non_letters(attacked_words[-1]))
    tag_counter = {k:0 for k in POS_IDS.keys()}
    for token in ori_sent_doc:
        if token.text in aw_set:
            tag_counter[token.pos_] += 1
    # print(token.text, token.pos_, token.tag_, token.is_alpha, token.is_stop)
    
    return list(tag_counter.values())


def get_relation_dict(ori_sent_doc:Doc) -> Tuple[Dict, Dict]:
    token_index = {token:i for i, token in enumerate(ori_sent_doc)}
    relation_dict = {}
    total_dependency_counter = {l:0 for l in DEPENDENCY_LABELS.keys()}
    for i, token in enumerate(ori_sent_doc):
        if (token_index[token.head], token.head.text) not in relation_dict:
            relation_dict[(token_index[token.head], token.head.text)] = []
        if (i, token.text) not in relation_dict:
            relation_dict[(i, token.text)] = []
        relation_dict[(token_index[token.head], token.head.text)].append((i, token.text, token.dep_))
        relation_dict[(i, token.text)].append((i, token.head.text, token.dep_))
        total_dependency_counter[token.dep_] += 1

    return relation_dict, total_dependency_counter


def get_attacked_dependency_relation(ori_sent_doc:Doc, attacked_dict:Dict, relation_dict:Dict, total_dependency_counter:Dict) -> List[int]:
    
    attacked_relation_set = set()
    attacked_dependency_counter = {l:0 for l in DEPENDENCY_LABELS.keys()}
    for i, token in enumerate(ori_sent_doc):
        if token.text in attacked_dict and i-1 <= attacked_dict[token.text] <= i+1:
            for rela in relation_dict[(i, token.text)]:
                if (i, rela[0]) not in attacked_relation_set:
                    attacked_relation_set.add((i, rela[0]))
                    attacked_relation_set.add((rela[0], i))
                    attacked_dependency_counter[token.dep_] += 1
    
    eps = 1e-10
    attacked_dependency_counts= [attacked_dependency_counter[k]/(total_dependency_counter[k]+eps) for k in attacked_dependency_counter.keys()]
    return attacked_dependency_counts


def cal_tag_importance(feat, label) -> None:
    if feat.shape[1] == len(list(POS_IDS.keys())):
        feat_labels = list(POS_IDS.keys())
    else:
        feat_labels = list(DEPENDENCY_LABELS.keys())
    X = feat
    Y = label
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X, Y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(clf.score(X, Y))
    for f in range(10):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


def convert_attack_type(a_t:str) -> str:
    if 'character_level' in a_t:
        return 'char_level'
    elif 'visual' in a_t:
        return 'visual_level'
    else:
        return 'word_level'


def check_empty_attack(attack_words, ori_context, para):
    if len(attack_words) == 0:
        if ori_context != para:
            print("============= no change ===============")
            print(ori_context)
            print(para)
        return True
    return False



def main(file_path:str, analysis_type:str):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    
    spacy_model = spacy.load("en_core_web_md")

    statistics_data = {
                        'char_level':{analysis_type+'_vec':[], 'word_num':[], 'is_changed':[]},
                        'visual_level':{analysis_type+'_vec':[], 'word_num':[], 'is_changed':[]},
                        'word_level':{analysis_type+'_vec':[], 'word_num':[], 'is_changed':[]}
                        }
    for id, item in tqdm(data_dict.items()):
        para_list = item["new_passage"]
        ori_sent_doc = spacy_model(item["passage"])
        if analysis_type == 'dep':
            relation_dict, total_dependency_counter = get_relation_dict(ori_sent_doc)

        for para_item in para_list:
            attack_words, attack_indices = find_attacked_word(item["passage"], para_item["para"], para_item["attack_type"])
            if check_empty_attack(attack_words, item["passage"], para_item["para"]): continue
            curr_dict = statistics_data[convert_attack_type(para_item["attack_type"])]

            if analysis_type == 'tag':
                vec = get_attacked_word_tag_info(ori_sent_doc, attack_words)
            else:
                attack_dict = {attack_words[j]:attack_indices[j] for j in range(len(attack_words))}
                vec = get_attacked_dependency_relation(ori_sent_doc, attack_dict, relation_dict, total_dependency_counter)
            
            curr_dict[analysis_type+'_vec'].append(vec)
            curr_dict['word_num'].append(len(ori_sent_doc))
            change_rate = sum(list(map(lambda a, b: 0 if a==b else 1, item["ori_answer"], para_item["AnswerPath"])))/len(item["ori_answer"])
            curr_dict['is_changed'].append(int(change_rate))
    

    for a_type in statistics_data.keys():
        print('='*20 + a_type +'='*20)
        if analysis_type == 'tag':
            cal_tag_importance(
                np.asarray(statistics_data[a_type]['tag_vec'])/np.expand_dims(np.asarray(statistics_data[a_type]['word_num']), axis=1), 
                np.asarray(statistics_data[a_type]['is_changed']))
        else:
            cal_tag_importance(
                np.asarray(statistics_data[a_type]['dep_vec']),  np.asarray(statistics_data[a_type]['is_changed']))
            
    return statistics_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type = str,
        nargs = "?",
        default = None,
        help = "input directory"
    )
    opt = parser.parse_args()
     
    main(opt.indir, analysis_type='dep') # or 'tag'