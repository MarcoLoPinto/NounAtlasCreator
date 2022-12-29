import os, sys

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from copy import deepcopy

import pdb

def return_list(nested_dict):
    res_list = list(nested_dict.keys())
    for k,v in nested_dict.items():
        res_list += return_list(v)
    return res_list

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def recursive_printing(current_exploration, level = ""):
    for syn_father, syn_sons in current_exploration.items():

        wn_syn_father = wordnet.synset(syn_father)

        p_char = "|"
        print(f"{level}{p_char}{syn_father}{p_char}", wn_syn_father.definition())
        
        for syn_son, syn_son_sons  in syn_sons.items():
            deeper_level = "└-->" if level == "" else "    "+level

            recursive_printing({syn_son:syn_son_sons}, deeper_level)


def get_whitelist(saved_exploration = {}):
    current_exploration = {'entity.n.01':{s.name():{} for s in wordnet.synset('entity.n.01').hyponyms()}} if saved_exploration == {} else saved_exploration
    current_path = []

    while True:
        cls()
        recursive_printing(current_exploration)

        print('\ncurrent path:', ' -> '.join(current_path) if len(current_path) > 0 else '<root>')
        req = input("""
commands:
    "r number" = remove from the graph the i-th father, expanding the sons (starting from 0).
    "e number" = expand and enter in the path of the i-th current root (starting from 0).
    "up" = go up in the path.
    "exit" = terminate the program.
""")
        
        if req == "exit":
            break

        elif req == "up":
            current_path = current_path[:-1]
            continue
        
        req = req.split(' ')

        pos_graph = current_exploration
        for p in current_path:
            pos_graph = pos_graph[p]

        if req[0] == 'r':
            target = list(pos_graph.keys())[int(req[1])]
            for k,v in pos_graph[target].items():
                pos_graph[k] = deepcopy(v)
            del pos_graph[target]

        elif req[0] == 'e':
            target = list(pos_graph.keys())[int(req[1])]
            hyponyms = wordnet.synset(target).hyponyms()
            pos_graph[target] = {s.name():{} for s in hyponyms}
            current_path.append(target)

    return current_exploration

if __name__ == "__main__":
    cw = get_whitelist()
    print(return_list(cw))

