import os, sys

import requests
import json

import nltk
from nltk.corpus import wordnet

import stanza

from collections import Counter
import random

from tqdm import tqdm
from typing import List, Tuple, Union

def amuse_request(text = "Marco is eating an apple.", lang = "EN", amuse_url = 'http://127.0.0.1:3002/api/model'):
    if type(text) == list:
        http_input = []
        for sentence in text:
            http_input.append({'text':sentence, 'lang':lang})
    else:
        http_input = [{'text':text, 'lang':lang}]
    res = requests.post(amuse_url, json = http_input)
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        None

def invero_request(text = "Marco is eating an apple.", lang = "EN", invero_url = 'http://127.0.0.1:3003/api/model'):
    if type(text) == list:
        http_input = []
        for sentence in text:
            http_input.append({'text':sentence, 'lang':lang})
    else:
        http_input = [{'text':text, 'lang':lang}]
    res = requests.post(invero_url, json = http_input)
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        None

def chatgpt_request(text_request:str, api_key:str, temperature = 1.0, timeout = 10):
    res = requests.post('https://api.openai.com/v1/completions', 
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": "text-davinci-003", "prompt": text_request, "max_tokens": 4000, "temperature": temperature},
            timeout = timeout)
    return [answer["text"] for answer in json.loads(res.text)['choices']] if res.status_code == 200 else None
        

def create_nounited_dataset(sentences_list: List[str], unambiguous_candidates_path: Union[str,dict], amuse_url: str, invero_url: str, chunk_size = 16, window_span_error = 3, lang = 'EN') -> Tuple[List[dict], Counter]:
    """Generates the noUniteD dataset.

    Args:
        sentences_list (List[str]): A list of sentences, the starting dataset.
        unambiguous_candidates_path (Union[str,dict]): Path to the unambiguous candidates to be used to create the nominal part. It is generated via SynsetExplorer. If it's not a string, then it is assumed that the loaded file is passed as input.
        amuse_url (str): URL to the API endpoint of amuse.
        invero_url (str): URL to the API endpoint of invero.
        chunk_size (int, optional): Number of sentences to query amuse and invero. Defaults to 16.
        window_span_error (int, optional): The displacement between invero and amuse tokenization indices. The greater, the less are the incorrelations errors. Defaults to 3.
        lang (str, optional): Language of the sentences. Defaults to "EN" (English). Se amuse and invero for more details.

    Returns:
        (List[dict], dict): A tuple composed of a list of dictionaries (each of them is a sample) and a dictionary to check which nominal synsets are the most used and in which phrases (useful for debugging).
    """
    # load the unambiguous nominal events to be used to build the final dataset:
    if isinstance(unambiguous_candidates_path, str):
        with open(unambiguous_candidates_path, 'r') as json_file:
            candidates_unambiguous = json.load(json_file)
    else:
        candidates_unambiguous = unambiguous_candidates_path

    # initializing parameters:
    syn_phrases_id = {} # used to see which phrases has which synsets
    noUniteD_srl_result = [] # the final dataset
    nominal_found = 0 # nominal synsets used
    verbal_found = 0 # verbal synsets used
    error_chunks = 0 # number of chunk errors
    error_incorrelations = 0 # number of times a word token from the invero response is not found in amuse (because of the tokenization incorrelation between invero and amuse)
    num_generated_sentences = 0

    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)

    pbar = tqdm(range(0,len(sentences_list),chunk_size), disable=False)
    pbar_desc = lambda: f"Nominal found: {nominal_found}, Verbal found: {verbal_found}, Incorrelations: {error_incorrelations}, Chunk errors: {error_chunks}, Sentences: {num_generated_sentences}"
    pbar.set_description(pbar_desc())

    for c in pbar:
        # sending chunk request to amuse and invero:
        phrases_chunk = sentences_list[c:c+chunk_size]
        res_amuse_sentences = amuse_request(phrases_chunk, lang=lang, amuse_url=amuse_url)
        res_invero_sentences = invero_request(phrases_chunk, lang=lang, invero_url=invero_url)
        if res_amuse_sentences is None or res_invero_sentences is None:
            error_chunks += 1
            continue
        # computing results for each sentence:
        for res_amuse, res_invero in zip(res_amuse_sentences, res_invero_sentences):
            predictates = ["_"]*len(res_invero['tokens'])
            predictates_v = ["_"]*len(predictates)
            predictates_n = ["_"]*len(predictates)
            roles = {}
            phrase_nominal_found = 0
            phrase_verbals_found = 0
            wn_synsets = ["_"]*len(predictates)
            # for each word token in sentence:
            for i, token in enumerate(res_invero['tokens']):
                wn_synset_name = None
                # for each word token in sentence, compute the corresponding synset name:
                for curr_window_span_error in range(0,window_span_error):
                    if (i + curr_window_span_error)>=0 and (i + curr_window_span_error)<len(res_amuse['tokens']) and token['rawText'] == res_amuse['tokens'][i + curr_window_span_error]['text']:
                        wn_synset_name = res_amuse['tokens'][i + curr_window_span_error]['nltkSynset']
                        break
                    elif (i - curr_window_span_error)>=0 and (i - curr_window_span_error)<len(res_amuse['tokens']) and token['rawText'] == res_amuse['tokens'][i - curr_window_span_error]['text']:
                        wn_synset_name = res_amuse['tokens'][i - curr_window_span_error]['nltkSynset']
                        break
                if wn_synset_name == None:
                    error_incorrelations += 1
                    continue
                # if it's a nominal or verbal synset name:
                if wn_synset_name != 'O':
                    pos_type = wn_synset_name.split('.')[1]

                    if pos_type == 'n' and wn_synset_name in candidates_unambiguous:
                        predictates[i] = candidates_unambiguous[wn_synset_name]['frames'][0].upper() # predicates upper-case
                        predictates_n[i] = candidates_unambiguous[wn_synset_name]['frames'][0].upper() # predicates upper-case
                        
                        wn_synsets[i] = wn_synset_name

                        phrase_nominal_found += 1
                        nominal_found += 1
                        
                    elif pos_type == 'v':
                        invero_index = token['index']
                        
                        for ann in res_invero['annotations']:
                            if ann['tokenIndex'] == invero_index:
                                predictates[i] = ann['verbatlas']['frameName'].upper() # predicates upper-case
                                predictates_v[i] = ann['verbatlas']['frameName'].upper() # predicates upper-case

                                wn_synsets[i] = wn_synset_name

                                roles[str(i)] = ["_"]*len(predictates)
                                for role in ann['verbatlas']['roles']:
                                    roles[str(i)][role['span'][0]] = role['role'].lower() # roles lower-case

                        phrase_verbals_found += 1
                        verbal_found += 1
                        pass

                    pbar.set_description(pbar_desc())
            
            if any([p != "_" for p in predictates]): # if there are predicates in the phrase
                words = [token['rawText'] for token in res_invero['tokens']]
                stanza_result = nlp([words])

                assert len(stanza_result.sentences[0].words) == len(words), "Error on stanza pipeline. The length of the result does not match the length of the input!"

                noUniteD_srl_result.append({
                    'words':                words, # list of words
                    'predicates':           predictates, # list of predicates in the sentence (upper-case)
                    'predicates_v':         predictates_v, # list of verbal predicates in the sentence (upper-case)
                    'predicates_n':         predictates_n, # list of nominal predicates in the sentence (upper-case)
                    'roles':                roles, # list of roles in the sentence (lower-case)
                    'roles_v':              roles, # list of verbal roles in the sentence (lower-case)
                    'roles_n':              {}, # list of nominal roles in the sentence (not computed because VerbAtlas can't curretly do it, lower-case)
                    'num_v':                phrase_verbals_found, # number of verbal synsets found (can be used for balancing the dataset)
                    'num_n':                phrase_nominal_found, # number of nominal synsets found (can be used for balancing the dataset)
                    "lemmas":               [word.lemma for word in stanza_result.sentences[0].words], # used stanza instead of amuse because of the incorrelations errors
                    'pos_tags':             [word.upos for word in stanza_result.sentences[0].words], # used stanza instead of amuse because of the incorrelations errors
                    'dependency_heads':     [word.head for word in stanza_result.sentences[0].words], # stanza dependency heads indices
                    'dependency_relations': [word.deprel for word in stanza_result.sentences[0].words], # stanza dependency relations
                })
                num_generated_sentences += 1

                res_id = num_generated_sentences - 1
                for wn_i, wn_synset_name_i in enumerate(wn_synsets):
                    if wn_synset_name_i != '_' and wn_synset_name_i.split('.')[1] in ['v','n']:
                        if wn_synset_name_i not in syn_phrases_id: syn_phrases_id[wn_synset_name_i] = [(res_id,wn_i)]
                        else: syn_phrases_id[wn_synset_name_i] += [(res_id,wn_i)]

    return (noUniteD_srl_result, syn_phrases_id)


def save_nounited_dataset(noUniteD_srl_result: List[dict], dir_path: str, lang = 'EN', train_ratio = 0.8, num_dataset_divisions = 2, shuffle = False):
    """Save the generated noUniteD dataset in the correct way

    Args:
        noUniteD_srl_result (List[dict]): The returned dataset from the function create_nounited_dataset()
        dir_path (str): The root directory for the dataset. For example, the resulting training file will be saved in dir_path/lang/train.json.
        lang (str, optional): Language of the sentences. Defaults to "EN" (English). Se amuse and invero for more details.
        train_ratio (float, optional): The train percentage dimension over the whole dataset, between 0 and 1. Defaults to 0.8.
        num_dataset_divisions (int, optional): If it's 2, then a train and dev dataset will be generated, else if it's 3 a test dataset is generated. Defaults to 2.
        shuffle (bool, optional): If the dataset needs to be shuffled before splitting the dataset. Defaults to False.
    """
    dir_save = os.path.join(dir_path, lang)
    os.makedirs(dir_save, exist_ok=True) if os.path.exists(dir_path) else None
    if shuffle:
        random.shuffle(noUniteD_srl_result)

    test_ratio = (1.0-train_ratio)/(num_dataset_divisions-1)

    dataset_len = len(noUniteD_srl_result)
    train_len = int(dataset_len*train_ratio)
    test_len = int(dataset_len*test_ratio)

    with open(os.path.join(dir_save,'train.json'), 'w') as fout:
        json.dump(noUniteD_srl_result[:train_len], fout, indent=4)

    with open(os.path.join(dir_save,'dev.json'), 'w') as fout: # aka valid dataset
        json.dump(noUniteD_srl_result[train_len:train_len+test_len], fout, indent=4)

    if num_dataset_divisions == 3: # test dataset
        with open(os.path.join(dir_save,'test.json'), 'w') as fout:
            json.dump(noUniteD_srl_result[train_len+test_len:train_len+2*test_len], fout, indent=4)
