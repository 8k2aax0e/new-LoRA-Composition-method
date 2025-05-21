import os
import random
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from functools import partial
import json

def category_list():
    with open(os.path.expanduser('./resource/train_list.json'), 'r') as f:
        return json.load(f)
def eval_category_list():
    with open(os.path.expanduser('./resource/eval_list.json'), 'r') as f:
        return json.load(f)


def _set_prompt(define, input, output=''):
    if output == '':
        return 'System: ' + define + '\nInput: ' + input + '\nOutput: '
    return 'System: ' + define + '\nInput: ' + input + '\nOutput: ' + output + '\n'
    


def get_category_data(category, all_eval=False, pure_eval = False):
    import json
    files = os.listdir(os.path.expanduser(f'./dataset/natural-instructions-2.8/eng_tasks_catagry_splited/{category}'))
    files = [os.path.join(os.path.expanduser(f'./dataset/natural-instructions-2.8/eng_tasks_catagry_splited/{category}'), file) for file in files]
    datas = []
    evals = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        for i in data['Instances']:
            rand_index = random.randint(0, len(i['output'])-1)
            prompt = _set_prompt(data['Definition'][0], i['input'], i['output'][rand_index])
            datas.append(prompt)
            eval_prompt = _set_prompt(data['Definition'][0], i['input'])
            if pure_eval:
                eval_prompt = (data['Definition'][0], i['input'])
            target = i['output'][rand_index]
            evals.append({'input': eval_prompt, 'target': target})

    # get train and eval data
    combined = list(zip(datas, evals))
    random.shuffle(combined)
    datas[:], evals[:] = zip(*combined)
    train_data = datas[:int(len(datas)*0.8)]
    
    eval_data = evals[int(len(evals)*0.8):]
    if all_eval:
        eval_data = evals
    
    return train_data, eval_data

    
    
def read_sni_sample(data, tokenizer, n_samples=0, seq_len=128):
    Instances = data[:n_samples] if n_samples > 0 else data
    tokens = tokenizer(Instances, return_tensors="pt", padding=True, max_length=seq_len, truncation=True)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    return input_ids, attention_mask

def input_collection(data, tokenizer, n_samples=0, seq_len=128, labels=True, device=None):
    Instances = data[:n_samples] if n_samples > 0 else data
    padding_side = 'right' if labels else 'left'
    tokenizer.padding_side = padding_side
    eos = tokenizer.eos_token if labels else ''
    eos_instences = [i + eos for i in Instances]
    tokens = tokenizer(eos_instences, return_tensors="pt", padding=True, max_length=seq_len, truncation=True)
    if labels: tokens['labels'] = tokens['input_ids'].detach().clone()

    return tokens

def vanilla_input_collection(data, seq_len=128, labels=True):
    #pad to the right
    max_len = max([len(i) for i in data])
    if max_len > seq_len: 
        max_len = seq_len
        data = [i[:max_len] for i in data]
    masks = [[1]*len(i) + [0]*(max_len-len(i)) for i in data]
    data  = [i + [0]*(max_len-len(i)) for i in data]
    tokens = {'input_ids':torch.tensor(data), 'attention_mask':torch.tensor(masks)}
    if labels: tokens['labels'] = tokens['input_ids'].detach().clone()
    return tokens

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class Datasets(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def trans_eval_sample(eval_category_list):
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--swap_rate', type=float, default=0.5)
    args = parser.parse_args()
    
    prefix = f'pct_z{int(args.swap_rate*10)}_'
    print(f'processing {prefix}')
    from textattack.augmentation import EmbeddingAugmenter
    from transformers import LlamaTokenizer
    import json
    augmenter = EmbeddingAugmenter(transformations_per_example=5, pct_words_to_swap=args.swap_rate, fast_augment=True, high_yield=True)
    set_random_seed(42)
    eval_category_list = eval_category_list()
    for eval_category_idx in range(6,20):
        category = eval_category_list[eval_category_idx]
        _, all_eval_data = get_category_data(category, all_eval=False, pure_eval=True)

        eval_datas = [x['input'] for x in all_eval_data[:10]]
        targets = [x['target'] for x in all_eval_data[:10]]
        eval_prompts = []
        for eval_data, target in zip(tqdm(eval_datas), targets):

            new_text = augmenter.augment(eval_data[1])
            alter_evals = []
            for i in new_text:
                alter_evals.append(_set_prompt(eval_data[0], i))
            prompt = _set_prompt(eval_data[0], eval_data[1])
            eval_prompts.append({'input':prompt, 'alter_input':alter_evals,'target':target})
        with open(os.path.expanduser(f'./data/samples/{prefix}{category}.json'), 'w') as f:
            json.dump(eval_prompts, f)
        print(f'Finished {category} augmentation')

def solid_eval(eval_category_list):

    import json
    set_random_seed(42)
    eval_category_list = eval_category_list()
    for eval_category_idx in range(20):
        category = eval_category_list[eval_category_idx]
        _, all_eval_data = get_category_data(category, all_eval=False, pure_eval=True)
        length = len(all_eval_data) if len(all_eval_data) < 1000 else 1000
        eval_datas = [x['input'] for x in all_eval_data[:length]]
        targets = [x['target'] for x in all_eval_data[:length]]
        eval_prompts = []
        for eval_data, target in zip(eval_datas, targets):
            prompt = _set_prompt(eval_data[0], eval_data[1])
            eval_prompts.append({'input':prompt, 'target':target})
        with open(f'./data/evals/{category}.json', 'w') as f:
            json.dump(eval_prompts, f)
        print(f'Finished {category} augmentation')

if __name__ == '__main__':
    
    trans_eval_sample(eval_category_list)
            
