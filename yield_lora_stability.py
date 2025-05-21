from transformers import LlamaTokenizer
from datasets import load_dataset
import os
import sys
from transformers import LlamaForCausalLM, LlamaTokenizer
from part_lora import get_empty_full_model, get_cache, noise_cache
from data_loader import category_list, get_category_data, eval_category_list
import torch
import numpy as np
import random
from tqdm import tqdm
import json

eval_category_list = eval_category_list()
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_category_idx', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--sample', type=int, default=5)
    parser.add_argument('--sample_prefix', type=str, default='')
    return parser.parse_args()

def generate_to_select_lora_by_x(batch_input, batch_alter_inputs, args):
    model = LlamaForCausalLM.from_pretrained(os.path.expanduser('llama-7b'))
    
    model = get_empty_full_model(model)
    model.bfloat16().to('cuda')
    model.eval()
    caches = get_cache(prefix='eos_added_',device='cuda')
    category_list = list(caches.keys())
    category_mean_standard_deviation = []
    category_mean_MAD = []

    for category in tqdm(category_list):
        with torch.no_grad():
            model.load_state_dict(caches[category], strict=False)
            model.bfloat16()
            origin_input_length = batch_input['input_ids'].shape[-1]
            origin_output = model.generate(**batch_input, max_new_tokens=32, do_sample=False, output_scores=True, output_logits=True, return_dict_in_generate=True, num_beams=1, num_return_sequences=1)
            origin_scores = model.compute_transition_scores(sequences=origin_output['sequences'], scores=origin_output['logits'], normalize_logits=True)

            output_sequences = origin_output['sequences'][:,origin_input_length-1:-1]

            attention_masks = torch.stack([(i != args.tokenizer.pad_token_id).int() for i in origin_output['sequences']]).to(torch.int64)
            output_masks = attention_masks[:,origin_input_length-1:-1]

            batch_noised_std = []
            batch_noised_diff = []
            
            for output_sequence, output_masks, batch_alter_input, origin_score in zip(output_sequences, output_masks, batch_alter_inputs, origin_scores):
                alter_input_ids = batch_alter_input['input_ids']
                alter_input_length = alter_input_ids.shape[-1]
                alter_attention_masks = batch_alter_input['attention_mask']

                repeated_output_sequence = output_sequence.unsqueeze(0).repeat(alter_input_ids.shape[0],1)
                repeated_output_masks = output_masks.unsqueeze(0).repeat(alter_input_ids.shape[0],1)
                alter_input = torch.cat([alter_input_ids, repeated_output_sequence], dim=1)
                alter_attention_mask = torch.cat([alter_attention_masks, repeated_output_masks], dim=1)
                
                noise_output = model(alter_input, attention_mask=alter_attention_mask, return_dict=True)
                noised_logits = noise_output.logits.transpose(0,1)[alter_input_length-1:-1]
                noised_score = model.compute_transition_scores(sequences=alter_input, scores=[i for i in noised_logits], normalize_logits=True)
                
                repeated_origin_score = origin_score.unsqueeze(0).repeat(alter_input_ids.shape[0],1) # mask

                d = torch.sum((torch.exp(repeated_origin_score) - torch.exp(noised_score)) * repeated_origin_score, dim=-1)
                batch_noised_std.append(torch.std(d))
                batch_noised_diff.append(torch.mean(abs(d-torch.mean(d))))
                
            mean_standard_deviation = torch.stack(batch_noised_std).mean(dim=0)
            mean_MAD = torch.stack(batch_noised_diff).mean(dim=0)
            category_mean_standard_deviation.append(mean_standard_deviation)
            category_mean_MAD.append(mean_MAD)

    category_mean_standard_deviation = torch.stack(category_mean_standard_deviation)
    category_mean_MAD = torch.stack(category_mean_MAD)


    topk_mad = category_mean_MAD.topk(args.topk, largest=False)


    print(f'topk_mad: {topk_mad.indices}')


    result_dict = {'topk_mad':topk_mad.indices.tolist(), 'category_mean_MAD':category_mean_MAD.tolist()}
    return result_dict

def read_data(category, prefix=''):
    alter_data_path = os.path.expanduser('./data/samples')
    with open(os.path.join(alter_data_path, f'{prefix}{category}.json'), 'r') as f:
        sample_datas = json.load(f)
    return sample_datas

def main():
    args = args_parser()
    seed_everything(42)
    prefix = args.sample_prefix + f'altersample_{args.sample}_'
    
    args.tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser('llama-7b'))
    args.tokenizer.pad_token_id = 0
    args.tokenizer.padding_side = 'left'
    category = eval_category_list[args.eval_category_idx]
    eval_samples = read_data(category, prefix=args.sample_prefix)
    
    eval_data = [x['input'] for x in eval_samples]
    alter_datas = [x['alter_input'][:args.sample] for x in eval_samples]
    inputs = args.tokenizer(eval_data, return_tensors="pt", max_length=1024-32, truncation=True, padding=True)
    alters = [args.tokenizer(alter_data, return_tensors="pt", max_length=1024-32, truncation=True, padding=True) for alter_data in alter_datas]
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    alters = [{k: v.to('cuda') for k, v in alter.items()} for alter in alters]

    result_dict = generate_to_select_lora_by_x(inputs,alters, args)
    
    with open(os.path.expanduser(f'./result/xonly_lora_selection_idx/{prefix}{category}.json'), 'w') as f:
        json.dump(result_dict, f)
    print(f'save to ./result/lora_selection_idx/{prefix}{category}.json')



    
main()


# read_data()