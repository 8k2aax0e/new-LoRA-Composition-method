from transformers import LlamaTokenizer
from datasets import load_dataset
import os
import sys
from transformers import LlamaForCausalLM, LlamaTokenizer
from part_lora import get_empty_full_model, get_cache, get_selected_lora_cache
from data_loader import get_category_data, eval_category_list
from collections import OrderedDict
import torch
import numpy as np
import random
from tqdm import tqdm
import json
eval_category_list = eval_category_list()
lora_prefix = 'eos_added_'
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
    parser.add_argument('--example_num', type=int, default=10)
    parser.add_argument('--in_length', type=int, default=1024)
    parser.add_argument('--llama_path', type=str, default='llama-7b')
    parser.add_argument('--lora_selection_prefix', type=str, default='')
    parser.add_argument('--indices_type', type=str, default='')
    parser.add_argument('--toplora_num', type=int, default=10)
    
    return parser.parse_args()

def generate_message(test_dataset, args):
    print('now start generating')
    model = LlamaForCausalLM.from_pretrained(os.path.expanduser(args.llama_path))
    model = get_empty_full_model(model)
    model.bfloat16().to('cuda').eval()
    caches = get_selected_lora_cache(lora_prefix=lora_prefix, eval_category=eval_category_list[args.eval_category_idx], indices_prefix=args.lora_selection_prefix, indices_type=args.indices_type, topN=args.toplora_num)
    exist_category_list = list(caches.keys())
    inputs = test_dataset
    beams_num = 10
    inputs_scores = []
    with torch.no_grad():
        for origin_input in tqdm(inputs):
            output_scores = []
            for current_lora_category in exist_category_list:
                model.load_state_dict(caches[current_lora_category], strict=False)
                model.bfloat16()
                input_length = origin_input.shape[-1]
                
                # output = model.generate(origin_input, max_new_tokens=32, do_sample=False, output_scores=True, output_logits=True, return_dict_in_generate=True,num_beams=beams_num,num_beam_groups=beams_num, num_return_sequences=beams_num, repetition_penalty=2.0, diversity_penalty=0.5)
                output = model.generate(origin_input, max_new_tokens=32, do_sample=True, output_scores=True, output_logits=True, return_dict_in_generate=True, num_return_sequences=beams_num, repetition_penalty=2.0, temperature=1)

                attention_masks = torch.stack([(i != args.tokenizer.pad_token_id).int() for i in output['sequences']]).to(torch.int64)
                output_mask = attention_masks[:,input_length-1:-1]
                output_sequences = output['sequences'][:, input_length:]
                for i in output_sequences:
                    end_idx = torch.where(i==2)
                    if end_idx[0].shape[0] == 0:
                        output_text = args.tokenizer.decode(i[:])
                    else:
                        output_text = args.tokenizer.decode(i[:end_idx[0][0]])
                output_texts = []
                for i in output['sequences']:
                    end_idx = torch.where(i==2)
                    if end_idx[0].shape[0] == 0:
                        output_text = args.tokenizer.decode(i[:], skip_special_tokens=True)
                    else:
                        output_text = args.tokenizer.decode(i[:end_idx[0][0]], skip_special_tokens=True)
                    output_texts.append(output_text)
                main_scores = []
                others_scores = []
                
                for other_lora_category in exist_category_list:
                    if current_lora_category == other_lora_category:
                        continue
                    model.load_state_dict(caches[other_lora_category], strict=False)
                    model.bfloat16()

                    new_outputs = model(input_ids=output['sequences'], attention_mask=attention_masks)
                    new_logits = new_outputs.logits.transpose(0,1)[input_length-1:-1]
    
                    other_scores = model.compute_transition_scores(sequences=output['sequences'], scores=[i for i in new_logits], normalize_logits=True)
                    
                    others_mean_scores = []
                    for i in range(len(output_sequences)):
                        normal_score = (torch.exp(other_scores[i]) * output_mask[i]).sum() / output_mask[i].sum()
                        others_mean_scores.append(normal_score)
                    others_scores.append(others_mean_scores)
                
                
                output_scores.append({'curr_lora_category':current_lora_category, 'output': output['sequences'].tolist(), 'main_scores':main_scores, 'others_scores':others_scores,'output_texts':output_texts})
            inputs_scores.append( {'input': origin_input.squeeze().tolist(), 'scores': output_scores})
    return inputs_scores


def main():
    args = args_parser()
    category = eval_category_list[args.eval_category_idx]
    
    save_path = os.path.expanduser(f'./result/score_result/{lora_prefix}LoraTopN{args.toplora_num}_{args.lora_selection_prefix.replace(".json","_")}sample{args.example_num}_{category}.pt')
    if os.path.exists(save_path):
        print(f'Already exists {save_path}')
        sys.exit()
    
    seed_everything(42)
    args.tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser('llama-7b'))
    args.tokenizer.pad_token_id = 0
    args.tokenizer.padding_side = 'left'


    eval_root_path = os.path.expanduser('./data/evals')
    eval_path = os.path.join(eval_root_path, f'{category}.json')
    if not os.path.exists(eval_path):
        print(f'No such file {eval_path}')
        sys.exit()
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
        
    eval_data = [x['input'] for x in eval_data[:args.example_num]]
    example_inputs = []
    for d in eval_data:
        input_ids = args.tokenizer(d, return_tensors="pt", max_length=args.in_length, truncation=True)['input_ids'].to('cuda')
        example_inputs.append(input_ids)

    results = generate_message(example_inputs, args)

    torch.save(results, save_path)
    print(f'saved to {save_path}')

main()