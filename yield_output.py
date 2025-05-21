#test changes
from data_loader import eval_category_list, get_category_data, set_random_seed, input_collection, Datasets
from part_lora import get_lambda_merged_model, get_cache, convert_cache, get_selected_lora_cache
import sys
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
from tqdm import tqdm
from functools import partial
import torch
eval_category_list = eval_category_list()
def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='topk', choices=['topk', 'random', 'zeroshot', 'all', 'task_arithmic', 'adamerging', 'average', 'DARE', 'all_lora'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_category_idx', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data_select_random_seed', type=int, default=42)
    parser.add_argument('--toplora_num', type=int, default=10)
    parser.add_argument('--lora_selection_prefix', type=str, default='eos_added_bf16_full_lora_params_')
    parser.add_argument('--ckpt_prefix', type=str, default='eos_added_diverse_10_topk_epoch10_lr0.0005_top10_gradient_accumulation10_')
    
    return parser.parse_args()

args = args_parser()
category = eval_category_list[args.eval_category_idx]

if args.ckpt_path is not None:
    save_path = os.path.expanduser(f'./result/task_output/pre_value_{args.ckpt_path.split("/")[-1].replace(".pt","")}.json')
elif args.mode == 'DARE':
    save_path = os.path.expanduser(f'./result/task_output/DARE_{category}.json')    
elif args.mode == 'average':
    save_path = os.path.expanduser(f'./result/task_output/average_toplora{args.toplora_num}_{args.lora_selection_prefix}{category}.json')
else:
    save_path = os.path.expanduser(f'./result/task_output/{args.ckpt_prefix}{category}.json')

if os.path.exists(save_path):
    print(f'Already exists {category}')
    sys.exit()    

lora_prefix = f'eos_added_'
lora_selection_prefix=args.lora_selection_prefix
if args.mode in ['topk','random','all','adamerging']:
    if args.ckpt_path is not None:
        prefix = os.path.expanduser(args.ckpt_path)
    else:
        prefix = f'{args.ckpt_prefix}'
elif args.mode == 'zeroshot':
    if args.ckpt_path is not None:
        prefix = os.path.expanduser(args.ckpt_path)
    prefix = 'zeroshot_'
elif args.mode == 'task_arithmic':
    prefix = 'task_arithmic_'
elif args.mode == 'average':
    prefix = 'average_'
elif args.mode == 'DARE':
    prefix = 'DARE_'
print(f'Using prefix {prefix}')
print(args)

if args.model_type == 'llama':
    if args.mode == 'DARE':
        model = LlamaForCausalLM.from_pretrained(os.path.expanduser(f'/home/wangzhiqi/model_cache/DARE/llama7b_{category}'))
    else:
        model = LlamaForCausalLM.from_pretrained(os.path.expanduser('~/cachedir/model/llama-7b'))
    tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser('~/dir/model/llama-7b'))
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = 0
set_random_seed(42)
train, eval = get_category_data(category)
if args.mode != 'zeroshot' and args.mode != 'DARE':
    category_cache = get_selected_lora_cache(eval_category=category, model_type=args.model_type, lora_prefix=lora_prefix, indices_prefix=lora_selection_prefix, topN=args.toplora_num)
    model = get_lambda_merged_model(model, cache=category_cache)
    if args.mode == 'task_arithmic':
        ckpt = torch.load(os.path.expanduser(f'./result/ckpt/llama/Task_Arithmetic.pt'))
    elif args.mode == 'average':
        pass
    else:
        if args.ckpt_path is not None:
            ckpt = torch.load(os.path.expanduser(args.ckpt_path))
        else:
            ckpt = torch.load(os.path.expanduser(f'./result/ckpt/{args.model_type}/{prefix}{category}.pt'))
    if args.mode != 'average':
        model.load_state_dict(ckpt, strict=False)

model.bfloat16().to('cuda').eval()
collection = partial(input_collection, tokenizer=tokenizer, n_samples=0, seq_len=1024-32, labels=False)
length = len(eval) if len(eval) < 1000 else 1000
eval = eval[:length]
dataloader = torch.utils.data.DataLoader(Datasets([i['input'] for i in eval]), batch_size=args.batch_size, collate_fn=collection, shuffle=False)

outputs = []
for batch in tqdm(dataloader):
    new_input = {k: v.to('cuda') for k, v in batch.items()}
    ids = model.generate(**new_input, do_sample=False, max_new_tokens=32)
    for i in ids:
        output = tokenizer.decode(i, skip_special_tokens=True)
        outputs.append(output)

outputs_dict = [{'input': eval[i]['input'], 'target': eval[i]['target'], 'output': outputs[i].replace(eval[i]['input'],'')} for i in range(len(eval))]

with open(save_path, 'w') as f:
    json.dump(outputs_dict, f)
print(save_path)

    
        
        
        