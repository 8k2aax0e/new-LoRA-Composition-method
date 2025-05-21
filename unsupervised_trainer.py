from part_lora import get_selected_lora_cache, convert_cache, get_lambda_merged_model
import os
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback

import sys
import torch
from data_loader import eval_category_list, input_collection, set_random_seed, Datasets
from functools import partial
import random

eval_category_list = eval_category_list()


class SaveEachEpochCallback(TrainerCallback):
    def __init__(self, save_path_template):
        self.save_path_template = save_path_template

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        current_save_path = self.save_path_template.format(epoch=epoch)
        model = kwargs['model']
        trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        
        if args.save_every:
            if  os.path.exists(current_save_path):
                print(f"Already exists {current_save_path}")
            else:     
                torch.save(trainable_params, current_save_path)
                print(f"Saved parameters at epoch {epoch} to {current_save_path}")
        else:
            if epoch == args.epochs:
                if  os.path.exists(current_save_path):
                    print(f"Already exists {current_save_path}")
                else:     
                    torch.save(trainable_params, current_save_path)
                    print(f"Saved parameters at epoch {epoch} to {current_save_path}")
            
        

def compute_shannon_entropy(output_logits):
    ln_probs = torch.nn.functional.log_softmax(output_logits, dim=-1)
    entropy_per_token = -torch.sum(torch.exp(ln_probs) * ln_probs, dim=-1).mean()
    return entropy_per_token

class AdaMergingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = compute_shannon_entropy
        loss = loss_fct(logits.view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_category_idx', type=int, default=0)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--mode', type=str, default='topk', choices=['topk', 'all', 'random', 'adamerging'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=10)
    parser.add_argument('--data_select_random_seed', type=int, default=42)
    parser.add_argument('--lora_selection_prefix', type=str, default='')
    parser.add_argument('--sample_selection_prefix', type=str, default='')
    parser.add_argument('--indices_type', type=str, default='')
    parser.add_argument('--toplora_num', type=int, default=10)
    parser.add_argument('--save_every', action='store_true')
    return parser.parse_args()
def train_a_model(model,dataset,args):
    save_callback = SaveEachEpochCallback(args.save_path)
    training_args = TrainingArguments(
        do_train=True,                    
        num_train_epochs=args.epoch,              
        per_device_train_batch_size=args.batch_size,  
        gradient_accumulation_steps=args.gradient_accumulation,
        do_eval=False,                   
        save_strategy='no',
        warmup_steps=0,
        output_dir='.',
        lr_scheduler_type='constant',
        logging_steps=1000
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    labels = True if args.mode != 'adamerging' else False
    partial_collect_batch = partial(input_collection, seq_len=1024, tokenizer=args.tokenizer, labels=labels)
    
    training_data = Datasets(dataset)
    
    mytrainer = AdaMergingTrainer if args.mode == 'adamerging' else Trainer
    
    trainer = mytrainer(
        model=model,
        optimizers=(optimizer,None),
        args=training_args,
        train_dataset=training_data,
        data_collator=partial_collect_batch,
        callbacks=[save_callback]
    )
    trainer.train()
    model.eval()

    return model

args = arg_parser()
set_random_seed(42)
lora_prefix = 'eos_added_'
tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser('llama-7b'))
tokenizer.pad_token_id = 0
args.tokenizer = tokenizer
if args.mode == 'topk':
    save_path = os.path.expanduser(f'./result/ckpt/llama/{args.sample_selection_prefix}{args.mode}_epoch{{epoch}}_lr{args.lr}_top{args.topk}_gradient_accumulation{args.gradient_accumulation * args.batch_size}_{eval_category_list[args.eval_category_idx]}.pt')
elif args.mode == 'random':
    save_path = os.path.expanduser(f'./result/ckpt/llama/{args.sample_selection_prefix}{args.mode}_dataSseed{args.data_select_random_seed}_epoch{{epoch}}_lr{args.lr}_top{args.topk}_gradient_accumulation{args.gradient_accumulation * args.batch_size}_{eval_category_list[args.eval_category_idx]}.pt')
elif args.mode == 'adamerging':
    save_path = os.path.expanduser(f'./result/ckpt/llama/eos_added_LoraTopN{args.toplora_num}_{args.lora_selection_prefix.replace(".json","_")}{args.mode}_epoch{{epoch}}_lr{args.lr}_top{args.topk}_gradient_accumulation{args.gradient_accumulation * args.batch_size}_{eval_category_list[args.eval_category_idx]}.pt')
else:
    save_path = os.path.expanduser(f'./result/ckpt/llama/{args.sample_selection_prefix}{args.mode}_epoch{{epoch}}_lr{args.lr}_top{args.topk}_gradient_accumulation{args.gradient_accumulation * args.batch_size}_{eval_category_list[args.eval_category_idx]}.pt')

if all([os.path.exists(save_path.format(epoch=i + 1)) for i in range(args.epoch)]):
    print(f'Already exists {save_path}')
    sys.exit()

args.save_path = save_path
path = os.path.expanduser(f'./result/score_result/{args.sample_selection_prefix}{eval_category_list[args.eval_category_idx]}.pt')

json_data = torch.load(path)

if args.mode in ['topk', 'random','all']:

    topk = args.topk
    training_tokens = []
    random_tokens = []
    for i in range(len(json_data)):
        current_input = json_data[i]['input']
        loraoutput_scores = []
        for j in range(len(json_data[i]['scores'])):
            for k in range(len(json_data[i]['scores'][j]['output'])):
                beam_score = sum(json_data[i]['scores'][j]['others_scores'][l][k] for l in range(len(json_data[i]['scores'][j]['others_scores'])))/len(json_data[i]['scores'][j]['others_scores'])
                output = json_data[i]['scores'][j]['output'][k]
                output_text = json_data[i]['scores'][j]['output_texts'][k]
                loraoutput_scores.append((beam_score, output_text, json_data[i]['scores'][j]['curr_lora_category']))

        if args.mode == 'topk':
            selected_output = sorted(loraoutput_scores, key=lambda x: x[0], reverse=True)[:topk]
        elif args.mode == 'random':
            random.seed(args.data_select_random_seed)
            selected_output = random.choices(loraoutput_scores, k=topk)
        else:
            selected_output = loraoutput_scores
        for j in range(len(selected_output)):
            training_tokens.append(selected_output[j][1])
        # training_data.append((selected_output, current_input))
elif args.mode == 'adamerging':
    tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser('llama-7b'))
    training_tokens = [tokenizer.decode(x['input'], skip_special_tokens=True) for x in json_data]

cache = get_selected_lora_cache(lora_prefix=lora_prefix, indices_prefix=args.lora_selection_prefix, eval_category=eval_category_list[args.eval_category_idx], indices_type=args.indices_type, topN=args.toplora_num)
model = LlamaForCausalLM.from_pretrained(os.path.expanduser('llama-7b'))
model = get_lambda_merged_model(model, cache)
model.bfloat16()
model = train_a_model(model, training_tokens, args)


