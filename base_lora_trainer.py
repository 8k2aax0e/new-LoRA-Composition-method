import json
import os
from transformers import LlamaTokenizer, Trainer, TrainingArguments
import sys
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from part_lora import get_empty_full_model
from data_loader import category_list, get_category_data, input_collection, set_random_seed, Datasets
from functools import partial
import torch
import logging
logging.basicConfig(level=logging.INFO)

category_list = category_list()
def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--category_idx', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()

def train_a_model(model,tokenizer,dataset,args):

    
    training_args = TrainingArguments(
        do_train=True,                    
        num_train_epochs=1,              
        per_device_train_batch_size=args.batch_size,  
        gradient_accumulation_steps=16//args.batch_size,
        do_eval=False,                   
        save_strategy='no',
        warmup_steps=0,
        output_dir='.',
        lr_scheduler_type='constant',
        logging_steps=1000
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-4)
    partial_collect_batch_SIN = partial(input_collection, tokenizer=tokenizer, seq_len=1024)
    
    training_data = Datasets(dataset)
    
    trainer = Trainer(
        model=model,
        optimizers=(optimizer,None),
        args=training_args,
        train_dataset=training_data,
        data_collator=partial_collect_batch_SIN
    )
    trainer.train()
    model.eval()

    return model


def main():
    
    args = arg_parser()
    set_random_seed(42)
    category = category_list[args.category_idx]

    pathname = os.path.expanduser(f'./resource/lora_liberary/eos_added_{category}.pt')
    if os.path.exists(pathname):
        logging.info(f'{pathname} exists, skipping...')
        return
    logging.info(f'Category: {category}; index: {args.category_idx}')
    logging.info(f'Loading {args.model_type} model...')

    model = LlamaForCausalLM.from_pretrained(os.path.expanduser('llama-7b'))
    tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser('llama-7b'))
    tokenizer.pad_token_id = 0



    train_data, eval_data = get_category_data(category)
    train_data = train_data[:10000] if len(train_data) > 10000 else train_data
    logging.info(f'Train data: {len(train_data)}')

    model = get_empty_full_model(model)
            
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if name != '' and 'lora' in name:
            param.requires_grad = True

    model = model.bfloat16()

    model = train_a_model(model, tokenizer, train_data, args)
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    torch.save(trainable_params, pathname)

main()