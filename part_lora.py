import torch
from torch import nn
import math
from data_loader import category_list
import os
import json
from tqdm import tqdm
category_list = category_list()


class param_weight(nn.Module):
    def __init__(self, lora_caches):
        super(param_weight, self).__init__()
        self.loRAs_weight = nn.Parameter(torch.empty(len(lora_caches)))
        nn.init.constant_(self.loRAs_weight, 1/len(lora_caches))
    def forward(self):
        return self.loRAs_weight
        
class replaced_lora_linear(nn.Module):
    def __init__(self, origin_linear, use_index:list, rank=16, alpha=32):
        super(replaced_lora_linear, self).__init__()

        self.lora_A = nn.Linear(origin_linear.in_features, rank, bias=False)
        nn.init.kaiming_normal_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_active_B = nn.Linear(rank, len(use_index), bias=False)
        nn.init.zeros_(self.lora_active_B.weight)
        self.dropout = nn.Dropout(0.01)
        self.lora_alpha = alpha
        self.lora_use_index = use_index
        self.linear = origin_linear
        self.scale = alpha/rank
        
    def forward(self, x):

        lora_output = torch.zeros(x.shape[0], x.shape[1], self.linear.out_features, dtype=x.dtype, device=x.device)
        lora_output_A = self.scale * self.lora_active_B(self.lora_A(self.dropout(x)))
        lora_output[:,:,self.lora_use_index] = lora_output_A
        
        return self.linear(x) + lora_output

class replaced_lora_linear_complete(nn.Module):
    def __init__(self, origin_linear, rank=16, alpha=32):
        super(replaced_lora_linear_complete, self).__init__()
        self.lora_A = nn.Linear(origin_linear.in_features, rank, bias=False)
        nn.init.kaiming_normal_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_B = nn.Linear(rank, origin_linear.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(0.01)
        self.lora_alpha = alpha
        self.linear = origin_linear
        self.scale = alpha/rank
        
    def forward(self, x):
        lora_output = self.scale * self.lora_B(self.lora_A(self.dropout(x)))
        return self.linear(x) + lora_output

class replaced_compose_lora(nn.Module):
    def __init__(self, origin_linear, lora_caches={}, rank=16, alpha=32):
        lora_caches = list(lora_caches.values())
        super(replaced_compose_lora, self).__init__()
        self.lora_As = torch.nn.ModuleList([nn.Linear(origin_linear.in_features, rank, bias=False) for _ in range(len(lora_caches))]) 
        for num, _ in enumerate(self.lora_As):
            self.lora_As[num].weight = lora_caches[num]['lora_A.weight']
        self.lora_Bs = torch.nn.ModuleList([nn.Linear(rank, origin_linear.out_features, bias=False) for _ in range(len(lora_caches))])
        for num, _ in enumerate(self.lora_Bs):
            self.lora_Bs[num].weight = lora_caches[num]['lora_B.weight']
        self.dropout = nn.Dropout(0.01)
        self.lora_alpha = alpha
        self.linear = origin_linear
        self.scale = alpha/rank
        
        self.lora_caches = lora_caches
        self.loRAs_weight = param_weight(lora_caches)
        
    def forward(self, x):
        loras_output = []
        for num, _ in enumerate(self.lora_caches):
            lora_output = self.scale * self.lora_Bs[num](self.lora_As[num](self.dropout(x)))
            loras_output.append(lora_output)
        loras_output = torch.stack(loras_output, dim=0)
        loras_output = torch.sum(loras_output * self.loRAs_weight().view(len(self.lora_caches),1,1,1), dim=0)
        
        return self.linear(x) + loras_output

def get_cache(category=None, prefix='bf16_full_lora_params_', streaming=False, streaming_num=0, model_type='llama',device='cpu'):
    cache = {}
    if not streaming:
        if category is None:
            for category in category_list:        
                cache[category] = torch.load(os.path.expanduser(f'./resource/lora_liberary/{model_type}/{prefix}{category}.pt'), map_location=device, weights_only=True)
        elif type(category) == list:
            for category in category:        
                cache[category] = torch.load(os.path.expanduser(f'./resource/lora_liberary/{model_type}/{prefix}{category}.pt'), map_location=device, weights_only=True)            
        else:
            cache = torch.load(os.path.expanduser(f'./resource/lora_liberary/{model_type}/{prefix}{category}.pt'))
        return cache
    else:
        category = category_list[streaming_num]       
        cache = torch.load(os.path.expanduser(f'./resource/lora_liberary/{model_type}/{prefix}{category}.pt'), map_location=device)
        return cache

def get_selected_lora_cache(topN=0,lora_prefix='bf16_full_lora_params_',indices_prefix='bf16_full_lora_params_',eval_category='fill_in_the_blank',model_type='llama',device='cpu',indices_type='old'):
    
    print(f"using lora index {os.path.expanduser(f'./result/lora_selection_idx/{indices_prefix}{eval_category}_eval_indices.json')}")
 

    with open(os.path.expanduser(f'./result/xonly_lora_selection_idx/{indices_prefix}{eval_category}.json'), 'r') as f:
        data = json.load(f)
    if topN == 0:
        data['topk_mad'].sort()
        indices = data['topk_mad']
    else:
        len_topN = topN if topN < len(data['topk_mad']) else len(data['topk_mad'])
        indices = data['topk_mad'][:len_topN]
    categorys = [category_list[i] for i in indices]
    caches = get_cache(prefix=lora_prefix,category=categorys, model_type=model_type, device=device)
    return caches

def set_nested_attribute(obj, attr_string, value):

    attributes = attr_string.split('.')
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)

def get_nested_attribute(obj, attr_string):

    attributes = attr_string.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj

def convert_cache(cache):
    converted_cache = {}
    for lora_idx,category in enumerate(cache):
        for layer in cache[category]:
            origin_layer = layer[:-14]
            AorB = layer[-13:]
            
            idx = category_list.index(category)
            
            if origin_layer not in converted_cache:
                converted_cache[origin_layer] = {}
                converted_cache[origin_layer][idx] = {AorB: cache[category][layer]}
            else:
                if idx not in converted_cache[origin_layer]:
                    converted_cache[origin_layer][idx] = {AorB: cache[category][layer]}
                else:
                    converted_cache[origin_layer][idx][AorB] = cache[category][layer]
    return converted_cache

def get_lambda_merged_model(model, cache):
    cache = convert_cache(cache)
    for name in cache:
        module = get_nested_attribute(model, name)
        layer = replaced_compose_lora(module, cache[name])
        set_nested_attribute(model, name, layer)
    for name,param in model.named_parameters():
        if 'loRAs_weight' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def noise_cache(cache:dict, epoch=3, device='cpu'):
    for random_seed in range(epoch):
        noise_cache = {}
        noise_matrix = {}
        torch.manual_seed(random_seed)
        lora_category = list(cache.keys())
        layer_name = list(cache[lora_category[0]].keys())
        for name in layer_name:
            noise_matrix[name] = torch.normal(mean=0, std=0.01, size=cache[lora_category[0]][name].size()).to(device)
        for category in lora_category:
            noise_cache[category] = {}
            for name in layer_name:
                noise_cache[category][name] = cache[category][name] + noise_matrix[name]
        yield noise_cache


def get_empty_full_model(model, moduleNames=['q_proj', 'k_proj', 'v_proj', 'o_proj']):
    modules_list = get_module(model, moduleNames)
    for module_name in modules_list:
        module = get_nested_attribute(model, module_name)
        layer = replaced_lora_linear_complete(module)
        set_nested_attribute(model, module_name, layer)
    return model

def get_merge_full_model(model, category_params):
    moduleNames_idx = get_cache_module(model, category_params)
    for name in moduleNames_idx:
        module = get_nested_attribute(model, name)
        layer = replaced_lora_linear_complete(module)
        set_nested_attribute(model, name, layer)
    model.load_state_dict(category_params, strict=False)
    return model

def get_module(model, module_keys):
    module_names = [name for name, _ in model.named_modules()]
    filted_model_name = [name.replace('.weight','') for name in module_names if any(key in name for key in module_keys)]
    return filted_model_name

def get_cache_module(model, module_dicts):
    module_names = [name for name, _ in model.named_modules()]
    filted_model_name = [name.replace('.weight','') for name in module_names if any(key in name for key in module_dicts)]
    return filted_model_name
