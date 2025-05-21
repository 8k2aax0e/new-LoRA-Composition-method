import json
import math
from rouge import Rouge
from data_loader import category_list, eval_category_list
import numpy as np
eval_category_list = eval_category_list()

def re_yesno(input_text):
    import re
    pattern = r'^.*?(Yes|No|True|False)'
    match = re.search(pattern, input_text, re.IGNORECASE)
    if match:
        first_yesno = match.group(1)
        return first_yesno, True 
    else:
        return input_text, False

def get_rouge_score(output, target):
    
    rouge = Rouge()
    target = target.lower()
    output = output.lower()

    score = rouge.get_scores(output, target)
    return score[0]['rouge-l']['f']
    
def score(path):
    with open(path, 'r') as f:
        s = json.load(f)
    scores = []
    for d in s:
        output = d['output'].replace('\n', ' ')
        target = d['target'].replace('\n', ' ')
        
        if output.replace(' ','').replace('.', '') == '' or target.replace(' ','').replace('.', '') == '':
            scores.append(0)
            continue
        score = get_rouge_score(output, target)
        scores.append(score)

    return sum(scores) / len(scores)
    


def main():
    import torch
    import os
    from data_loader import eval_category_list
    eval_category_list = eval_category_list()

    main_score_summery = []
    for category in eval_category_list:
        path = f'./result/task_output/prefix_{category}.json'
        main_score_summery.append(score(path))
    print(f'score: {round(100 * sum(main_score_summery) / len(main_score_summery), 2)}')
    


    
if __name__ == '__main__':
    main()