"""
This is the evaluation script for the results of enhanced ToG.
"""

import argparse
import json
import re
from typing import List, Union


def read_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error message: {e}")
    return data


def answer_match(reference: Union[str|List[str]], answers: List[str]):
    reference = [reference] if isinstance(reference, str) else reference
    for ref_ans in reference:
        for answer in answers:
            if answer == "":
                return False
            for ans in answer.split(','):
                ans = re.sub(r'\s+', '', ans.strip()).lower()
                ref_ans = re.sub(r'\s+', '', ref_ans.strip()).lower()
                if ans == ref_ans or ans in ref_ans or ref_ans in ans:
                    return True
    return False


def main(args):
    # load data.
    if "cwq" in args.dataset_name:
        question_string = 'question'
    elif "WebQSP" in args.dataset_name:
        question_string = 'question'
    elif "MedQA" in args.dataset_name:
        question_string = 'question'
    else:
        raise(f"unknown dataset {args.dataset_name}.")
    with open(f'./data/{args.dataset_name}.json',encoding='utf-8') as f:
            question_json = json.load(f)
    
    answer_json = read_json_lines(args.answer_path)
    
    num_right = 0
    num_error = 0
    evaluated_questions = set()
    for answer in answer_json:
        # drop duplicated answers in raw ToG code.
        if answer['question'] in evaluated_questions:
            continue
        evaluated_questions.add(answer['question'])
        
        # load reference answers.
        matched_question = [q for q in question_json if q[question_string] == answer['question']]
        if len(matched_question) == 0:
            continue
        reference_answer = matched_question[0]['answer']
        # match results.
        results = answer['results']
        if results is None:
            num_error += 1
        else:
            results = results.split(',')
            if answer_match(reference=reference_answer, answers=results):
                num_right += 1
            else:
                num_error += 1
        
    # save results.
    results_dict = {
        'dataset': args.dataset_name,
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
    with open(f"{args.answer_path.rstrip('.jsonl')}_eval.txt", 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="choose the dataset.")
    parser.add_argument("--answer_path", type=str, help="the answer file name.")
    args = parser.parse_args()
    main(args)