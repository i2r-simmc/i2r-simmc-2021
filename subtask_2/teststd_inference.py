import os
import json
import argparse

from collections import defaultdict

if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--target_format_json", help="input path to the target format of evaluation on devtest json files")
    parser.add_argument(
            "--dot_product_path", help="input path to the dot product between context and candidate objects")
    parser.add_argument(
            "--subtask1_result", help="input path to the results of sub-task 1")
    parser.add_argument(
            "--predict_path", help="output path to the predicted objects of SIMMC2 sub-task 2")
    parser.add_argument(
            "--top_k", type=int, help="the number of returned candidate objects")
    args = parser.parse_args()

    dot_products = []
    with open(args.dot_product_path, "r") as inf:
        for line in inf:
            if not line:
                break
            lines = line.strip().split()
            dot_product = []
            for value in lines:
                dot_product.append(float(value))
            dot_products.append(dot_product)
    targets = []
    with open(args.target_format_json, "r") as inf:
        for line in inf:
            if not line:
                break
            target = json.loads(line.strip())
            targets.append(target)
    target_num = len(targets)
    test_case_num = len(dot_products)

    subtask1_results = defaultdict(lambda : defaultdict(int))
    disambiguation_num = 0
    with open(args.subtask1_result, "r") as inf:
        subtask1_data = json.load(inf)
    for dialog in subtask1_data:
        dialog_id = dialog["dialog_id"]
        for turn in dialog["predictions"]:
            turn_id = turn["turn_id"]
            disambiguation_label = turn["disambiguation_label"]
            if disambiguation_label == 0 or disambiguation_label == 1:
                subtask1_results[dialog_id][turn_id] = disambiguation_label
                disambiguation_num += 1
    
    previous_dialog_id = -1
    subtask2_predict = []
    dialog_predict = []
    
    for i in range(target_num):
        dialog_id = targets[i]["dialogue_idx"]
        if dialog_id != previous_dialog_id:
            if dialog_predict:
                subtask2_predict.append({
                    "dialogue": dialog_predict,
                    "dialogue_idx": previous_dialog_id
                    })
            dialog_predict = []
        turn_id = targets[i]["turn_idx"]
        cand_objs = targets[i]["candidate_objects"]
        cand_obj_num = len(cand_objs)
        
        predicts = {}
        for j in range(cand_obj_num):
            predicts[cand_objs[j]] = dot_products[i][j]
        predict_sorted = sorted(predicts.items(), key=lambda x: x[1], reverse=True)
        predict_objs = []
        if dialog_id in subtask1_results and turn_id in subtask1_results[dialog_id]:
            if len(predicts) > args.top_k:
                return_obj_num = args.top_k
            else:
                return_obj_num = len(predicts)
            for j in range(return_obj_num):
                predict_objs.append(predict_sorted[j][0]) 
        else:
            for j in range(args.top_k):
                predict_objs.append(predict_sorted[j][0])
        dialog_predict.append({
            "transcript_annotated": {
                'act': '',
                'act_attributes': {
                    'slot_values': {},
                    'request_slots': [],
                    'objects': predict_objs
                    }
                },
            "turn_idx": turn_id
            })
        
        previous_dialog_id = dialog_id
        
    if dialog_predict:
        subtask2_predict.append({
            "dialogue": dialog_predict,
            "dialogue_idx": dialog_id
        })

    # output predict results of sub-task 2
    with open(args.predict_path, "w") as outf:
        json.dump({"dialogue_data": subtask2_predict}, outf)
