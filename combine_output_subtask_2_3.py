import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_subtask2_path", required=True, type=str, help="The path to the output json file of Sub-Task 2")
    parser.add_argument("--input_subtask3_path", required=True, type=str, help="The path to the output json file of Sub-Task 3")
    parser.add_argument("--output_json_path", default="./outputs/dstc10-simmc-teststd-pred-subtask-3.json", type=str, help="The path to the output json file of Sub-Task 2 and 3")

    args = parser.parse_args()

    with open(args.input_subtask2_path, 'r') as inf:
        subtask2_data = json.load(inf)
    with open(args.input_subtask3_path, 'r') as inf:
        subtask3_data = json.load(inf)

    subtask2_dialogue_data = subtask2_data["dialogue_data"]
    subtask3_dialogue_data = subtask3_data["dialogue_data"]
    print("len of subtask2 is {}, len of subtask3 is {}".format(len(subtask2_dialogue_data), len(subtask3_dialogue_data)))
    subtask3_dicts = {}
    for dialog in subtask3_dialogue_data:
        dialogue_idx = dialog["dialog_id"]
        dialogue = dialog["dialogue"]
        dialogue_dict = {}
        for turn in dialogue:
            turn_idx = turn["turn_id"]
            transcript_annotated = turn["transcript_annotated"]
            dialogue_dict[turn_idx] = transcript_annotated
        subtask3_dicts[dialogue_idx] = dialogue_dict

    subtask23_dialogue_data = []
    for dialog in subtask2_dialogue_data:
        dialogue_idx = dialog["dialogue_idx"]
        dialogue = dialog["dialogue"]
        dialogue_23 = []
        for turn in dialogue:
            turn_idx = turn["turn_idx"]
            transcript_annotated = turn["transcript_annotated"]
            objects = transcript_annotated["act_attributes"]["objects"]
            act = subtask3_dicts[dialogue_idx][turn_idx]["act"]
            slot_values = subtask3_dicts[dialogue_idx][turn_idx]["act_attributes"]["slot_values"]
            request_slots = subtask3_dicts[dialogue_idx][turn_idx]["act_attributes"]["request_slots"]
            dialogue_23.append({
                "transcript_annotated": {
                    "act": act,
                    "act_attributes": {
                        "slot_values": slot_values,
                        "request_slots": request_slots,
                        "objects": objects
                        }
                    },
                "turn_idx": turn_idx
                })
        subtask23_dialogue_data.append({
            "dialogue": dialogue_23,
            "dialogue_idx": dialogue_idx
            })

    # output predicted results of Sub-Task 2 and 3
    with open(args.output_json_path, 'w') as outf:
        json.dump({"dialogue_data": subtask23_dialogue_data}, outf)
