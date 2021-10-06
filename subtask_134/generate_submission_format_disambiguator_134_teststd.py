import argparse
import json
from copy import deepcopy
import numpy as np
import re
def write_submission_output(dialog_turn_id_data, generation_texts, output_submission_format_path, save_all_output):
    """
    Write the model_scores in
    """
    submission_format_output=[]
    count=0
    for dialog in dialog_turn_id_data:
        _dialog=[]
        for turn_id in range(len(dialog['turn_info'])):
            _turn=dialog['turn_info'][turn_id]
            assert turn_id==_turn['turn_id']
            # _flat_id=_turn['flat_id']
            predicted_text = generation_texts[count]
            count+=1
            if save_all_output=="1":
                _dialog.append({"turn_id":turn_id, "disambiguation_label":predicted_text})
            else:
                if turn_id in dialog["turns_disambiguation_label"]:
                    _dialog.append({"turn_id": turn_id, "disambiguation_label": predicted_text})
        submission_format_output.append({"dialog_id":dialog["dialog_id"],
                                         "predictions":_dialog})

    with open(output_submission_format_path, "w") as f_generation_submission_format:
        json.dump(submission_format_output, f_generation_submission_format)

def main(args):
    print("Reading: {}".format(args["dialog_turn_id_json_path"]))
    with open(args["dialog_turn_id_json_path"], "r") as file_id:
        dialog_turn_id_data = json.load(file_id)
    print("Reading: {}".format(args["generated_text_path"]))
    disambiguation_labels=[]
    with open(args["generated_text_path"], "r") as f_text:
        for ii in f_text.readlines():
            split_line = ii.split("<EOB>", 1)
            to_parse = split_line[0].strip()
            dialog_act_regex = re.compile(
                r'([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\>'
            )
            disambiguation_regex = re.compile(r"([A-Za-z0-9]+)")
            for dialog_act in dialog_act_regex.finditer(to_parse):
                d=[]
                for disambiguation_label in disambiguation_regex.finditer(dialog_act.group(4)):
                    d.append(disambiguation_label.group(1).strip())
            if len(d)>0:
                disambiguation_labels.append(int(d[0]))
            else:
                disambiguation_labels.append("")
    write_submission_output(
        dialog_turn_id_data, disambiguation_labels, args["output_submission_format_path"], args["output_all_turn_id"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Response Retrieval Evaluation")
    parser.add_argument(
        "--dialog_turn_id_json_path",
        default="data/furniture_train_retrieval_candidates.json",
        help="Data with retrieval candidates, gt",
    )
    parser.add_argument(
        "--generated_text_path",
        default=None,
        help="Generation Text that include substasks 1,3,4",
    )
    parser.add_argument(
        "--output_submission_format_path",
        default=None,
        help="generate output_submission_format",
    )
    parser.add_argument(
        "--output_all_turn_id",
        default=1,
        help="output_all_turn_id if = 1",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)