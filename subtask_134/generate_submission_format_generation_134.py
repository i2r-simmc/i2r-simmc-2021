import argparse
import json
from copy import deepcopy
import numpy as np
def write_submission_output(dialog_turn_id_data, generation_texts, output_submission_format_path):
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
            _flat_id=_turn['flat_id']
            predicted_text = generation_texts[count]
            count+=1
            _dialog.append({"turn_id":turn_id, "response":predicted_text})
        submission_format_output.append({"dialog_id":dialog["dialog_id"],
                                         "predictions":_dialog})

    with open(output_submission_format_path, "w") as f_generation_submission_format:
        json.dump(submission_format_output, f_generation_submission_format)

def main(args):
    print("Reading: {}".format(args["dialog_turn_id_json_path"]))
    with open(args["dialog_turn_id_json_path"], "r") as file_id:
        dialog_turn_id_data = json.load(file_id)
    print("Reading: {}".format(args["generated_text_path"]))
    generated_text=[]
    with open(args["generated_text_path"], "r") as f_text:
        for ii in f_text.readlines():
            split_line = ii.split("<EOB>", 1)
            generated_text.append(
                split_line[1].strip("\n").strip("<pad>").strip(" <EOS> ")
            )
    write_submission_output(
        dialog_turn_id_data, generated_text, args["output_submission_format_path"]
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
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)