import argparse
import json
from copy import deepcopy
import numpy as np
def write_submission_output(dialog_turn_id_data, retrieval_scores, output_submission_format_path):
    """
    Write the model_scores in
    """
    submission_format_output=[]
    for dialog in dialog_turn_id_data:
        _dialog=[]
        for turn_id in range(len(dialog['turn_info'])):
            _turn=dialog['turn_info'][turn_id]
            assert turn_id==_turn['turn_id']
            if turn_id == dialog["final_turn_id"]:
                _flat_id=_turn['flat_id']
                start_index = _flat_id[0]
                end_index = _flat_id[1]
                round_scores = retrieval_scores[start_index:end_index]
                _dialog.append({"turn_id":turn_id, "scores":round_scores})
        submission_format_output.append({"dialog_id":dialog["dialog_id"],
                                         "candidate_scores":_dialog})

    with open(output_submission_format_path, "w") as f_retrieval_submission_format:
        json.dump(submission_format_output, f_retrieval_submission_format)

def main(args):
    print("Reading: {}".format(args["dialog_turn_id_json_path"]))
    with open(args["dialog_turn_id_json_path"], "r") as file_id:
        dialog_turn_id_data = json.load(file_id)
    print("Reading: {}".format(args["model_flat_score_path"]))
    with open(args["model_flat_score_path"], "r") as f_score:
        retrieval_scores = f_score.readlines()
        retrieval_scores = [-float(x.strip()) for x in retrieval_scores]
    write_submission_output(
        dialog_turn_id_data, retrieval_scores, args["output_submission_format_path"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Response Retrieval Evaluation")
    parser.add_argument(
        "--dialog_turn_id_json_path",
        default="data/furniture_train_retrieval_candidates.json",
        help="Data with retrieval candidates, gt",
    )
    parser.add_argument(
        "--model_flat_score_path",
        default=None,
        help="Candidate scores generated by the model",
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