#!/usr/bin/env bash
python generate_submission_format_retrieval_teststd.py \
--dialog_turn_id_json_path output_134/teststd/simmc2_dials_dstc10_teststd_dialog_turn_id.json \
--model_flat_score_path output_134/teststd/retrieval_scores.txt \
--output_submission_format_path output_134/teststd/dstc10-simmc-teststd-pred-subtask-4-retrieval.json