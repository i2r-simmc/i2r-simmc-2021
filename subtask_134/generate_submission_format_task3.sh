#!/usr/bin/env bash
python generate_submission_format_belief_state_134_teststd.py \
--dialog_turn_id_json_path output_134/teststd/simmc2_dials_dstc10_teststd_dialog_turn_id.json \
--generated_text_path output_134/teststd/predictions.txt \
--output_submission_format_path output_134/teststd/dstc10-simmc-teststd-pred-subtask-3.json
