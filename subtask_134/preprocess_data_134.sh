#!/usr/bin/env bash
PREPROCESS_SPLIT="${PREPROCESS_SPLIT:-train}"
export DATA_DIR=data/original
export OUTPUT_DIR="./data"
export Input_Dialogue="${DATA_DIR}/simmc2_dials_dstc10_${PREPROCESS_SPLIT}.json"
export Scene_Path="${DATA_DIR}/public"
export Meta_Dir="${DATA_DIR}/"
export Output_Predict_Path="${OUTPUT_DIR}/preprocess/output_1_3_4_input_response_meta/${PREPROCESS_SPLIT}_predict.txt"
export Output_Target_Path="${OUTPUT_DIR}/preprocess/output_1_3_4_input_response_meta/${PREPROCESS_SPLIT}_target.txt"

#echo ${Input_Dialogue}

python preprocess_input_res_output_134.py --input_path_json $Input_Dialogue \
--input_meta_json_folder $Meta_Dir \
--input_scene_path_json $Scene_Path \
--output_path_predict $Output_Predict_Path \
--output_path_target $Output_Target_Path

