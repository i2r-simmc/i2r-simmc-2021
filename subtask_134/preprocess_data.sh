#!/usr/bin/env bash
#!/usr/bin/env bash
PREPROCESS_SPLIT="${PREPROCESS_SPLIT:-train}"
export DATA_DIR=data/original/
export OUTPUT_DIR="./data"

export Input_Dialogue="${DATA_DIR}/simmc2_dials_dstc10_${PREPROCESS_SPLIT}.json"
export Output_Predict_Path="${OUTPUT_DIR}/preprocess/response_only/${PREPROCESS_SPLIT}_predict.txt"
export Output_Target_Path="${OUTPUT_DIR}/preprocess/response_only/${PREPROCESS_SPLIT}_target.txt"

python preprocess_input.py --input_path_json $Input_Dialogue \
--output_path_predict $Output_Predict_Path \
--output_path_target $Output_Target_Path \
--no_belief_states