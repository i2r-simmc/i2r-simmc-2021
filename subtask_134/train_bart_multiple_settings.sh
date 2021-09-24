set -e
# todo: specify gpus
[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}

SIMMC_EXP_MODE="${SIMMC_EXP_MODE:-foo}"
SIMMC_DATA_DIR="${SIMMC_DATA_DIR:-foo}"
SIMMC_SAVE_DIR="${SIMMC_SAVE_DIR:-foo}"
EXP_MODE="${EXP_MODE:-train}"
PATIENCE_SETTINGS="${PATIENCE_SETTINGS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [ ${SIMMC_EXP_MODE} == "OUT_34_IN_RES" ]; then
  export DATA_DIR=${SIMMC_DATA_DIR}/output_3_4_input_response_only
  export OUTPUT_DIR=${SIMMC_SAVE_DIR}/output_3_4_input_response_only

elif [ ${SIMMC_EXP_MODE} == "OUT_34_IN_RES_META" ]; then
  export DATA_DIR=${SIMMC_DATA_DIR}/output_3_4_input_response_meta
  export OUTPUT_DIR=${SIMMC_SAVE_DIR}/output_3_4_input_response_meta

elif [ ${SIMMC_EXP_MODE} == "OUT_234_IN_RES" ]; then
  export DATA_DIR=${SIMMC_DATA_DIR}/output_2_3_4_input_response_only
  export OUTPUT_DIR=${SIMMC_SAVE_DIR}/output_2_3_4_input_response_only

elif [ ${SIMMC_EXP_MODE} == "OUT_134_IN_RES" ]; then
  export DATA_DIR=${SIMMC_DATA_DIR}/output_1_3_4_input_response_only
  export OUTPUT_DIR=${SIMMC_SAVE_DIR}/output_1_3_4_input_response_only

else
  echo "The Experiment Mode is not supported"

fi
echo $NUM_GPU

if [[ $EXP_MODE == *"train"* ]]; then
  echo "DATA_DIR:${DATA_DIR}"
  echo "OUTPUT_DIR:${OUTPUT_DIR}"
  python train_bart.py --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --patience $PATIENCE_SETTINGS \
  --train_batch_size $BATCH_SIZE
fi

export GENERATION_DATA_DIR=${DATA_DIR}
export GENERATION_MODEL_DIR=${OUTPUT_DIR}/
export GENERATION_OUTPUT_DIR=${GENERATION_MODEL_DIR}/predictions
if [[ $EXP_MODE == *"generation"* ]]; then
  echo "GENERATION_DATA_DIR:${GENERATION_DATA_DIR}"
  echo "GENERATION_MODEL_DIR:${GENERATION_MODEL_DIR}"
  echo "GENERATION_OUTPUT_DIR:${GENERATION_OUTPUT_DIR}"

  python generate_bart.py \
  --data_dir $GENERATION_DATA_DIR \
  --model_name_or_path $GENERATION_MODEL_DIR \
  --output_dir $GENERATION_OUTPUT_DIR
fi

if [[ $EXP_MODE == *"eval"* ]]; then
  echo "GENERATION_DATA_DIR:${GENERATION_DATA_DIR}"
  echo "GENERATION_MODEL_DIR:${GENERATION_MODEL_DIR}"
  echo "GENERATION_OUTPUT_DIR:${GENERATION_OUTPUT_DIR}"

  python ./utils/evaluate_dst_34.py \
  --input_path_target $GENERATION_DATA_DIR/devtest_target.txt \
  --input_path_predicted $GENERATION_OUTPUT_DIR/predictions.txt \
  --output_path_report $GENERATION_OUTPUT_DIR/report_dst.json

  python ./utils/evaluate_response_34.py \
  --input_path_target $GENERATION_DATA_DIR/devtest_target.txt \
  --input_path_predicted $GENERATION_OUTPUT_DIR/predictions.txt \
  --output_path_report $GENERATION_OUTPUT_DIR/report_response.json
fi
#eval "python -m torch.distributed.launch train_bart.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --patience $PATIENCE_SETTINGS"