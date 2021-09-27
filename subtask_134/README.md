# Overview
This is the code for Sub-Task #1, Sub-Task #3 , Sub-Task #4

# Installation
Our code works in python 3.7
`pip install -r requirements.txt`

# Prepare Data:
Copy all data from https://github.com/facebookresearch/simmc2/tree/master/data to ./data/original/
and merge 2 folders `simmc2_scene_images_dstc10_public_part1` and `simmc2_scene_images_dstc10_public_part2` into `simmc2_scene_images_dstc10_public`


For 2 models, we use 2 process files.
```
PREPROCESS_SPLIT=train bash ./preprocess_data.sh
PREPROCESS_SPLIT=dev bash ./preprocess_data.sh
PREPROCESS_SPLIT=devtest bash ./preprocess_data.sh

PREPROCESS_SPLIT=train bash ./preprocess_data_134.sh
PREPROCESS_SPLIT=dev bash ./preprocess_data_134.sh
PREPROCESS_SPLIT=devtest bash ./preprocess_data_134.sh
```
After running above scripts, we would have two folders: 
```
data/preprocess/response_only
data/preprocess/output_1_3_4_input_response_meta
```

To train our models, we create a new folder with the input is only of context (predict files from `data/preprocess/response_only`) and output is the combined string of Sub-Task 1,3,4 (target files from `data/preprocess/output_1_3_4_input_response_meta`).
That results the following folder

```
mkdir data/preprocess/output_1_3_4_input_response_only
cp data/preprocess/output_1_3_4_input_response_meta/*_target.txt data/preprocess/output_1_3_4_input_response_only 
cp data/preprocess/response_only/*_predict.txt data/preprocess/output_1_3_4_input_response_only
```

For retrieval task, we run the following code
```
bash ./run_preprocess_gpt2_retrieval_separate_predict_target.sh
```
# Train and Inference:
For the multitask model, we run as followings:
```
export CUDA_VISIBLE_DEVICES=0
export SIMMC_EXP_MODE=OUT_134_IN_RES
export SIMMC_DATA_DIR=data/preprocess
export SIMMC_SAVE_DIR=model
export BATCH_SIZE=24
export EXP_MODE="train,generation"
bash ./train_bart_multiple_settings.sh
```

For Sub-Task 4 retrieval, we run as followings:

# Train:
```
export RESPONSE_ONLY_DATA_DIR=data/preprocess/response_only
export RESPONSE_ONLY_SAVE_DIR=model/output_4_input_response_only
python train_bart.py --data_dir $RESPONSE_ONLY_DATA_DIR \
--output_dir $RESPONSE_ONLY_SAVE_DIR
```

# Generate Output for Retrieval
```
export RETRIEVAL_DATA_DIR=data/preprocess/retrieval_data
export RETRIEVAL_OUTPUT_DIR=$model/output_4_input_response_only/retrieval
python generate_bart_retrieval_score.py --data_dir $RETRIEVAL_DATA_DIR \
--output_dir $RETRIEVAL_OUTPUT_DIR \
--model_name_or_path ${RESPONSE_ONLY_SAVE_DIR}/
```
The above scripts train and generate the outputs of devtest set in flat text file.
Here we put it in folder `output_134` for further evaluation.

Make sure you have two prediction files `predictions.txt` and `devtest_retrieval_scores.txt`.

Besides, we also have files to help to convert them to submission format `simmc2_dials_dstc10_devtest_dialog_turn_id.json`, which is created
from preprocessing for retrieval task above.

For the ground true files, we need to include `simmc2_dials_dstc10_devtest.json` and `simmc2_dials_dstc10_devtest_retrieval_candidates.json`

# Convert Output to Submission Format
**For subtask 1**
```
python generate_submission_format_disambiguator_134.py \ 
--dialog_turn_id_json_path output_134/simmc2_dials_dstc10_devtest_dialog_turn_id.json \ 
--generated_text_path output_134/predictions.txt \
--output_submission_format_path output_134/dstc10-simmc-devtest-pred-subtask-1.json
```

**For subtask 3**

```
python generate_submission_format_belief_state_134.py \
--dialog_turn_id_json_path output_134/simmc2_dials_dstc10_devtest_dialog_turn_id.json \
--generated_text_path output_134/predictions.txt \
--output_submission_format_path output_134/dstc10-simmc-devtest-pred-subtask-3.json
```

**For subtask 4 (Generation)**

```
python generate_submission_format_generation_134.py \
--dialog_turn_id_json_path output_134/simmc2_dials_dstc10_devtest_dialog_turn_id.json \
--generated_text_path output_134/predictions.txt \
--output_submission_format_path output_134/dstc10-simmc-devtest-pred-subtask-4-generation.json
```

**For subtask 4 (Retrieval)**

```
python generate_submission_format_retrieval.py \
--dialog_turn_id_json_path output_134/simmc2_dials_dstc10_devtest_dialog_turn_id.json \
--model_flat_score_path output_134/devtest_retrieval_scores.txt \
--output_submission_format_path output_134/dstc10-simmc-devtest-pred-subtask-4-retrieval.json
```
# Evaluation:
**For subtask 1**
```
python utils/disambiguator_evaluation.py \
--data_json_path output_134/simmc2_dials_dstc10_devtest.json \
--model_result_path output_134/dstc10-simmc-devtest-pred-subtask-1.json
```

**For subtask 3**
```
python utils/evaluate_dst.py \
--input_path_target output_134/simmc2_dials_dstc10_devtest.json \
--input_path_predicted output_134/dstc10-simmc-devtest-pred-subtask-3.json
```

**For subtask 4 (Generation)**

```
 python utils/response_evaluation.py \
 --data_json_path output_134/simmc2_dials_dstc10_devtest.json \
 --model_response_path output_134/dstc10-simmc-devtest-pred-subtask-4-generation.json \
 --single_round_evaluation
```

**For subtask 4 (Retrieval)**

```
python utils/retrieval_evaluation.py \
--retrieval_json_path output_134/simmc2_dials_dstc10_devtest_retrieval_candidates.json \
--model_score_path output_134/dstc10-simmc-devtest-pred-subtask-4-retrieval.json \
--single_round_evaluation
```
Currently, there are some issues with official evaluation scripts (need to verify)
https://github.com/facebookresearch/simmc2/issues/40