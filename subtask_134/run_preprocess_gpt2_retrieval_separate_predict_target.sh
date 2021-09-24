#!/bin/bash


# Train split
#python3 -m gpt2_dst.scripts.preprocess_input \
#    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_train.json \
#    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_train_predict.txt \
#    --output_path_target="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_train_target.txt \
#    --len_context=2 --no_belief_states\
#    --use_multimodal_contexts=1 \
#    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/simmc2_special_tokens.json
#
## Dev split
#python3 -m gpt2_dst.scripts.preprocess_input_retrieval_separate_predict_target \
#    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_dev.json \
#    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_predict.txt \
#    --output_path_target="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt \
#    --len_context=2 --no_belief_states\
#    --use_multimodal_contexts=1 \
#    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/simmc2_special_tokens.json \
#    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/simmc2_special_tokens.json \
#    --input_path_retrieval="${PATH_DATA_DIR}"/simmc2_dials_dstc10_dev_retrieval_candidates.json \
#    --output_path_retrieval="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_retrieval_candidates.txt \
#    --output_path_src_retrieval="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_src_retrieval_candidates.txt

# Devtest split
export PATH_DIR=data/preprocess/retrieval_data
export PATH_DATA_DIR=data/original/
python3 -m preprocess_input_retrieval_separate_predict_target \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest.json \
    --output_path_predict="${PATH_DIR}"/simmc2_dials_dstc10_devtest_predict.txt \
    --output_path_target="${PATH_DIR}"/simmc2_dials_dstc10_devtest_target.txt \
    --len_context=2 --no_belief_states\
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/simmc2_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/simmc2_special_tokens.json \
    --input_path_retrieval="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest_retrieval_candidates.json \
    --output_path_retrieval="${PATH_DIR}"/simmc2_dials_dstc10_devtest_retrieval_candidates.txt \
    --output_path_src_retrieval="${PATH_DIR}"/simmc2_dials_dstc10_devtest_src_retrieval_candidates.txt \
    --output_path_ground_true="${PATH_DIR}"/simmc2_dials_dstc10_devtest_retrieval_ground_true.json \
    --output_path_dialog_turn_id="${PATH_DIR}"/simmc2_dials_dstc10_devtest_dialog_turn_id.json