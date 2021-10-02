# Bi-Encoder and Poly-Encoder for SIMMC 2.0 Sub-Task 2: Multimodal Coreference Resolution

- The codes are adapted from https://github.com/chijames/Poly-Encoder.


## Requirements

- Python 3.8

- Please see requirements.txt.


## Bert Model Setup

1. Download [BERT model](https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip) from Google.

2. Pick the model you like (I am using uncased_L-4_H-512_A-8.zip) and move it into bert_model/ then unzip it.

3. cd bert_model/ and change bert_config.json to config.json.

4. bash run.sh


## DSTC 10 Track 3 Data

1. Download the data from the official challenge [website](https://github.com/facebookresearch/simmc2) and put all the files in [data](https://github.com/facebookresearch/simmc2/tree/master/data) into simmc2_data and unzip all the zip files.

2. Put subtask_134/output_134/dstc10-simmc-devtest-pred-subtask-1.json from Sub-Task 1 into model_data.


## Preprocess

1. Preprocess the training data:

   ```shell
   python preprocess_input.py --input_path_json simmc2_data/simmc2_dials_dstc10_train.json --output_path model_data/train.txt --mode train
   ```

2. Preprocess the validation data:

   ```shell
   python preprocess_input.py --input_path_json simmc2_data/simmc2_dials_dstc10_dev.json --output_path model_data/dev.txt --mode dev
   ```

3. Preprocess the test data:

   ```shell
   python preprocess_input.py --input_path_json simmc2_data/simmc2_dials_dstc10_devtest.json --output_path model_data/devtest.txt --eval_target_path model_data/devtest_subtask2_eval_format.json
   ```


## Training

1. Train a **Bi-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir bi-encoder/ --train_dir model_data/ --max_contexts_length 64 --max_response_length 256 --use_pretrain --architecture bi
   ```

2. Train a **Poly-Encoder** with 16 codes:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir poly-encoder/ --train_dir model_data/ --max_contexts_length 64 --max_response_length 256 --use_pretrain --architecture poly --poly_m 16
   ```


## Inference

1. Test on **Bi-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir bi-encoder/ --train_dir model_data/ --max_contexts_length 64 --max_response_length 256 --use_pretrain --architecture bi --eval

   python inference.py --target_format_json model_data/devtest_subtask2_eval_format.json --dot_product_path bi-encoder/dot_product_devtest.txt --subtask1_result model_data/subtask1_disambigator_submission_format.json --predict_path bi-encoder_predict/bi_devtest_predict.json --target_path bi-encoder_predict/bi_devtest_target.json --top_k 2
   ```

2. Test on **Poly-Encoder** with 16 codes:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir poly-encoder/ --train_dir model_data/ --max_contexts_length 64 --max_response_length 256 --use_pretrain --architecture poly --poly_m 16 --eval

   python inference.py --target_format_json model_data/devtest_subtask2_eval_format.json --dot_product_path poly-encoder/dot_product_devtest.txt --subtask1_result model_data/subtask1_disambigator_submission_format.json --predict_path poly-encoder_predict/poly_devtest_predict.json --target_path poly-encoder_predict/poly_devtest_target.json --top_k 2
   ```


## Evaluation

1. Evaluation on **Bi-Encoder**:

   ```shell
   python -m utils.evaluate_dst --input_path_target=bi-encoder_predict/bi_devtest_target.json --input_path_predicted=bi-encoder_predict/bi_devtest_predict.json --output_path_report=bi-encoder_predict/simmc2_dials_dstc10_report.json
   ```

2. Evaluation on **Poly-Encoder**:

   ```shell
   python -m utils.evaluate_dst --input_path_target=poly-encoder_predict/poly_devtest_target.json --input_path_predicted=poly-encoder_predict/poly_devtest_predict.json --output_path_report=poly-encoder_predict/simmc2_dials_dstc10_report.json
   ```


## Results

- Results of Sub-Task 2 are saved at bi-encoder_predict/simmc2_dials_dstc10_report.json and poly-encoder_predict/simmc2_dials_dstc10_report.json. Please refer to object_f1.

- Default parameters in run.py are used, please refer to run.py for details.


## Test-Std

1. Preprocess the Test-Std data (if only predict the object ids mentioned in the last turn):

   ```shell
   python preprocess_teststd.py --input_path_json simmc2_data/simmc2_dials_dstc10_teststd_public.json --output_path model_data/teststd.txt --eval_target_path model_data/teststd_subtask2_eval_format.json
   ```
   
   Preprocess the Test-Std data (if predict the object ids mentioned in all the turns):
   
   ```shell
   python preprocess_input.py --input_path_json simmc2_data/simmc2_dials_dstc10_teststd_public.json --output_path model_data/teststd.txt --eval_target_path model_data/teststd_subtask2_eval_format.json
   ```

2. Inference the Test-Std data using **Bi-Encoder**:
  
   ```shell
   python3 run.py --bert_model bert_model/ --output_dir bi-encoder/ --train_dir model_data/ --test_fname teststd.txt --dot_product_outfname dot_product_teststd.txt --max_contexts_length 64 --max_response_length 256 --use_pretrain --architecture bi --eval
   
   python inference.py --target_format_json model_data/teststd_subtask2_eval_format.json --dot_product_path bi-encoder/dot_product_teststd.txt --subtask1_result model_data/subtask1_disambigator_submission_format.json --predict_path bi-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json --top_k 2
   ```

3. Inference the Test-Std data using **Poly-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir poly-encoder/ --train_dir model_data/ --test_fname teststd.txt --dot_product_outfname dot_product_teststd.txt --max_contexts_length 64 --max_response_length 256 --use_pretrain --architecture poly --poly_m 16 --eval

   python inference.py --target_format_json model_data/teststd_subtask2_eval_format.json --dot_product_path poly-encoder/dot_product_teststd.txt --subtask1_result model_data/subtask1_disambigator_submission_format.json --predict_path poly-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json --top_k 2
   ```

4. Evaluation the Test-Std data on **Bi-Encoder** and **Poly-Encoder**:
   
   Please use python -m utils.evaluate_dst 

5. For the results of Sub-Task 2 on the Test-Std data, please refer to object_f1.
