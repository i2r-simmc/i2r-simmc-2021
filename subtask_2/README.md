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
   python3 run.py --bert_model bert_model/ --output_dir bi-encoder/ --train_dir model_data/ --max_contexts_length 128 --max_response_length 256 --use_pretrain --architecture bi
   ```

2. Train a **Poly-Encoder** with 16 codes:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir poly-encoder/ --train_dir model_data/ --max_contexts_length 128 --max_response_length 256 --use_pretrain --architecture poly --poly_m 16
   ```


## Inference

1. Test on **Bi-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir bi-encoder/ --train_dir model_data/ --max_contexts_length 128 --max_response_length 256 --use_pretrain --architecture bi --eval

   python inference.py --target_format_json model_data/devtest_subtask2_eval_format.json --dot_product_path bi-encoder/dot_product_devtest.txt --subtask1_result model_data/dstc10-simmc-devtest-pred-subtask-1.json --predict_path bi-encoder_predict/bi_devtest_predict.json --target_path bi-encoder_predict/bi_devtest_target.json --top_k 2
   ```

2. Test on **Poly-Encoder** with 16 codes:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir poly-encoder/ --train_dir model_data/ --max_contexts_length 128 --max_response_length 256 --use_pretrain --architecture poly --poly_m 16 --eval

   python inference.py --target_format_json model_data/devtest_subtask2_eval_format.json --dot_product_path poly-encoder/dot_product_devtest.txt --subtask1_result model_data/dstc10-simmc-devtest-pred-subtask-1.json --predict_path poly-encoder_predict/poly_devtest_predict.json --target_path poly-encoder_predict/poly_devtest_target.json --top_k 2
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

1. Put subtask_134/output_134/teststd/dstc10-simmc-teststd-pred-subtask-1-full-turn.json from Sub-Task 1 into model_data.

2. Preprocess the Test-Std data:
   
   ```shell
   python preprocess_input.py --input_path_json simmc2_data/simmc2_dials_dstc10_teststd_public.json --scene_folder simmc2_data/simmc2_scene_jsons_dstc10_teststd --output_path model_data/teststd.txt --eval_target_path model_data/teststd_subtask2_eval_format.json
   ```

3. Inference the Test-Std data using **Bi-Encoder**:
  
   ```shell
   python3 run.py --bert_model bert_model/ --output_dir bi-encoder/ --train_dir model_data/ --test_fname teststd.txt --dot_product_outfname dot_product_teststd.txt --max_contexts_length 128 --max_response_length 256 --use_pretrain --architecture bi --eval
   
   python teststd_inference.py --target_format_json model_data/teststd_subtask2_eval_format.json --dot_product_path bi-encoder/dot_product_teststd.txt --subtask1_result model_data/dstc10-simmc-teststd-pred-subtask-1-full-turn.json --predict_path bi-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json --top_k 2
   ```

4. Inference the Test-Std data using **Poly-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir poly-encoder/ --train_dir model_data/ --test_fname teststd.txt --dot_product_outfname dot_product_teststd.txt --max_contexts_length 128 --max_response_length 256 --use_pretrain --architecture poly --poly_m 16 --eval

   python teststd_inference.py --target_format_json model_data/teststd_subtask2_eval_format.json --dot_product_path poly-encoder/dot_product_teststd.txt --subtask1_result model_data/dstc10-simmc-teststd-pred-subtask-1-full-turn.json --predict_path poly-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json --top_k 2
   ```

5. Evaluation the Test-Std data on **Bi-Encoder** and **Poly-Encoder**:
   
   Predicted object ids are saved at bi-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json for Bi-Encoder and poly-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json for Poly-Encoder, respectively. Please use utils.evaluate_dst for the evaluation. Note that ``***.json`` is the json file contains the ground-truth of Test-Std. Please replace it with the real json file containing ground-truth object IDs of Test-Std.
   
   ```shell
   python -m utils.evaluate_dst --input_path_target=***.json --input_path_predicted=bi-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json --output_path_report=bi-encoder_teststd_predict/simmc2_dials_dstc10_report.json
   ```
   
   Please note that since dstc10-simmc-teststd-pred-subtask-2.json only contains the results of Sub-Task 2, if directly use the original simm2/model/mm_dst/util/evaluate_dst.py, there will be division by zero error. I revised evaluate_dst.py by adding an if-else sentence. In funtion **d_f1**, ``r = n_correct / n_true`` is changed to ``r = n_correct / n_true if n_true != 0 else 0`` and ``p = n_correct / n_pred`` is changed to ``p = n_correct / n_pred if n_pred != 0 else 0``. In function **b_stderr**, ``return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)`` is changed to ``np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total) if n_total != 0 else 0``.

6. For the results of Sub-Task 2 on the Test-Std data, please refer to object_f1.


## GPU memory issue

- If you occur CUDA error: out of memory, please modify the parameters:  --max_contexts_length, --max_response_length, --train_batch_size and --eval_batch_size.
