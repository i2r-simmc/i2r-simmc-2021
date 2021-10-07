Codes submitted to SIMMC2 challenge (https://github.com/facebookresearch/simmc2), 
a track of DSTC 10 (https://dstc10.dstc.community/home)

# Overview
For Sub-Task 1,3,4, we developed an end-to-end encoder-decoder model based on BART (Lewis et al., 2020) 
for generating outputs of the tasks (Sub-Task #1, Sub-Task #3 , Sub-Task #4 Response) 
in a single string, called joint learning model, and another variant 
for generating outputs of the Sub-Task #4 Retrieval, called retrieval model. 
The retrieval model trains on Sub-Task 4 only. 
The two models are trained and evaluated separately. Please refer to folder `subtask_134` for further details.

For Sub-Task 2, we developed Bi-Encoder and Poly-Encoder based models (Humeau et al., 2020). We regard the objects in the corresponding scenes as the candidate objects. We regard the previous system transcript and current user transcript as the context. And for each candidate object, we compute the dot-product between it and the context. If Sub-Task 1 returns a binary label (0 or 1) for the current turn, Sub-Task 2 will return top-K (K=2 as default) object IDs as the results. Please refer to folder `subtask_2` for further details.

# Important Links

* [Instructions for Sub-Tasks 1, 3, and 4](subtask_134/README.md)
* [Instructions for Sub-Tasks 2](subtask_2/README.md)

# Results
So far our results for devtest set are as followings:

| Sub-Task 1 | Accuracy |
| :------: | :------: |
| Baseline | 73.9 |
| Our model | 88.9 |


| Sub-Task 2 | Object F1 |
| :------: | :-------: |
| GPT2     |   0.366   |
| MTN-SIMMC2 | - |
| Our model | 0.405 |

| Sub-Task 3 | Dialog Act F1 | Slot F1 | Request Slot F1 | Joint Accuracy |
| :------: | :-----------: | :-----: | :-------------: | :------------: |
| GPT2     | 0.945         | 0.817   | 0.896           | 0.446          |
| MTN-SIMMC2 | 0.934 | 0.748 | 0.854     | 0.283          |
| Our model | 0.963 | 0.864 | 0.932     | 0.815          |

| Sub-Task 4 (Generation) |      BLEU |
| :------: | :-------: |
| GPT2     |   0.192   |
| MTN-SIMMC2 | 0.217 |
| Our model | 0.3462 |

| Sub-Task 4 (Retrieval) |    MRR    |  R@1 | R@5 | R@10 | Mean Rank |
| :------: | :-------: | :---: | :-------: | :------: | :-------: |
| Random   |   0.052   |   0.010   |   0.050   |   0.100   |   50.0   |
| GPT2     |   0.088   |   0.026   |   0.107   |   0.184   |   38.0   |
| Our model     |   0.666   |   0.558   |   0.799   |   0.876   |   5.6   |

# Combine the outputs of Sub-Task 2 and Sub-Task 3
Since we use different models for Sub-Task 2 and Sub-Task 3, please run the following Python command to combine the outputs of Sub-Task 2 and Sub-Task 3 into a json file.

```shell
python combine_output_subtask_2_3.py --input_subtask2_path subtask2/bi-encoder_teststd_predict/dstc10-simmc-teststd-pred-subtask-2.json --input_subtask3_path subtask_134/output_134/teststd/dstc10-simmc-teststd-pred-subtask-only-3.json --output_json_path outputs/dstc10-simmc-teststd-pred-subtask-3.json
```

# Output json files of Sub-Task 1, 2, 3 and 4 for Test-Std
We put the required json files for Sub-Task 1, 2, 3 and 4 under folder ``outputs``. Please use them for evaluation.
