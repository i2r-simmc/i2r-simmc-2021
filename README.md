Codes submitted to SIMMC2 challenge (https://github.com/facebookresearch/simmc2), 
a track of DSTC 10 (https://dstc10.dstc.community/home)

# Overview
For Sub-Task 1,3,4, we developed an end-to-end encoder-decoder model based on BART (Lewis et al., 2020) 
for generating outputs of the tasks (Sub-Task #1, Sub-Task #3 , Sub-Task #4 Response) 
in a single string, called joint learning model, and another variant 
for generating outputs of the Sub-Task #4 Retrieval, called retrieval model. 
The retrieval model trains on Sub-Task 4 only. 
The two models are trained and evaluated separately. Please refer to folder `subtask_134` for further details.

For Sub-Task 2, TBA

So far our results for devtest set are as followings:

| Sub-Task 1 | Accuracy |
| :------: | :------: |
| Baseline | 73.9 |
| Our model | 88.9 |


| Sub-Task 2 | Object F1 |
| :------: | :-------: |
| GPT2     |   0.366   |
| MTN-SIMMC2 | - |
| Our model | TBA |

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