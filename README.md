# ReceiptDetection

`This repo was tested on OS: Ubuntu 18.04.4 LTS`

`Goal: localizing text instances in a receipt image`

# Content
1. [Requirements](https://github.com/Funix-Software/ReceiptDetection/tree/develop#1-requirements)
2. [Data](https://github.com/Funix-Software/ReceiptDetection/tree/develop#2-data)
    - 2.1. [Download data](https://github.com/Funix-Software/ReceiptDetection/tree/develop#21-downloading-data-via-one-of-these-links-below)
    - 2.2. [Data preparation](https://github.com/Funix-Software/ReceiptDetection/tree/develop#22-data-preparation)
        - 2.2.1. [For training](https://github.com/Funix-Software/ReceiptDetection/tree/develop#221-for-training)
        - 2.2.2. [For validation and testing](https://github.com/Funix-Software/ReceiptDetection/tree/develop#222-for-validation-and-testing)
3. [Pretrained model](https://github.com/Funix-Software/ReceiptDetection/tree/develop#3-pretrained-model)
4. [Training](https://github.com/Funix-Software/ReceiptDetection/tree/develop#4-training)
5. [Evaluation](https://github.com/Funix-Software/ReceiptDetection/tree/develop#5-evaluation)
    - 5.1. [Result](https://github.com/Funix-Software/ReceiptDetection/tree/develop#51-result)
6. [Citations](https://github.com/Funix-Software/ReceiptDetection/tree/develop#6-citations)
7. [Appendix: PAN architecture](https://github.com/Funix-Software/ReceiptDetection/tree/develop#7-appendix)
    
# 1. Requirements
```
pip install -r requirements.txt
```
or create a new env with conda
```
conda env create -f environment.yml
```

# 2. Data
## 2.1. Downloading data via one of these links below:

[Google Drive](https://drive.google.com/open?id=1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2)

[Baidu](https://pan.baidu.com/s/1a57eKCSq8SV8Njz8-jO4Ww#list/path=%2FSROIE2019&parentPath=%2F)

## 2.2. Data preparation
- Extracting data from `2.1` into `dataset` folder
- An example of image:

	<img width="300" height="600" src="./assets/6567.jpg"> <br>

- All gt.txt files should follow:
    ```
    x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1,transcript_1
    x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2,transcript_2
    x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3,transcript_3
    ```
- Overall, the dataset folder should be:
```
dataset
│
└───all_imgs_gts_training_file
│   │   train_img_name_0.jpg
│   │   train_gt_name_0.txt
│   │   val_img_name_1.jpg
│   │   val_gt_name_1.txt
│   |   ...
│
└───path_for_train
│   │
│   └───img
│   │
│   └───gt
|
└───path_for_val
│   │
│   └───img
│   │
│   └───gt
|
└───path_for_test
│   │
│   └───img
│   │
│   └───gt
|
└───preprocessed
    │
    └───path_for_val
    │   │  
    │   └───img
    │   │   
    |   └───gt
    │   │   
    |   └───diff_gt
    │   │   
    |   └───converted_gt        
    │
    └───path_for_test
        │  
        └───img
        │  
        └───gt 
        │   
        └───diff_gt
        │   
        └───converted_gt         
```

### 2.2.1. For training:
- **imgs and gts path will be written in one text file as**:
```
/path/to/img.jpg\tpath/to/label.txt
```
- To do this:
    - First, inside the `dataset` folder, run:
        ```
        python prepare_data.py
        ```
    - Second :

        Config the `train_data_path` in `config.json` file

### 2.2.2. For validation and testing:
- Getting RoI of validation and testing images, inside the `dataset` folder, run:
```
python crop.py
```
-  Config the `val_data_path` in `config.json` file
# 3. Pretrained model
[Pretrained model](https://drive.google.com/drive/folders/1bKPQEEOJ5kgSSRMpnDB8HIRecnD_s4bR)

# 4. Training
Run:
```
python train.py
```

# 5. Evaluation:
- Using hmean score to rank model's performance:

    <img src="./assets/hmean_gif.latex.gif"> <br>

- Config the `model_path`, `gt_path`, `img_path`, `save_path` in `eval.py` file then run:
```
python eval.py
```
- In case the `img_path` and `gt_path` are cropped RoI, to get back to the original coordinate, get in the `dataset` folder then modify the `target` name in `convert_label_crop2Ori.py` file then run:
```
python convert_label_crop2Ori.py
```
- The final result is saved at `preprocessed/path_for_{target}/converted_gt` folder

## 5.1. Result

**One of the worst result got hmean = 0.6567**
<img src="./assets/f1_6567_worst_rslt_.png">

=====================

**One of the best result with hmean = 1**
<img src="./assets/f1_100_best_rslt_.png">


# 6. Citations
[ arXiv:1908.05900v2 [cs.CV] ](https://arxiv.org/abs/1908.05900v2)

[https://github.com/WenmuZhou/PAN.pytorch](https://github.com/WenmuZhou/PAN.pytorch)

# 7. Appendix
- For the purpose of ICDAR-SROIE Task 1, we can apply `the achor-based text detectiors` which is object detectors or `the anchor-free text detectors` which is text segmentation. In this project, We chose the second method by applying **[Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/abs/1908.05900)** or **PAN** in short.
    - According to PAN's author, there are two main challenges still exist in scene text detection:
        - first: the trade-off between speed and accuracy 
        - second: the model can detect the arbitrary-shaped text instance. This second advantage may not be useful in this case but the result has shown that this's still be the powerful tool for this task.
    - Recently, some methods have been proposed to tackle arbitrary-shaped text detection, but they rarely take the speed of the entire pipeline into consideration, which may fall short in practical applications. PAN is equipped with a `low computational-cost segmentation head` and a `learnable post-processing`.

## 7.1. PAN pipeline

<img src="./assets/PAN's_pipeline.png">

## 7.2. PAN architecture

<img src="./assets/PAN_architecture.png">

- For high efficiency, the backbone of the segmentation network must be lightweight: resnet18 is used for lightweight backbone
- But the features offered by a lightweight backbone often have small receptive fields and weak representation capabilities. Therefore PAN model proposed the segmentation-head which includes 2 modules:
	- Feature Pyramid Enhancement Module (FPEM)
	- Feature Fusion Module (FFM)
### 7.2.3. Pixel Aggregation (PA)
- In the post-processing phase, there are 2 ideas:
	- Distance between text pixel and kernel in the same text instance must be small enough in order to assign the text pixels to the right kernel. To achieve this, PAN used Loss aggregation:<br><img src="./assets/Lagg_gif.latex.gif">
	- The other idea is that the distance between kernels must be far enough. To achive this, PAN used Loss discriminate:<br><img src="./assets/Ldis_gif.latex.gif">

## 7.3. PAN's Loss

<img src="./assets/Lall_gif.latex.gif">
<img src="./assets/Lall_explain_gif.latex.gif">

## 7.4. Conclusion
- There are two phases: segmentation head and post processing which may slow down the model. But comparing with other models (in the comparion section of the paper), PAN still gets good performance while keeping high speed for (curve) text detection. 

- The inference phase highly depends on cv2 to find the connected component. 

- It's hard to explain the reason of choosing alpha, beta, delta_dis, delta_agg for the loss function which may cost a lot of time.

- To increase the performance of model, we can increase the number of FPEM module or replace the resnet18 backbone by resnet50, ... or applying new method for FFM module to fuse the FPEM modules but it will somehow affect the model's speed.