[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/danish-fungi-2020-not-just-another-image/image-classification-on-df20)](https://paperswithcode.com/sota/image-classification-on-df20?p=danish-fungi-2020-not-just-another-image)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/danish-fungi-2020-not-just-another-image/image-classification-on-df20-mini)](https://paperswithcode.com/sota/image-classification-on-df20-mini?p=danish-fungi-2020-not-just-another-image)

# News
- Updated dataset with ObservationIDs from 2023
- New train/test splits for DF20M and DF20 with images grouped based on the same ObservationID - No more images of the
  same observation both in the train and test split simultaneously.
- Updated baseline performance. All models are retrained and the results are updated with new scores.
- Model checkpoints are newly available at [Hugging Face Hub Repository](https://huggingface.co/BVRA).

# Danish Fungi 2020 - Not Just Another Image Recognition Dataset

By [Lukas Picek](https://sites.google.com/view/picekl) et al. 
[MAIL](mailto:lukaspicek@gmail.com?subject=[GitHub]%20DanishFungi2020%20Project)

## Introduction

Supplementary material to:
Danish Fungi 2020 - Not Just Another Image Recognition Dataset 

In order to support research in fine-grained plant classification and to allow full reproducibility of our results, we share the training scripts and data tools.
- Checkpoints are available at [Hugging Face Hub Repository](https://huggingface.co/BVRA).
- Train and Validation logs are available at [Weights & Biases Workspace](https://wandb.ai/zcu_cv/DanishFungi2023).
## Training Data

Available at -> https://sites.google.com/view/danish-fungi-dataset

## Training

1. Download PyTorch NGC Docker Image and RUN docker container

```
docker pull nvcr.io/nvidia/pytorch:21.07-py3
docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:21.07-py3
```

2. Install dependencies inside docker container

```
pip install pandas seaborn timm albumentation tqdm efficientnet_pytorch pretrainedmodels
```
3. Install following:
   * [FGVC](https://github.com/BohemianVRA/FGVC/tree/main)
   * [Hugging Face](https://huggingface.co/docs/transformers/installation)
   
4. RUN jupyterlab and start training / experiments
```
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
```
* Check your paths! 



## Results

### **Updated** - CNN Performance Evaluation
Updated results with the dataset date split based on the unique grouped observationIDs.
Classification performance of selected CNN architectures on DF20 and DF20 - Mini.
All networks share the settings described in Section 6.1 and were trained on 299×299 images.

|  | Top1 [%] | Top3 [%]          | F1     | Top1 [%] | Top3 [%] | F1     |
| ---------------- | ---- |-------------------|--------|------|-------|--------|
| MobileNet-V2         | 60.58 | 78.90 | 0.4859 | 66.12 | 82.17 | 0.5521 |
| ResNet-18            | 55.8  | 75.17 | 0.4298 | 60.16 | 77.66 | 0.4913 |
| ResNet-34            | 56.8  | 77.17 | 0.4321 | 63.54 | 80.30 | 0.5288 |
| ResNet-50            | 60.58 | 79.82 | 0.4843 | 66.63 | 82.53 | 0.5624 |
| EfficientNet-B0      | 63.04 | 80.25 | 0.5039 | 67.99 | 83.58 | 0.5731 |
| EfficientNet-B1      | 64.14 | 81.22 | 0.5259 | 69.20 | 84.28 | 0.5854 |
| EfficientNet-B3      | 63.77 | 80.76 | 0.5144 | 70.38 | 85.13 | 0.5968 |
| EfficientNet-B5      | 63.28 | 81.25 | 0.5150 | 71.51 | 85.89 | 0.6094 |
| Inception-V3         | 60.79 | 79.06 | 0.4788 | 68.49 | 83.74 | 0.5776 |
| InceptionResnet-V2   | 63.06 | 79.68 | 0.5049 | 70.16 | 84.75 | 0.5934 |
| Inception-V4         | 63.47 | 81.63 | 0.5183 | 70.39 | 85.13 | 0.6011 |
| SE-ResNeXt-101-32x4d | 65.85 | 83.03 | 0.5332 | 72.89 | 86.80 | 0.6280 |
| ---------------- | ---- | ---- | ----   | ---- | ----  | ----   |
| Dataset | DF20M | DF20M             | DF20M  | DF20 | DF20  | DF20   | 


### **Updated** - ViT x CNN Performance Evaluation
Updated results with the dataset date split based on the unique grouped observationIDs.
Classification results of selected CNN and ViT architectures on DF20 and DF20 - Mini dataset for two input resolutions 224×224, 384×384.

* 224×224 Resolution:

|  | Top1 [%] | Top3 [%] | F1     | Top1 [%] | Top3 [%] | F1     |
| ---------------- |---------|----------|--------|----------|--------|--------|
| EfficientNet-B0     | 58.58   | 77.01    | 0.4600 | 64.57    | 81.20  | 0.5374 |
| EfficientNet-B3     | 59.31   | 78.79    | 0.4783 | 67.13    | 82.74  | 0.5661 |
| SE-ResNeXt-101      | 62.42   | 80.71    | 0.5001 | 69.83    | 84.76  | 0.5969 |
| ViT-Base/16         | 65.33   | 82.44    | 0.5228 | 70.26    | 84.86  | 0.6031 |
| ViT-Large/16        | 67.52   | 84.46    | 0.5590 | 73.65    | 87.30  | 0.6430 |
| ---------------- | ----    | ----     | ----   | ----     | ----   | ----   |
| Dataset | DF20M   | DF20M    | DF20M  | DF20     | DF20   | DF20   | 

* 384×384 Resolution:

|  | Top1 [%] | Top3 [%] | F1     | Top1 [%] | Top3 [%] | F1      |
| ---------------- |----------|---------|--------|----------|----------|---------|
| EfficientNet-B0  | 63.79    | 81.60   | 0.5122 | 70.16    | 85.00    | 0.5934 |
| EfficientNet-B3  | 65.14    | 82.46   | 0.5255 | 72.47    | 86.63    | 0.6231 |
| SE-ResNeXt-101   | 68.06    | 84.00   | 0.5622 | 74.83    | 88.13    | 0.6532 |
| ViT-Base/16      | 69.33    | 85.22   | 0.5794 | 76.08    | 88.91    | 0.6676 |
| ViT-Large/16     | 72.20    | 87.46   | 0.6023 | 78.81    | 90.64    | 0.7025 |
| ---------------- | ----     | ----    | ----   | ----     | ----     | ----    |
| Dataset | DF20M    | DF20M   | DF20M  | DF20     | DF20     | DF20    |

### **Updated** - Metadata Usage Experiment

Performance gains from Fungus observation metadata: H - Habitat, S - Substrate, M - Month, and their combinations.
Additionally, performance gains based on the ObservationID grouping of predictions (average over class score) and calibration.

#### DF20 - ViT-Large/16 with image size 384×384. 

| H | M | S   | Top1 [%] | Top3 [%] | F1      |
| ---- | ---- |-----| ---- | ---- |---------|
| 𐄂 | 𐄂 | 𐄂 | 78.89 | 90.71 | 0.7038  |
| ✔ | 𐄂 | 𐄂 | +1.55 | +1.10 | +0.0322 |
| 𐄂 | ✔ | 𐄂 | +0.71 | +0.62 | +0.0117 |
| 𐄂 | 𐄂 | ✔ | +0.90 | +0.76 | +0.0186 |
| 𐄂 | ✔ | ✔ | +1.53 | +1.22 | +0.0287 |
| ✔ | 𐄂 | ✔ | +2.12 | +1.57 | +0.0453 |
| ✔ | ✔ | 𐄂 | +2.01 | +1.54 | +0.0398 |
| ✔ | ✔ | ✔ | +2.53 | +1.95 | +0.0513 |

#### DF20 - ViT-Large/16 - 384×384 - With ObservationID grouping and calibration. 

| H | M | S   | Top1 [%] | Top3 [%] | F1      |
| ---- | ---- |-----| ---- | ---- |---------|
| 𐄂 | 𐄂 | 𐄂 | 85.89 | 95.47 | 0.7787  |
| ✔ | 𐄂 | 𐄂 | +1.17 | +0.73 | +0.0304 |
| 𐄂 | ✔ | 𐄂 | +0.65 | +0.33 | +0.0163 |
| 𐄂 | 𐄂 | ✔ | +0.46 | +0.45 | +0.0096 |
| 𐄂 | ✔ | ✔ | +1.07 | +0.71 | +0.0236 |
| ✔ | 𐄂 | ✔ | +1.64 | +1.03 | +0.0381 |
| ✔ | ✔ | 𐄂 | +1.8  | +1.05 | +0.0428 |
| ✔ | ✔ | ✔ | +2.07 | +1.22 | +0.0481 |


 #### DF20 - ViT-Base/16 with image size 224×224.
| H | M | S | Top1 | Top3 | F1      |
| ---- | ---- | ---- | ---- | ---- |---------|
| 𐄂 | 𐄂 | 𐄂 | 70.33 | 84.88 | 0.6044  |
| ✔ | 𐄂 | 𐄂 | +1.95 | +1.75 | +0.0360 |
| 𐄂 | ✔ | 𐄂 | +1.26 | +1.06 | +0.0188 |
| 𐄂 | 𐄂 | ✔ | +1.41 | +1.19 | +0.0229 |
| 𐄂 | ✔ | ✔ | +2.28 | +1.96 | +0.0378 |
| ✔ | 𐄂 | ✔ | +2.85 | +2.61 | +0.0528 |
| ✔ | ✔ | 𐄂 | +2.81 | +2.52 | +0.0495 |
| ✔ | ✔ | ✔ | +3.56 | +3.22 | +0.0639 |


 #### DF20 - ViT-Base/16 - 224×224 - With ObservationID grouping and calibration. 

| H | M | S | Top1 | Top3 | F1      |
| ---- | ---- | ---- | ---- | ---- |---------|
| 𐄂 | 𐄂 | 𐄂 | 79.49 | 92.10  | 0.6918  |
| ✔ | 𐄂 | 𐄂 | +1.88 | +1.08 | +0.0396 |
| 𐄂 | ✔ | 𐄂 | +1.07 | +0.77 | +0.0192 |
| 𐄂 | 𐄂 | ✔ | +1.15 | +0.75 | +0.0203 |
| 𐄂 | ✔ | ✔ | +2.04 | +1.30 | +0.0348 |
| ✔ | 𐄂 | ✔ | +2.61 | +1.61 | +0.0523 |
| ✔ | ✔ | 𐄂 | +2.75 | +1.59 | +0.0527 |
| ✔ | ✔ | ✔ | +3.31 | +2.07 | +0.0636 |


### ~~CNN Performance Evaluation~~
Classification performance of selected CNN architectures on DF20 and DF20 - Mini. All networks share the settings described in Section 6.1 and were trained on 299×299 images.

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| MobileNet-V2         | 65.58 | 83.65 | 0.559 | 69.77 | 85.01 | 0.606 
| ResNet-18            | 62.91 | 81.65 | 0.514 | 67.13 | 82.65 | 0.580
| ResNet-34            | 65.63 | 83.52 | 0.559 | 69.81 | 84.76 | 0.600
| ResNet-50            | 68.39 | 85.22 | 0.587 | 73.49 | 87.13 | 0.649
| EfficientNet-B0      | 67.94 | 85.71 | 0.567 | 73.65 | 87.55 | 0.653
| EfficientNet-B1      | 68.35 | 84.67 | 0.572 | 74.08 | 87.68 | 0.654
| EfficientNet-B3      | 69.59 | 85.55 | 0.590 | 75.69 | 88.72 | 0.673
| EfficientNet-B5      | 68.76 | 85.00 | 0.579 | 76.10 | 88.85 | 0.678
| Inception-V3         | 65.91 | 82.97 | 0.535 | 72.10 | 86.58 | 0.630
| InceptionResnet-V2   | 64.67 | 81.42 | 0.542 | 74.01 | 87.49 | 0.651
| Inception-V4         | 67.45 | 82.78 | 0.560 | 73.00 | 86.87 | 0.637
| SE-ResNeXt-101-32x4d | 72.23 | 87.28 | 0.620 | 77.13 | 89.48 | 0.693 
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20M | DF20M | DF20M | DF20 | DF20 | DF20 | 

### ~~ViT x CNN Performance Evaluation~~
Classification results of selected CNN and ViT architectures on DF20 and DF20\,-\,Mini dataset for two input resolutions [224𐄂224, 384𐄂384].

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| EfficientNet-B0     | 65.66 | 83.35 | 0.531 | 70.33 | 85.19 | 0.613
| EfficientNet-B3     | 67.39 | 83.74 | 0.550 | 72.51 | 86.77 | 0.634
| SE-ResNeXt-101      | 68.87 | 85.14 | 0.585 | 74.26 | 87.78 | 0.660
| ViT-Base/16         | 70.11 | 86.81 | 0.600 | 73.51 | 87.55 | 0.655
| ViT-Large/16        | 71.04 | 86.15 | 0.603 | 75.29 | 88.34 | 0.675
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20M | DF20M | DF20M | DF20 | DF20 | DF20 | 

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| EfficientNet-B0  | 69.62 | 85.96 | 0.582 | 75.35 | 88.67 | 0.670
| EfficientNet-B3  | 71.59 | 87.39 | 0.613 | 77.59 | 90.07 | 0.699
| SE-ResNeXt-101   | 74.23 | 88.27 | 0.651 | 78.72 | 90.54 | 0.708
| ViT-Base/16      | 74.23 | 89.12 | 0.639 | 79.48 | 90.95 | 0.727
| ViT-Large/16     | 75.85 | 89.95 | 0.669 | 80.45 | 91.68 | 0.743
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20M | DF20M | DF20M | DF20 | DF20 | DF20 | 

### ~~Metadata Usage Experiment~~
Performance gains from Fungus observation metadata: H - Habitat, S - Substrate, M - Month, and their combinations, on DF20. 

#### DF20 - ViT-Large/16 with image size 384𐄂384. 
| H | M | S | Top1 [%] | Top3 [%] | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 𐄂  | 𐄂  | 𐄂  |  80.45 | 91.68 | 0.743 |
| ✔ | 𐄂  | 𐄂  | +1.50 | +1.00 | +0.027  | 
| 𐄂  | ✔ | 𐄂  | +0.95 | +0.62 | +0.014 |
| 𐄂  | 𐄂  | ✔ | +1.13 | +0.69 | +0.020 |
| 𐄂  | ✔ | ✔ | +1.93 | +1.27 | +0.032 |
| ✔ | 𐄂  | ✔ | +2.48 | +1.66 | +0.044 |
| ✔ | ✔ | 𐄂  | +2.31 | +1.48 | +0.040 |
| ✔ | ✔ | ✔ | +2.95 | +1.92 | +0.053 |
 #### DF20-Mini - ViT-Base/16 with image size 224𐄂224. 
| H | M | S | Top1 | Top3 | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 𐄂  | 𐄂  | 𐄂  | 73.51 | 87.55 | 0.655 |
| ✔ | 𐄂  | 𐄂  | +1.94 | +1.50 | +0.040  |
| 𐄂  | ✔ | 𐄂  | +1.23 | +0.95 | +0.020   |
| 𐄂  | 𐄂  | ✔ | +1.39 | +1.17 | +0.025  |
| 𐄂  | ✔ | ✔ | +2.47 | +1.98 | +0.042   |
| ✔ | 𐄂  | ✔ | +3.23 | +2.47 | +0.062   |
| ✔ | ✔ | 𐄂  | +3.11 | +2.30 | +0.057  | 
| ✔ | ✔ | ✔ | +3.81 | +2.84 | +0.070 |



## License

The code and dataset is released under the BSD License. There is some limitations for commercial usage.
In other words, the training data, metadata, and models are available only for non-commercial research purposes only.

## Citation

If you use *Danish Fungi* for your research or aplication, please consider citation:

```
@article{picek2021danish,
title={Danish Fungi 2020 - Not Just Another Image Recognition Dataset},
author={Lukáš Picek and Milan Šulc and Jiří Matas and Jacob Heilmann-Clausen and Thomas S. Jeppesen and Thomas Læssøe and Tobias Frøslev},
year={2021},
eprint={2103.10107},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```

## Contact

```
[Lukas Picek](lukaspicek@gmail.com, picekl@ntis.zcu.cz)
```
