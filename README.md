[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/danish-fungi-2020-not-just-another-image/image-classification-on-df20)](https://paperswithcode.com/sota/image-classification-on-df20?p=danish-fungi-2020-not-just-another-image)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/danish-fungi-2020-not-just-another-image/image-classification-on-df20-mini)](https://paperswithcode.com/sota/image-classification-on-df20-mini?p=danish-fungi-2020-not-just-another-image)

# News
- Metrics slightly updated! Retrained with PyTorch NGC Docker Container 20.07 and on Ampere GPUs only (3080 / 3090)
- EXIF metadata available! You can read it dirrectly from the images.

# Danish Fungi 2020 - Not Just Another Image Recognition Dataset

By [Lukas Picek](https://sites.google.com/view/picekl) et al. 
[MAIL](mailto:lukaspicek@gmail.com?subject=[GitHub]%20DanishFungi2020%20Project)

## Introduction

Supplementary material to:
Danish Fungi 2020 - Not Just Another Image Recognition Dataset 

In order to support research in fine-grained plant classification and to allow full reproducibility of our results, we share the Training Logs and Trained scripts.
- The Images, Checkpoints and Metadata are not included based on size constrains and will be published after the review.

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
3. RUN jupyterlab and start training / experiments
```
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
```
* Check your paths! 

## Results

### CNN Performance Evaluation
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

### ViT x CNN Performance Evaluation
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

### Metadata Usage Experiment
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
