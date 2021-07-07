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
docker pull nvcr.io/nvidia/pytorch:21.02-py3
docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:21.02-py3
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
| MobileNet-V2         | 65.74 | 83.65 | 0.546 | 69.51 | 84.55 | 0.602 
| ResNet-18            | 63.24 | 82.23 | 0.526 | 67.21 | 82.71 | 0.580
| ResNet-34            | 63.60 | 81.68 | 0.522 | 69.92 | 84.72 | 0.605
| ResNet-50            | 69.26 | 85.03 | 0.590 | 73.15 | 87.03 | 0.643
| EfficientNet-B0      | 69.12 | 85.66 | 0.579 | 73.63 | 87.51 | 0.652
| EfficientNet-B1      | 69.23 | 85.38 | 0.592 | 74.11 | 87.62 | 0.658
| EfficientNet-B3      | 70.05 | 85.27 | 0.595 | 74.73 | 88.01 | 0.662
| EfficientNet-B5      | 66.87 | 84.04 | 0.560 | 73.07 | 86.91 | 0.636
| Inception-V3         | 65.30 | 82.83 | 0.530 | 71.45 | 85.64 | 0.622
| InceptionResnet-V2   | 67.42 | 83.60 | 0.559 | 72.68 | 86.37 | 0.629
| Inception-V4         | 67.50 | 83.63 | 0.572 | 74.19 | 87.63 | 0.655
| SE-ResNeXt-101-32x4d | 72.39 | 86.57 | 0.635 | 76.73 | 89.09 | 0.691 
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20 | DF20 | DF20 | DF20M | DF20M | DF20M

### ViT x CNN Performance Evaluation
Classification results of selected CNN and ViT architectures on DF20 and DF20\,-\,Mini dataset for two input resolutions [299𐄂299, 384𐄂384].

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| EfficientNet-B0     | 65.66 | 83.35 | 0.531 | 70.38 | 85.18 | 0.613
| EfficientNet-B3     | 66.90 | 83.49 | 0.537 |  𐄂  | 𐄂  | 𐄂
| SE-ResNeXt-101      | 69.48 | 85.58 | 0.593 |  𐄂  | 𐄂  | 𐄂
| ViT-Base/16         | 69.37 | 86.54 | 0.589 | 70.38 | 85.18 | 0.613
| ViT-Large/16        | 70.71 | 86.51 | 0.599 | 75.34 | 88.11 | 0.679
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20 | DF20 | DF20 | DF20M | DF20M | DF20M

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| EfficientNet-B0  | 70.22 | 85.69 | 0.596 | 75.27 | 88.65 | 0.670
| EfficientNet-B3  | 72.09 | 87.17 | 0.624 |  𐄂  | 𐄂  | 𐄂
| SE-ResNeXt-101   | 72.34 | 87.53 | 0.631 |  𐄂  | 𐄂  | 𐄂
| ViT-Base/16      | 74.84 | 88.74 | 0.655 | 79.40 | 90.93 | 0.724
| ViT-Large/16     | 75.96 | 89.37 | 0.664 | 81.25 | 91.93 | 0.747
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20 | DF20 | DF20 | DF20M | DF20M | DF20M

### Metadata Usage Experiment
Performance gains from Fungus observation metadata: H - Habitat, S - Substrate, M - Month, and their combinations, on DF20 and DF20-Mini. ViT-Base/16 with image size 224𐄂224. 

#### DF20-Mini
| H | M | S | Top1 [%] | Top3 [%] | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 𐄂  | 𐄂  | 𐄂  |  73.45 | 87.15 | 0.658 |
| ✔ | 𐄂  | 𐄂  | +2.00 | +1.42 | +0.036  | 
| 𐄂  | ✔ | 𐄂  | +1.37 | +1.23 | +0.024 |
| 𐄂  | 𐄂  | ✔ | +0.98 | +0.96 | +0.016 |
| 𐄂  | ✔ | ✔ | +2.30 | +2.10 | +0.039 |
| ✔ | 𐄂  | ✔ | +2.92 | +2.41 | +0.051 |
| ✔ | ✔ | 𐄂  | +3.16 | +2.50 | +0.056 |
| ✔ | ✔ | ✔ | +3.58 | +3.05 | +0.062 |
 #### DF20
| H | M | S | Top1 | Top3 | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 𐄂  | 𐄂  | 𐄂  | 69.37 | 86.54 | 0.589 |
| ✔ | 𐄂  | 𐄂  | +1.70 | +1.10 | +0.029  |
| 𐄂  | ✔ | 𐄂  | +0.77 | +0.19 | +0.011   |
| 𐄂  | 𐄂  | ✔ | +0.85 | +0.69 | +0.014  |
| 𐄂  | ✔ | ✔ | +1.29 | +0.80 | +0.020   |
| ✔ | 𐄂  | ✔ | +2.75 | +2.01 | +0.043   |
| ✔ | ✔ | 𐄂  | +2.20 | +1.24 | +0.037  | 
| ✔ | ✔ | ✔ | +2.88 | +1.65 | +0.047 |



## License

The code and dataset is released under the BSD License. There is some limitations for commercial usage.
In other words, the training data, metadata, and models are are available only for non-commercial research purposes only.

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
