[![PWC](https://img.shields.io/badge/WACV-2022-blue)](https://wacv2022.thecvf.com/)
[![PWC](https://img.shields.io/badge/Proceedings-CVF-red)](https://openaccess.thecvf.com/content/WACV2022/html/Picek_Danish_Fungi_2020_-_Not_Just_Another_Image_Recognition_Dataset_WACV_2022_paper.html)
[![PWC](https://img.shields.io/badge/Pretrained--Models-HuggingFace-blue)](https://huggingface.co/collections/BVRA/danish-fungi-2020-66a2228d0f4902df59d549e8)


# 🍄 Danish Fungi – Not Just Another Image Recognition Dataset

By [Lukas Picek](https://sites.google.com/view/picekl) et al. 
[MAIL](mailto:lukaspicek@gmail.com?subject=[GitHub]%20DanishFungi2020%20Project)

## Description

Danish Fungi 2020 (DF20) is a new dataset for fine-grained visual categorization. The dataset is 
constructed from observations submitted to the Atlas of Danish Fungi and is unique in
(i) its taxonomy-accurate labels with little to no label errors,
(ii) highly unbalanced long-tailed class distribution,
(iii) rich observation metadata about surrounding environment, location, time and device.
DF20 has zero overlap with ImageNet, allowing unbiased comparison of models fine-tuned from publicly 
available ImageNet checkpoints. The proposed evaluation protocol enables testing the ability to 
improve classification using metadata – e.g. precise geographic location, habitat, and substrate,
facilitates classifier calibration testing, and finally allows to study the impact of the device 
settings on the classification performance.

![Species Similarities and differences](figures/fungi_samples.png)

## News
- **30.7. 2024**: We made a new train/test split based on ObservationIDs where data from the same observation do not occur in the test set. 
- **30.7. 2024**: To distinguish it from the original split we call it **DanishFungi24**.
- **30.7. 2024**: Updated baseline performance. All models are retrained and the results bellow updated (there is just a small drop in performance).
- **30.7. 2024**: Model checkpoints are newly available at [Hugging Face Hub Repository](https://huggingface.co/collections/BVRA/danish-fungi-2020-66a2228d0f4902df59d549e8).

## Data
| Subset                   | Images (full-size)                                                                        | Images <br/>(max side size 300px)                                                       | Metadata                                                           |
|--------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Danish Fungi 2020        | [LINK [~110GB]](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-train_val.tar.gz) | [LINK [~6.5GB]](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz) | [LINK](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-metadata.zip) |
| Danish Fungi 2020 – Mini | [LINK [~12.5GB]](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20M-images.tar.gz)  | ---                                                                                     | [LINK](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20M-metadata.zip) |
| Danish Fungi 2024        | ❗Same as for DF20❗                                                                        | ❗Same as for DF20❗                                                                      | [LINK](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DanishFungi2024.zip)                                                           |
| Danish Fungi 2024 – Mini | ❗Same as for DF20 – Mini❗                                                                 | ---                                                                                     |    [LINK](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DanishFungi2024-Mini.zip)                                                   |


To download the dataset files in CMD, use:
```
wget --no-check-certificate URL
```

In order to support research in fine-grained plant classification and to allow full reproducibility of our results, we share the training scripts and data tools.
- Checkpoints are available at [Hugging Face Hub Repository](https://huggingface.co/collections/BVRA/danish-fungi-2020-66a2228d0f4902df59d549e8).
- Train and Validation logs are available at [Weights & Biases Workspace](https://wandb.ai/zcu_cv/DanishFungi2024).



## Installation
Python 3.10+ is required.
### Local instalation
1. Install dependencies
You can use any virtual or local environment. Just use the following commands in your terminal.
```
pip install -r requirements.txt
```
2. Login to [Weights & Biases](https://wandb.ai/site) to log results [*optional].
```
wandb login
```
3. Login to [Hugging Face Hub](https://huggingface.co/) to save and download model checkpoints [*optional].
```
huggingface-cli login
```

## Training
For training navigate to `./training` folder.
To run the training you can use the provided `train.ipynb` notebook or `train.py` CLI.
In both you have to:
* Specify valid paths, wandb settings, etc. in **train.ipynb** or local environment and run. In the notebook
all variables that must be "set" have `"changethis"` as value.

```
python train.py \
    --train-path $TRAIN_METADATA_PATH \
    --test-path $TEST_METADATA_PATH \
    --config-path ../configs/DF24M_224_config.yaml \
    --cuda-devices $CUDA_DEVICES \
    --wandb-entity $WANDB_ENTITY [**optional**] \
    --wandb-project $WANDB_PROJECT [**optional**] \
    --hfhub-owner $HFHUB_OWNER [**optional**]
```


os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICES"]


## Exploiting metadata with late-fusion
To allow easy use of the available observation metadata (i.e., information about habitat, substrate etc.)
we provide a notebook `./inference/metadata_fusion.ipynb` that uses the late metadata-fusion (described in the paper Section 5.2).

## Results

### CNN Performance Evaluation [**Updated**]
Classification performance of selected CNN architectures on DF24 and DF24 - Mini.
All networks share the settings described in the paper (Section 5.1) and were trained on 299×299 images.


|  | Top1 [%] | Top3 [%]          | F1 [%]   | Top1 [%] | Top3 [%] | F1 [%]   |
| ---------------- |----------|-------------------|-------|------|-------|-------|
| MobileNet-V2         | 60.58    | 78.90 | 48.59 | 66.12 | 82.17 | 55.21 |
| ResNet-18            | 55.80    | 75.17 | 42.98 | 60.16 | 77.66 | 49.13 |
| ResNet-34            | 56.80    | 77.17 | 43.21 | 63.54 | 80.30 | 52.88 |
| ResNet-50            | 60.58    | 79.82 | 48.43 | 66.63 | 82.53 | 56.24 |
| EfficientNet-B0      | 63.04    | 80.25 | 50.39 | 67.99 | 83.58 | 57.31 |
| EfficientNet-B1      | 64.14    | 81.22 | 52.59 | 69.20 | 84.28 | 58.54 |
| EfficientNet-B3      | 63.77    | 80.76 | 51.44 | 70.38 | 85.13 | 59.68 |
| EfficientNet-B5      | 63.28    | 81.25 | 51.50 | 71.51 | 85.89 | 60.94 |
| Inception-V3         | 60.79    | 79.06 | 47.88 | 68.49 | 83.74 | 57.76 |
| InceptionResnet-V2   | 63.06    | 79.68 | 50.49 | 70.16 | 84.75 | 59.34 |
| Inception-V4         | 63.47    | 81.63 | 51.83 | 70.39 | 85.13 | 60.11 |
| SE-ResNeXt-101-32x4d | 65.85    | 83.03 | 53.32 | 72.89 | 86.80 | 62.80 |
| ---------------- | ----     | ---- | ----  | ---- | ----  | ----  |
| Dataset | DF24M    | DF24M             | DF24M | DF24 | DF24  | DF24  | 


### ViT x CNN Performance Evaluation [**Updated**]
Classification results of selected CNN and ViT architectures on DF24 and DF24 - Mini dataset for two input resolutions 224×224, 384×384.
**Updated results based on new dataset split.**

#### Input resolution: 224×224
|  | Top1 [%] | Top3 [%] | F1 [%]   | Top1 [%] | Top3 [%] | F1 [%]   |
| ---------------- |---------|----------|-------|----------|--------|-------|
| EfficientNet-B0     | 58.58   | 77.01    | 46.00 | 64.57    | 81.20  | 53.74 |
| EfficientNet-B3     | 59.31   | 78.79    | 47.83 | 67.13    | 82.74  | 56.61 |
| SE-ResNeXt-101      | 62.42   | 80.71    | 50.01 | 69.83    | 84.76  | 59.69 |
| ViT-Base/16         | 65.33   | 82.44    | 52.28 | 70.26    | 84.86  | 60.31 |
| ViT-Large/16        | 67.52   | 84.46    | 55.90 | 73.65    | 87.30  | 64.30 |
| ---------------- | ----    | ----     | ----  | ----     | ----   | ----  |
| Dataset | DF24M   | DF24M    | DF24M | DF24     | DF24   | DF24  | 

#### Input resolution: 384×384

|  | Top1 [%] | Top3 [%] | F1 [%]   | Top1 [%] | Top3 [%] | F1 [%]   |
| ---------------- |----------|---------|-------|----------|----------|-------|
| EfficientNet-B0  | 63.79    | 81.60   | 51.22 | 70.16    | 85.00    | 59.34 |
| EfficientNet-B3  | 65.14    | 82.46   | 52.55 | 72.47    | 86.63    | 62.31 |
| SE-ResNeXt-101   | 68.06    | 84.00   | 56.22 | 74.83    | 88.13    | 65.32 |
| ViT-Base/16      | 69.33    | 85.22   | 57.94 | 76.08    | 88.91    | 66.76 |
| ViT-Large/16     | 72.20    | 87.46   | 60.23 | 78.81    | 90.64    | 70.25 |
| ---------------- | ----     | ----    | ----  | ----     | ----     | ----  |
| Dataset | DF24M    | DF24M   | DF24M | DF24     | DF24     | DF24  |

### Metadata-fusion experiment [**Updated**]
Performance gains aquired by exploiting the observation metadata, i.e. Habitat (H), Substrate (S), and Month (M).
Additionally, we provide performance gains based on the ObservationID grouping of calibrated predictions (average over class score).
The method for late metadata-fusion is described in the paper (Section 5.2). 


#### DF24 - ViT-Large/16 with image size 384×384. [**Updated**]

| H | M | S   | Top1 [%] | Top3 [%] | F1 [%]  |
| ---- | ---- |-----|----------|----------|---------|
| 𐄂 | 𐄂 | 𐄂 | _78.89_  | _90.71_  | _70.38_ |
| ✔ | 𐄂 | 𐄂 | +1.55    | +1.10    | +3.22   |
| 𐄂 | ✔ | 𐄂 | +0.71    | +0.62    | +1.17   |
| 𐄂 | 𐄂 | ✔ | +0.90    | +0.76    | +1.86   |
| 𐄂 | ✔ | ✔ | +1.53    | +1.22    | +2.87   |
| ✔ | 𐄂 | ✔ | +2.12    | +1.57    | +4.53   |
| ✔ | ✔ | 𐄂 | +2.01    | +1.54    | +3.98   |
| ✔ | ✔ | ✔ | +2.53    | +1.95    | +5.13   |

#### DF24 - ViT-Large/16 - 384×384 - With ObservationID grouping and calibration.

| H | M | S   | Top1 [%] | Top3 [%] | F1 [%]  |
| ---- | ---- |-----|----------|----------|---------|
| 𐄂 | 𐄂 | 𐄂 | _85.89_  | _95.47_  | _77.87_ |
| ✔ | 𐄂 | 𐄂 | +1.17    | +0.73    | +3.04   |
| 𐄂 | ✔ | 𐄂 | +0.65    | +0.33    | +1.63   |
| 𐄂 | 𐄂 | ✔ | +0.46    | +0.45    | +0.96   |
| 𐄂 | ✔ | ✔ | +1.07    | +0.71    | +2.36   |
| ✔ | 𐄂 | ✔ | +1.64    | +1.03    | +3.81   |
| ✔ | ✔ | 𐄂 | +1.80    | +1.05    | +4.28   |
| ✔ | ✔ | ✔ | +2.07    | +1.22    | +4.81   |


 #### DF24  ViT-Base/16 with image size 224×224.
| H | M | S | Top1    | Top3    | F1      |
| ---- | ---- | ---- |---------|---------|---------|
| 𐄂 | 𐄂 | 𐄂 | _70.33_ | _84.88_ | _60.44_ |
| ✔ | 𐄂 | 𐄂 | +1.95   | +1.75   | +3.60   |
| 𐄂 | ✔ | 𐄂 | +1.26   | +1.06   | +1.88   |
| 𐄂 | 𐄂 | ✔ | +1.41   | +1.19   | +2.29   |
| 𐄂 | ✔ | ✔ | +2.28   | +1.96   | +3.78   |
| ✔ | 𐄂 | ✔ | +2.85   | +2.61   | +5.28   |
| ✔ | ✔ | 𐄂 | +2.81   | +2.52   | +4.95   |
| ✔ | ✔ | ✔ | +3.56   | +3.22   | +6.39   |


 #### DF24 - ViT-Base/16 - 224×224 - With ObservationID grouping and calibration. 

| H | M | S | Top1    | Top3    | F1      |
| ---- | ---- | ---- |---------|---------|---------|
| 𐄂 | 𐄂 | 𐄂 | _79.49_ | _92.10_ | _69.18_ |
| ✔ | 𐄂 | 𐄂 | +1.88   | +1.08   | +3.96   |
| 𐄂 | ✔ | 𐄂 | +1.07   | +0.77   | +1.92   |
| 𐄂 | 𐄂 | ✔ | +1.15   | +0.75   | +2.03   |
| 𐄂 | ✔ | ✔ | +2.04   | +1.30   | +3.48   |
| ✔ | 𐄂 | ✔ | +2.61   | +1.61   | +5.23   |
| ✔ | ✔ | 𐄂 | +2.75   | +1.59   | +5.27   |
| ✔ | ✔ | ✔ | +3.31   | +2.07   | +6.36   |


## License

The code and dataset is released under the BSD License. There is some limitations for commercial usage.
In other words, the training data, metadata, and models are available only for non-commercial research purposes only.

## Citation

If you use *Danish Fungi* for your research or application, please consider citation:

```
@inproceedings{picek2022danish,
  title={Danish fungi 2020-not just another image recognition dataset},
  author={Picek, Luk{\'a}{\v{s}} and {\v{S}}ulc, Milan and Matas, Ji{\v{r}}{\'\i} and Jeppesen, Thomas S and Heilmann-Clausen, Jacob and L{\ae}ss{\o}e, Thomas and Fr{\o}slev, Tobias},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1525--1535},
  year={2022}
}
```
```
@article{picek2022automatic,
  title={Automatic fungi recognition: deep learning meets mycology},
  author={Picek, Luk{\'a}{\v{s}} and {\v{S}}ulc, Milan and Matas, Ji{\v{r}}{\'\i} and Heilmann-Clausen, Jacob and Jeppesen, Thomas S and Lind, Emil},
  journal={Sensors},
  volume={22},
  number={2},
  pages={633},
  year={2022},
  publisher={Mdpi}
}
```
## Contact

```
[Lukas Picek](lukaspicek@gmail.com, picekl@ntis.zcu.cz)
```
