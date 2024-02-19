import gc
import json
import os
import os.path as osp
import random
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import tqdm
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


class TestDataset(Dataset):
    def __init__(self, df, selected_features: list, transform=None):
        self.df = df
        self.selected_features = selected_features
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_path = self.df["image_path"].values[idx]
        label = self.df["class_id"].values[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        selected_feature_values = {
            feature: self.df[feature].values[idx] for feature in self.selected_features
        }

        return image, label, file_path, selected_feature_values


def get_transforms(model_mean, model_std, image_size):

    return Compose(
        [Resize(*image_size), Normalize(mean=model_mean, std=model_std), ToTensorV2()]
    )
