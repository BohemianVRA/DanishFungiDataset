import cv2
import pandas as pd
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class DanishFungiDataset(Dataset):
    """
    Dataset class that supplies extra features along with images and labels.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing dataset information.
    image_path_feature : str
        Column name for image paths in the DataFrame.
    target_feature : str
        Column name for target labels in the DataFrame.
    extra_features : list of str
        List of column names for extra features in the DataFrame.
    transform : callable, optional
        Transformations to be applied to the images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_path_feature: str,
        target_feature: str,
        extra_features: list[str],
        transform=None,
    ):
        """
        Initialize the dataset with the DataFrame and feature information.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing dataset information.
        image_path_feature : str
            Column name for image paths in the DataFrame.
        target_feature : str
            Column name for target labels in the DataFrame.
        extra_features : list of str
            List of column names for extra features in the DataFrame.
        transform : callable, optional
            Transformations to be applied to the images.
        """
        self.df = df
        self.image_path_feature = image_path_feature
        self.target_feature = target_feature
        self.extra_features = extra_features
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.df)

    def get_extra_features_names(self) -> list[str]:
        """
        Return the list of extra feature names.

        Returns
        -------
        list of str
            List of extra feature names.
        """
        return self.extra_features

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve the image, label, and extra features for the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - image : torch.Tensor
                The transformed image.
            - label : any
                The target label.
            - file_path : str
                The file path of the image.
            - selected_feature_values : dict
                Dictionary of extra feature values.
        """
        file_path = self.df[self.image_path_feature].iloc[idx]
        label = self.df[self.target_feature].iloc[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        selected_feature_values = {
            feature: self.df[feature].iloc[idx] for feature in self.extra_features
        }

        return image, label, file_path, selected_feature_values


def get_transforms(
    model_mean: list[float], model_std: list[float], image_size: tuple[int, int]
) -> Compose:
    """
    Get the transformations to be applied to the images.

    Parameters
    ----------
    model_mean : list of float
        Mean values for normalization.
    model_std : list of float
        Standard deviation values for normalization.
    image_size : tuple of int
        Desired image size (height, width).

    Returns
    -------
    albumentations.core.composition.Compose
        Composition of transformations.
    """
    return Compose(
        [Resize(*image_size), Normalize(mean=model_mean, std=model_std), ToTensorV2()]
    )
