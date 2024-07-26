import itertools

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader

from .DanishFungiDataset import DanishFungiDataset


def late_metadata_fusion(
    df: pd.DataFrame,
    model,
    dataloader: DataLoader,
    device: torch.device,
    target_feature: str,
    selected_features: list[str],
) -> dict[str, dict]:
    """
    Fusing model predictions with species | metadata priors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    model : torch.nn.Module
        Trained model for making predictions.
    dataloader : DataLoader
        DataLoader for the dataset.
    device : torch.device
        Device to run the model on.
    target_feature : str
        Column name for target labels in the DataFrame.
    selected_features : list of str
        List of feature column names to use for weighting predictions.

    Returns
    -------
    dict of dict
        Dictionary containing weighted predictions and raw predictions for each feature.
    """
    metadata_distributions = {
        feature: get_target_to_feature_conditional_distributions(
            df, feature, target_feature, add_to_missing=False
        )
        for feature in selected_features
    }
    target_distribution = df[target_feature].value_counts(normalize=True)

    (
        predictions,
        predictions_raw,
        ground_truth_labels,
        ground_truth_features,
    ) = predict_with_features(model, dataloader, device)

    feature_prior_ratios = {}
    weighted_predictions_complete = {}
    for feature in selected_features:
        metadata_distribution = metadata_distributions[feature]
        seen_feature_values = ground_truth_features[feature]

        (
            weighted_predictions,
            weighted_predictions_raw,
            feature_prior_ratio,
        ) = weight_predictions_by_feature_distribution(
            target_to_feature_conditional_distributions=metadata_distribution,
            target_distribution=target_distribution,
            ground_truth_labels=ground_truth_labels,
            raw_predictions=predictions_raw,
            ground_truth_feature_categories=seen_feature_values,
        )
        weighted_predictions_complete[feature] = {
            "predictions": weighted_predictions,
            "predictions_raw": weighted_predictions_raw,
        }
        feature_prior_ratios[feature] = feature_prior_ratio

    merged_predictions = post_process_prior_combinations(
        predictions_raw, feature_prior_ratios
    )
    weighted_predictions_complete.update(merged_predictions)

    for combination, _weighted_predictions in weighted_predictions_complete.items():
        _predictions = _weighted_predictions["predictions"]
        _predictions_raw = _weighted_predictions["predictions_raw"]
        print(
            get_metrics(ground_truth_labels, _predictions, _predictions_raw),
        )

    return weighted_predictions_complete


def get_target_to_feature_conditional_distributions(
    df: pd.DataFrame, feature: str, target_feature: str, add_to_missing: bool = True
) -> pd.Series:
    """
    Returns the target conditional distributions per feature category.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    feature : str
        Column name for the feature in the DataFrame.
    target_feature : str
        Column name for the target labels in the DataFrame.
    add_to_missing : bool, optional
        Whether to add a count to missing combinations to avoid zero division.

    Returns
    -------
    pd.Series
        Series of target conditional distributions per feature category.
    """
    feature_per_target_counts = (
        df.groupby([feature, target_feature])[feature]
        .count()
        .unstack(fill_value=0)
        .stack()
    )
    if add_to_missing:
        feature_per_target_counts += 1

    per_feature_counts = feature_per_target_counts.groupby(level=0).sum()

    zeroed_features = per_feature_counts[per_feature_counts == 0].index
    for zeroed_feature in zeroed_features:
        feature_per_target_counts[zeroed_feature] += 1

    distributions = feature_per_target_counts / per_feature_counts
    assert all(
        abs(distributions.groupby(level=0).sum() - 1) < 1e-6
    ), f"Target distributions do not sum to 1 per feature category of {feature}"

    return distributions


def predict_with_features(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list, list, list, dict]:
    """
    Makes predictions using the model on the DataLoader, which must contain ExtraFeaturesDataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for making predictions.
    loader : DataLoader
        DataLoader containing the dataset.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    tuple of (list, list, list, dict)
        Predictions, raw predictions, ground truth labels, and extra feature values.
    """
    assert isinstance(
        loader.dataset, DanishFungiDataset
    ), "Dataset in loader must be of type ExtraFeaturesDataset"

    extra_features = loader.dataset.get_extra_features_names()
    batch_size = loader.batch_size

    predictions = np.zeros(len(loader.dataset))
    ground_truth_labels = []
    predictions_raw = []
    ground_truth_features = {feature: [] for feature in extra_features}

    for i, (images, labels, _, features) in enumerate(
        tqdm.tqdm(loader, total=len(loader))
    ):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            y_preds = model(images)

        start_idx = i * batch_size
        end_idx = start_idx + len(labels)
        predictions[start_idx:end_idx] = y_preds.argmax(1).cpu().numpy()
        ground_truth_labels.extend(labels.cpu().numpy())
        predictions_raw.extend(y_preds.cpu().numpy())

        for extra_feature in extra_features:
            ground_truth_features[extra_feature].extend(features[extra_feature])

    return (
        predictions.tolist(),
        predictions_raw,
        ground_truth_labels,
        ground_truth_features,
    )


def weight_predictions_by_feature_distribution(
    target_to_feature_conditional_distributions: pd.Series,
    target_distribution: pd.Series,
    ground_truth_labels: list,
    raw_predictions: list,
    ground_truth_feature_categories: list,
) -> tuple[list, list, list]:
    """
    Weights predictions by feature distribution.

    Parameters
    ----------
    target_to_feature_conditional_distributions : pd.Series
        Series of target conditional distributions per feature category.
    target_distribution : pd.Series
        Series of target label distributions.
    ground_truth_labels : list
        List of ground truth labels.
    raw_predictions : list
        List of raw predictions from the model.
    ground_truth_feature_categories : list
        List of feature values corresponding to the ground truth labels.

    Returns
    -------
    tuple of (list, list, list)
        Weighted predictions, raw weighted predictions, and feature prior ratios.
    """
    weighted_predictions = []
    weighted_predictions_raw = []
    feature_prior_ratios = []

    for lbl, raw_prediction, feature_category in tqdm.tqdm(
        zip(ground_truth_labels, raw_predictions, ground_truth_feature_categories),
        total=len(ground_truth_labels),
    ):
        predictions = softmax(raw_prediction)
        feature_conditional_distribution = target_to_feature_conditional_distributions[
            int(feature_category)
        ]
        p_feature = (predictions * feature_conditional_distribution) / (
            sum(predictions * feature_conditional_distribution)
        )

        prior_ratio = p_feature / target_distribution
        max_index = np.argmax(prior_ratio * predictions)

        feature_prior_ratios.append(prior_ratio)
        weighted_predictions_raw.append(prior_ratio * predictions)
        weighted_predictions.append(max_index)

    return weighted_predictions, weighted_predictions_raw, feature_prior_ratios


def post_process_prior_combinations(
    raw_predictions: list,
    feature_prior_ratios: dict,
) -> dict:
    """
    Post-processes predictions by combining feature prior ratios.

    Parameters
    ----------
    raw_predictions : list
        List of raw predictions from the model.
    feature_prior_ratios : dict
        Dictionary of feature prior ratios.

    Returns
    -------
    dict
        Dictionary containing merged predictions and raw predictions for each combination of
        features.
    """
    features = list(feature_prior_ratios.keys())
    merged_predictions = {}

    for num_features in range(2, len(features) + 1):
        for combination in itertools.combinations(features, num_features):
            selected_prior_ratios = [
                feature_prior_ratios[feature] for feature in combination
            ]
            merged_preds, merged_preds_raw = weight_predictions_combined_feature_priors(
                raw_predictions, selected_prior_ratios
            )
            merged_predictions["+ ".join(combination)] = {
                "predictions": merged_preds,
                "predictions_raw": merged_preds_raw,
            }

    return merged_predictions


def weight_predictions_combined_feature_priors(
    raw_predictions: list, feature_prior_ratios: list
) -> tuple[list, list]:
    """
    Weights predictions by combining feature prior ratios.

    Parameters
    ----------
    raw_predictions : list
        List of raw predictions from the model.
    feature_prior_ratios : list
        List of feature prior ratios.

    Returns
    -------
    tuple of (list, list)
        Merged predictions and raw merged predictions.
    """
    merged_predictions = []
    merged_predictions_raw = []

    for idx, raw_prediction in enumerate(raw_predictions):
        prediction = softmax(raw_prediction)
        for prior_ratio in feature_prior_ratios:
            prediction *= prior_ratio[idx]

        merged_prediction = prediction / prediction.sum()
        max_index = np.argmax(merged_prediction)

        merged_predictions_raw.append(merged_prediction)
        merged_predictions.append(max_index)

    return merged_predictions, merged_predictions_raw


def get_metrics(
    ground_truth_labels: list,
    predictions: list,
    predictions_raw: list,
) -> tuple[float, float, float]:
    """
    Computes evaluation metrics.

    Parameters
    ----------
    ground_truth_labels : list
        List of ground truth labels.
    predictions : list
        List of predicted labels.
    predictions_raw : list
        List of raw predictions from the model.

    Returns
    -------
    tuple of (float, float, float)
        F1 score, accuracy, and top-3 recall.
    """
    f1 = f1_score(ground_truth_labels, predictions, average="macro")
    accuracy = accuracy_score(ground_truth_labels, predictions)
    recall_3 = top_k_accuracy_score(ground_truth_labels, predictions_raw, k=3)
    return f1, accuracy, recall_3
