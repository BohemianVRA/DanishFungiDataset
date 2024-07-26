import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from fgvc.core.training import predict, train
from fgvc.datasets import get_dataloaders
from fgvc.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.utils.experiment import (get_optimizer_and_scheduler, load_args,
                                   load_config, load_model,
                                   load_train_metadata, save_config)
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import (finish_wandb, init_wandb, resume_wandb,
                              set_best_scores_in_summary)
from PIL import Image, ImageFile
from scipy.special import softmax
from torch.utils.data import DataLoader
from utils.hfhub import export_model_to_huggingface_hub_from_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("script")

SCRATCH_DIR = os.getenv("SCRATCHDIR", "./")


def evaluate(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    path: str,
    device: torch.device = "cpu",
    log_images: bool = False,
):
    """Evaluate model and create example visualizations.

    Parameters
    ----------
    model
        Model to evaluate.
    testloader
        Test data dataloader.
    path
        Directory to store example visualizations.
    device
        Cuda or CPU device.
    log_images
        Whether to store
    """
    if wandb.run is None:
        return

        # evaluate model
    logger.info("Creating predictions.")
    preds, targs, _, scores = predict(model, testloader, device=device)
    print(scores)
    argmax_preds = preds.argmax(1)
    max_conf = softmax(preds, 1).max(1)
    softmax_values = softmax(preds, 1)
    # create wandb prediction table
    train_df = trainloader.dataset.df
    test_df = testloader.dataset.df
    id2class = dict(zip(train_df["class_id"], train_df["species"]))

    pred_df = pd.DataFrame()
    if log_images:
        pred_df["image"] = test_df["image_path"].apply(
            lambda image_path: wandb.Image(data_or_path=Image.open(image_path))
        )

    top5_indices = np.argsort(-softmax_values, axis=1)[
        :, :5
    ]  # Get indices of top 5 softmax values
    top5_species = [str([id2class[i] for i in row]) for row in top5_indices]
    top5_softmax = [
        str(softmax_values[i][top5_indices[i]]) for i in range(len(top5_indices))
    ]

    # Create new columns in pred_df for top 5 predictions and softmax values
    pred_df["top5-species"] = top5_species
    pred_df["top5-softmax"] = top5_softmax
    pred_df["species"] = test_df["species"]
    pred_df["species-predicted"] = [id2class[x] for x in argmax_preds]
    pred_df["class_id"] = test_df["class_id"]
    pred_df["class_id-predicted"] = argmax_preds
    pred_df["max-confidence"] = max_conf
    for col in ["image_path"]:
        pred_df[col] = test_df[col]
    wandb.log({"pred_table": wandb.Table(dataframe=pred_df)})

    wandb.log(
        {
            "test/F1": scores["F1"],
            "test/Accuracy": scores["Accuracy"],
            "test/Recall@3": scores["Recall@3"],
        }
    )


def add_metadata_info_to_config(
    config: dict, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    """Include information from metadata to the training configuration."""
    assert "class_id" in train_df and "class_id" in test_df
    config["number_of_classes"] = len(train_df["class_id"].unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(test_df)
    return config


def train_clf(
    *,
    train_metadata: str = None,
    valid_metadata: str = None,
    config_path: str = None,
    cuda_devices: str = None,
    wandb_entity: str = None,
    wandb_project: str = None,
    resume_exp_name: str = None,
    **kwargs,
):
    """Train model on the classification task."""
    if train_metadata is None or valid_metadata is None or config_path is None:
        # load script args
        args, extra_args = load_args()
        config_path = args.config_path
        cuda_devices = args.cuda_devices
        wandb_entity = args.wandb_entity
        wandb_project = args.wandb_project
        save_to_hfhub = args.save_to_hfhub
    else:
        extra_args = kwargs

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # load training config
    logger.info("Loading training config.")
    config = load_config(
        config_path,
        extra_args,
        run_name_fmt="architecture-loss-augmentations",
        resume_exp_name=resume_exp_name,
    )

    # set device and random seed
    device = set_cuda_device(cuda_devices)
    set_random_seed(config["random_seed"])

    # load metadata
    logger.info("Loading training and validation metadata.")
    train_df, valid_df, test_df = load_train_metadata(config)
    config = add_metadata_info_to_config(config, train_df, valid_df)

    # load model and create optimizer and lr scheduler
    logger.info("Creating model, optimizer, and scheduler.")
    model, model_mean, model_std = load_model(config)

    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    # create dataloaders
    logger.info("Creating DataLoaders.")
    trainloader, validloader, _, _ = get_dataloaders(
        train_df,
        valid_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    # create loss function
    logger.info("Creating loss function.")
    if config["loss"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "FocalLoss":
        criterion = FocalLossWithLogits()
    elif config["loss"] == "SeeSawLoss":
        class_counts = train_df["class_id"].value_counts().sort_index().values
        criterion = SeesawLossWithLogits(class_counts=class_counts)
    else:
        logger.error(f"Unknown loss: {config['loss']}")
        raise ValueError()

    # init wandb
    if wandb_entity is not None and wandb_project is not None:
        if resume_exp_name is None:
            init_wandb(
                config, config["run_name"], entity=wandb_entity, project=wandb_project
            )
        else:
            if "wandb_run_id" not in config:
                raise ValueError("Config is missing 'wandb_run_id' field.")
            resume_wandb(
                run_id=config["wandb_run_id"],
                entity=wandb_entity,
                project=wandb_project,
            )

    # save config to json in experiment path
    if resume_exp_name is None:
        save_config(config)

    # train model
    logger.info("Training the model.")

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb_train_prefix="train/",
        wandb_valid_prefix="val/",
        num_epochs=config["epochs"],
        accumulation_steps=config.get("accumulation_steps", 1),
        clip_grad=config.get("clip_grad"),
        device=device,
        seed=config.get("random_seed", 777),
        path=config["exp_path"],
        resume=resume_exp_name is not None,
        mixup=config.get("mixup"),
        cutmix=config.get("cutmix"),
        mixup_prob=config.get("mixup_prob"),
        apply_ema=config.get("apply_ema"),
        ema_start_epoch=config.get("ema_start_epoch", 0),
        ema_decay=config.get("ema_decay", 0.9999),
    )

    # evaluate model
    model_filename = os.path.join(config["exp_path"] + "/best_f1.pth")
    model.load_state_dict(torch.load(model_filename, map_location="cpu"))

    _, test_loader, _, _ = get_dataloaders(
        None,
        test_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    evaluate(model, trainloader, test_loader, path=config["exp_path"], device=device)

    # finish wandb run
    run_id = finish_wandb()
    if run_id is not None:
        logger.info("Setting the best scores in the W&B run summary.")
        set_best_scores_in_summary(
            run_or_path=f"{wandb_entity}/{wandb_project}/{run_id}",
            primary_score="val/F1",
            scores=lambda df: [col for col in df if col.startswith("val/")],
        )

    def count_parameters(trained_model):
        return sum(p.numel() for p in trained_model.parameters() if p.requires_grad)

    if save_to_hfhub:
        try:
            num_params = count_parameters(model)
            config["mean"] = model_mean
            config["std"] = model_std
            config["params"] = np.round(num_params / 1000000, 1)
            export_model_to_huggingface_hub_from_checkpoint(
                config=config, repo_owner=args.hfhub_owner, saved_model="f1"
            )
        except Exception as e:
            print(f"Exception during export: {e}")


if __name__ == "__main__":
    train_clf()
