from typing import Tuple, Dict, Any
import torch
from torch import nn


def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward step of VIME during the first phase

    Args:
        model (nn.Module): An instance of VIME
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The predicted mask vector and the predicted feature vector
    """
    mask_preds, feature_preds = model(batch[0])
    return mask_preds, feature_preds


def first_phase_loss(
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
    mask: torch.Tensor,
    cat_feature_preds: torch.Tensor,
    cont_feature_preds: torch.Tensor,
    mask_preds: torch.Tensor,
    mask_loss_fn: nn.Module,
    categorical_loss_fn: nn.Module,
    continuous_loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the first phase loss of VIME

    Args:
        x_cat (torch.Tensor): The categorical input feature vector
        x_cont (torch.Tensor): The continuous input feature vector
        mask (torch.Tensor): The ground truth mask vector
        cat_feature_preds (torch.Tensor): The predicted categorical feature vector
        cont_feature_preds (torch.Tensor): The predicted continuous feature vector
        mask_preds (torch.Tensor): The predicted mask vector
        mask_loss_fn (nn.Module): The loss function for the mask estimation
        categorical_loss_fn (nn.Module): The loss function for the categorical feature reconstruction
        continuous_loss_fn (nn.Module): The loss function for the continuous feature reconstruction

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The losses for mask estimation and feature reconstruction
    """

    # Akhila added this to make it so that the categorical feature gets and output layer
    
    # I should be using sigmoid here and adding num_class neurons in the outpout for the categorical features, 
    # But that needs more effort, and since I already know that my feature here is only 2 clasess
    # I shall just use a sigmoid and BCE loss instead 
    cat_feature_preds = torch.sigmoid(cat_feature_preds)
    
    mask_loss = mask_loss_fn(mask_preds, mask)
    categorical_feature_loss = torch.tensor(0.0, device=mask_preds.device)
    continuous_feature_loss = torch.tensor(0.0, device=mask_preds.device)

    if x_cat.shape[1] > 0:
        categorical_feature_loss += categorical_loss_fn(cat_feature_preds, x_cat)
    if x_cont.shape[1] > 0:
        continuous_feature_loss += continuous_loss_fn(cont_feature_preds, x_cont)

    return mask_loss, categorical_feature_loss, continuous_feature_loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of VIME during the second phase

    Args:
        model (nn.Module): An instance of VIME
        batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

    Returns:
        torch.Tensor: The predicted label (logit)
    """
    x, _ = batch
    return model(x).squeeze()


def second_phase_loss(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    unlabeled_y_hat: torch.Tensor,
    consistency_loss_fn: nn.Module,
    task_loss_fn: nn.Module,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the second phase loss of VIME

    Args:
        y (torch.Tensor): The ground truth label
        y_hat (torch.Tensor): The predicted label
        unlabeled_y_hat (torch.Tensor): The predicted labels for the consistency regularization
        consistency_loss_fn (nn.Module): The loss function for the consistency regularization
        loss_fn (nn.Module): The loss function for the given task
        K (int): The number of perturbed samples for the consistency regularization
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The losses for the given task and consistency regularization and the ground truth labels
    """

    if y_hat.shape[0] == 0:
        task_loss = torch.tensor(0.0, device=y.device)     
    else: 
        task_loss = task_loss_fn(y_hat, y)
    #print('task_loss: ', task_loss, ' y_hat.shape: ', y_hat.shape)
    
    
    
    if len(unlabeled_y_hat) == 0:
        return task_loss, torch.tensor(0.0, device=y.device)

    consistency_len = K + 1
    
    # Select targets at intervals of consistency_len
    target = unlabeled_y_hat[::consistency_len]
    target = target.repeat_interleave(K, dim=0)

    # Select predictions that are not at intervals of consistency_len
    mask = torch.ones(len(unlabeled_y_hat), dtype=torch.bool, device=y.device)
    mask[::consistency_len] = False
    preds = unlabeled_y_hat[mask].view(-1, unlabeled_y_hat[mask].shape[-1]).squeeze()

    consistency_loss = consistency_loss_fn(preds, target)

    #print('consistency_loss: ', consistency_loss, ' unlabeled_y_hat.shape: ' , unlabeled_y_hat.shape , ' preds.shape: ', preds.shape)
    #print(y_hat.shape, unlabeled_y_hat.shape, preds.shape)
    
    return task_loss, consistency_loss
