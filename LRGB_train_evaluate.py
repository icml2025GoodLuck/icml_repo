import torch
from models.topo_concat import TopoConcat
from models.topo_only import TopoTest
from torch import nn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from torchmetrics.utilities import reduce
import numpy as np

# LRGB evaluation metrics (Paper: Where Did the Gap Go?)
EPS = 1e-5

"""
Evaluation functions from OGB.
https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
"""
def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

def eval_acc(y_true, y_pred):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return {'acc': sum(acc_list) / len(acc_list)}

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return {'ap': sum(ap_list) / len(ap_list)}


# regression
def pearsonr(preds: torch.Tensor, target: torch.Tensor,
             reduction: str = "elementwise_mean") -> torch.Tensor:
    r"""
    Computes the pearsonr correlation.

    Parameters:
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns:
        Tensor with the pearsonr

    !!! Example
        ``` python linenums="1"
        x = torch.tensor([0., 1, 2, 3])
        y = torch.tensor([0., 1, 2, 2])
        pearsonr(x, y)
        >>> tensor(0.9439)
        ```
    """

    preds, target = preds.to(torch.float32), target.to(torch.float32)

    shifted_x = preds - torch.mean(preds, dim=0)
    shifted_y = target - torch.mean(target, dim=0)
    sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
    sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

    pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + EPS)
    pearson = torch.clamp(pearson, min=-1, max=1)
    pearson = reduce(pearson, reduction=reduction)
    return pearson


def _get_rank(values):

    arange = torch.arange(values.shape[0],
                          dtype=values.dtype, device=values.device)

    val_sorter = torch.argsort(values, dim=0)
    val_rank = torch.empty_like(values)
    if values.ndim == 1:
        val_rank[val_sorter] = arange
    elif values.ndim == 2:
        for ii in range(val_rank.shape[1]):
            val_rank[val_sorter[:, ii], ii] = arange
    else:
        raise ValueError(f"Only supports tensors of dimensions 1 and 2, "
                         f"provided dim=`{values.ndim}`")

    return val_rank


def spearmanr(preds: torch.Tensor, target: torch.Tensor,
              reduction: str = "elementwise_mean") -> torch.Tensor:
    r"""
    Computes the spearmanr correlation.

    Parameters:
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns:
        Tensor with the spearmanr

    !!! Example
        x = torch.tensor([0., 1, 2, 3])
        y = torch.tensor([0., 1, 2, 1.5])
        spearmanr(x, y)
        tensor(0.8)
    """

    spearman = pearsonr(_get_rank(preds), _get_rank(target), reduction=reduction)
    return spearman


METRICS_REGRESSION = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "pearsonr": pearsonr,
    "spearmanr": spearmanr,
}


class GPUMemoryTracker:
    def __init__(self):
        self.batch_mem_log = []

    def record(self):
        mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        self.batch_mem_log.append(mem)

    def save_log(self, save_path="log/gpu_memory_log.txt"):
        with open(save_path, "w") as f:
            for i, mem in enumerate(self.batch_mem_log):
                f.write(f"Batch {i+1}: {mem:.2f} MB\n")

    def plot(self, save_path="log/gpu_memory_plot.png"):
        plt.figure(figsize=(8, 4))
        plt.plot(self.batch_mem_log, marker='o')
        plt.title("GPU Memory Usage per Batch")
        plt.xlabel("Batch Index")
        plt.ylabel("Memory (MB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        

# Define the training and evaluation process
def LRGB_train_and_evaluate(dataset_name, trial_num, train_loader, val_loader, test_loader, 
                        input_dim, output_dim, config, class_type='concat', num_iteration=1,
                        task_type='classification'):
    """
    Train and evaluate the specified model.

    Parameters:
    - dataset_name: Name of the dataset used
    - trial_num: Number of current trial
    - train_loader: DataLoader for the training dataset.
    - val_loader: DataLoader for the validation dataset.
    - test_loader: DataLoader for the test dataset.
    - input_dim: Dimension of the input features.
    - output_dim: Dimension of the output features.
    - config: Dictionary containing configuration parameters.
    - class_type: Type of model class to use ('concat' for TopoConcat, 'test' for TopoTest).
    - num_iteration: Iteration number for the model saving.

    Returns:
    - val_acc: Accuracy on the validation set.
    - val_loss: Loss on the validation set.
    - test_acc: Accuracy on the test set.
    - test_loss: Loss on the test set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, optimizer, and criterion
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': config['hidden_dim'],
        'output_dim': output_dim,
        'use_topo_features': config['use_topo_features'],
        'model_type': config['model_type'],
        'topo_feature_dim': train_loader.dataset[0][1].shape[0],
        'use_cnn': config.get('use_cnn', True),
        'task_type': task_type
    }

    # Choose the model based on the specified type
    if class_type == 'concat':
        model = TopoConcat(**model_params).to(device)
    elif class_type == 'test':
        model = TopoTest(**model_params).to(device)
    else:
        raise ValueError(f"Unsupported class type: {class_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = (
        nn.MSELoss() if task_type == 'regression'
        else nn.BCEWithLogitsLoss() if task_type == 'multilabel'
        else nn.CrossEntropyLoss()
    )

    # Count and print model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    print(f"[Model] Number of learnable parameters: {num_params}", flush=True)

    
    # Optional GPU memory tracking
    tracker = GPUMemoryTracker()

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        for data, topo_feature in train_loader:
            data, topo_feature = data.to(device), topo_feature.to(device)
            optimizer.zero_grad()
            out = model(data, data.batch, topo_feature)
            if task_type == 'multilabel':
                loss = criterion(out, data.y.to(torch.float))
            else:
                loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                tracker.record()
    
    if test_loader and torch.cuda.is_available():
        mem_log_dir = f"log/gpu_profile/{dataset_name}/{class_type}/{model_params['model_type']}_topo_{model_params['use_topo_features']}_cnn_{model_params['use_cnn']}/h{config['hidden_dim']}_lr{config['lr']}_wd{config['weight_decay']}/trial_{trial_num + 1}"
        os.makedirs(mem_log_dir, exist_ok=True)
        tracker.save_log(os.path.join(mem_log_dir, "mem_log.txt"))
        tracker.plot(os.path.join(mem_log_dir, "mem_curve.png"))
        with open(os.path.join(mem_log_dir, "param_count.txt"), "w") as f:
            f.write(f"{num_params}\n")

        print(f"[Memory Tracking] GPU memory curve saved to {mem_log_dir}", flush=True)

    # Evaluate on the validation set if provided
    if val_loader:
        val_acc, val_loss, val_f1, val_auc, val_regression_metrics = test(val_loader, model, "validate", task_type=task_type)
    else:
        val_acc, val_loss, val_f1, val_auc, val_regression_metrics = (None,) * 5

    # Evaluate on the test set if provided
    if test_loader:
        test_acc, test_loss, test_f1, test_auc, test_regression_metrics = test(test_loader, model, "test", task_type=task_type)

        # Save the final test model
        model_save_path = f"log/models/{dataset_name}/{class_type}/{model_params['model_type']}_topo_{model_params['use_topo_features']}_cnn_{model_params['use_cnn']}/h{config['hidden_dim']}_lr{config['lr']}_wd{config['weight_decay']}/trial_{trial_num + 1}.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    else:
        test_acc, test_loss, test_f1, test_auc, test_regression_metrics = (None,) * 5

    return {
        "num_params": num_params,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "val_regression_metrics": val_regression_metrics,
        "test_regression_metrics": test_regression_metrics
    }



def test(loader, model, set_type="validate", task_type="multilabel"):
    """
    Evaluate the model on the given DataLoader.

    Parameters:
    - loader: DataLoader for the dataset.
    - model: The model to evaluate.
    - set_type: 'validate' or 'test'
    - task_type: 'multilabel' or 'regression'

    Returns:
    - multilabel: acc, avg_loss, None, auc, {"ap": ap}
    - regression: None, loss, None, None, metrics_dict
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data, topo_feature in loader:
            data, topo_feature = data.to(device), topo_feature.to(device)
            out = model(data, data.batch, topo_feature)

            if task_type == "multilabel":
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(out, data.y.to(torch.float))
                total_loss += loss.item()

                probs = torch.sigmoid(out)
                preds = (probs > 0.5).int()

                all_preds.append(preds)
                all_targets.append(data.y.int())
                all_probs.append(probs)

            elif task_type == "regression":
                criterion = nn.MSELoss()
                loss = criterion(out.squeeze(), data.y.squeeze())
                total_loss += loss.item()

                all_preds.append(out.view(-1))
                all_targets.append(data.y.view(-1))

    avg_loss = total_loss / len(loader)

    # === Multilabel Evaluation ===
    if task_type == "multilabel":
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probs = torch.cat(all_probs, dim=0)

        # === Use OGB-style metric evaluators ===
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        probs_np = all_probs.cpu().numpy()

        acc = eval_acc(targets_np, preds_np)['acc']
        auc = eval_rocauc(targets_np, probs_np)['rocauc']
        ap = eval_ap(targets_np, probs_np)['ap']

        print(f"{set_type.capitalize()} Loss: {avg_loss:.6f}")
        print(f"{set_type.capitalize()} ACC: {acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

        return acc, avg_loss, None, auc, {"ap": ap}

    # === Regression Evaluation ===
    elif task_type == "regression":
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mae = mean_absolute_error(preds, targets).item()
        mse = mean_squared_error(preds, targets).item()
        rmse = mean_squared_error(preds, targets, squared=False).item()
        pearson = pearsonr(preds, targets).item()
        spearman = spearmanr(preds, targets).item()

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "pearsonr": pearson,
            "spearmanr": spearman
        }

        print(f"{set_type.capitalize()} Loss: {avg_loss:.6f}")
        for k, v in metrics.items():
            print(f"{set_type.capitalize()} {k.upper()}: {v:.4f}")

        return None, avg_loss, None, None, metrics
