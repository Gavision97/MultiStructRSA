import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
import logging
logger = logging.getLogger(__name__)


# custom RMSE loss function
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0

    for (inputs), y in loader:
        # move all model inputes & targed (pKd) to device
        inputs = [inp.to(device) for inp in inputs]
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(*inputs)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients to prevent exploding gradients that may cause NaN/Inf values during training
        optimizer.step()

        """
        Accumulate loss weighted by batch size so that
        the final epoch loss is the true average per sample.
        """
        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / max(1, n)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for (inputs), y in loader:
        # move all model inputes & targed (pKd) to device
        inputs = [inp.to(device) for inp in inputs]
        y = y.to(device)

        pred = model(*inputs).squeeze(1).detach().cpu().numpy()
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred)

    y_true, y_pred = np.concatenate(y_true).astype(np.float32), np.concatenate(y_pred).astype(np.float32)
    return y_true, y_pred


def safe_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics while handling potential NaN/Inf predictions.

    During training some hyperparameter configurations may produce non-finite
    values (NaN/Inf) in the model predictions due to numerical instability.
    Since sklearn metrics cannot handle such values, we first filter out
    non-finite (prediction, target) pairs.

    If all predictions are invalid, rmse is set to +inf so the epoch is treated
    as very poor by early stopping while preventing the training job from
    crashing.

    Returns:
    -   p : float - Pearson correlation
    -   s : float - Spearman correlation
    -   rmse : float - Root Mean Squared Error
    -   r2 : float - R² score
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan, np.nan, np.inf, np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    p = pearsonr(y_true, y_pred)[0]
    s = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return p, s, rmse, r2 


# stratified CV (same object used in RSAPred & DeepRSMA)
class regressor_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=10,random_state=0,strategy='quantile'):
        self.group_count=group_count
        self.strategy=strategy
        self.cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)  #Added shuffle here
        self.cv=RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer=KBinsDiscretizer(n_bins=self.group_count,encode='ordinal',strategy=self.strategy)  
            
    def split(self,X,y,groups=None):
        kgroups=self.discretizer.fit_transform(y[:,None])[:,0]
        return self.cv.split(X,kgroups,groups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)
    

class EarlyStopping:
    def __init__(self, mode="min", patience=20, warm_up_epochs=30, delta=0.0, verbose=False):
        """
        mode: "min" for RMSE/loss, "max" for Pearson/AUC
        """
        self.mode = mode
        self.patience = patience
        self.warm_up_epochs = warm_up_epochs
        self.delta = delta
        self.verbose = verbose

        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = -1
        self.no_improvement = 0
        self.stop_training = False
        self.best_metrics = [float("-inf"), float("-inf"), float("-inf")]  # p, s, r2

    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.delta
        else:  # "max"
            return score > self.best_score + self.delta

    def check_early_stop(self, score, metrics, train_loss, epoch):
        if self._is_improvement(score):
            self.best_score = score
            self.best_metrics[0], self.best_metrics[1], self.best_metrics[2] = metrics[0], metrics[1], metrics[2]
            self.best_epoch = epoch
            self.no_improvement = 0
            logger.info(f"NEW BEST @ epoch {epoch+1}: val loss={self.best_score:.4f}, "
                    f"ppc={self.best_metrics[0]:.4f}, ssc={self.best_metrics[1]:.4f}, r2={self.best_metrics[2]:.4f}; train loss={train_loss:.4f}")
        else:
            if epoch < self.warm_up_epochs:
                return
            self.no_improvement += 1
            if self.no_improvement >= self.patience:
                self.stop_training = True
                if self.verbose:
                    logger.info(f"Early stop at {epoch+1}. Best={self.best_score} @ {self.best_epoch+1}")


def plot_train_val_loss(
    train_losses,
    val_losses,
    save_path=None,
    title="Loss Curve"
):
    steps = range(len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", linewidth=2, color="blue")
    plt.plot(steps, val_losses, label="Validation Loss", linewidth=2, color="green")

    plt.title(title, fontsize=16)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
