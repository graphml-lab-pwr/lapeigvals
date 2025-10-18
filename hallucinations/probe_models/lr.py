from typing import Any

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from torch import Tensor

from hallucinations.probe_models.commons import get_pca_dim

DEFAULT_MAX_ITER = 2_000
DEFAULT_RANDOM_SEED = 42


def train_logistic_regression(
    features: Tensor,
    labels: Tensor,
    split: dict,
    pca_dim: int | None = None,
    use_cuda: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, Any]:
    train_features = features[split["train_idx"]].float().numpy()
    test_features = features[split["test_idx"]].float().numpy()
    train_labels = labels[split["train_idx"]].numpy()
    test_labels = labels[split["test_idx"]].numpy()

    pca_dim = get_pca_dim(pca_dim, train_features)

    assert len(train_features.shape) == 2
    assert len(test_features.shape) == 2

    if use_cuda:
        from cuml.decomposition import PCA
        from cuml.linear_model import LogisticRegression
        from lightning_fabric import seed_everything

        seed_everything(random_seed, verbose=False)
        lr_kwargs = dict(
            max_iter=max_iter,
            class_weight="balanced",
        )
        pca_kwargs = dict(
            n_components=pca_dim,
        )
    else:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        lr_kwargs = dict(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_seed,
        )
        pca_kwargs = dict(
            n_components=pca_dim,
            random_state=random_seed,
        )

    if pca_dim is not None:
        model = Pipeline(
            [
                (
                    "pca",
                    PCA(**pca_kwargs),
                ),
                (
                    "lr",
                    LogisticRegression(**lr_kwargs),
                ),
            ]
        )
    else:
        model = LogisticRegression(**lr_kwargs)

    model.fit(train_features, train_labels)

    train_proba = model.predict_proba(train_features)
    train_preds = train_proba.argmax(axis=-1)
    test_proba = model.predict_proba(test_features)
    test_preds = test_proba.argmax(axis=-1)

    if pca_dim is not None:
        num_params = +model.named_steps["lr"].coef_.size + model.named_steps["lr"].intercept_.size
    else:
        num_params = model.coef_.size + model.intercept_.size

    metadata = {
        "trainable_params": num_params,
        "random_seed": random_seed,
        "max_iter": max_iter,
        "pca_dim": pca_dim,
        "use_cuda": use_cuda,
    }

    metrics = {
        "train_auc": roc_auc_score(train_labels, train_proba[:, 1]).item(),
        "train_average_precision": average_precision_score(train_labels, train_proba[:, 1]).item(),
        "train_precision": precision_score(train_labels, train_preds),
        "train_recall": recall_score(train_labels, train_preds),
        "train_f1": f1_score(train_labels, train_preds),
        "test_auc": roc_auc_score(test_labels, test_proba[:, 1]).item(),
        "test_average_precision": average_precision_score(test_labels, test_proba[:, 1]).item(),
        "test_precision": precision_score(test_labels, test_preds),
        "test_recall": recall_score(test_labels, test_preds),
        "test_f1": f1_score(test_labels, test_preds),
    }

    return {
        "metadata": metadata,
        "metrics": metrics,
        "train_proba": train_proba.tolist(),
        "test_proba": test_proba.tolist(),
        "train_preds": train_preds.tolist(),
        "test_preds": test_preds.tolist(),
    }


def compute_auc_over_score(
    score: Tensor,
    labels: Tensor,
    split: dict[str, Tensor],
) -> dict[str, Any]:
    train_scores = score[split["train_idx"]].float().numpy()
    test_scores = score[split["test_idx"]].float().numpy()
    train_labels = labels[split["train_idx"]].numpy()
    test_labels = labels[split["test_idx"]].numpy()

    metadata = {
        "trainable_params": 0,
        "random_seed": None,
        "max_iter": None,
    }
    metrics = {
        "train_auc": roc_auc_score(train_labels, train_scores).item(),
        "train_average_precision": average_precision_score(train_labels, train_scores).item(),
        "train_precision": None,
        "train_recall": None,
        "train_f1": None,
        "test_auc": roc_auc_score(test_labels, test_scores).item(),
        "test_average_precision": average_precision_score(test_labels, test_scores).item(),
        "test_precision": None,
        "test_recall": None,
        "test_f1": None,
    }

    return {
        "metadata": metadata,
        "metrics": metrics,
        "train_preds": train_scores.tolist(),
        "test_preds": test_scores.tolist(),
    }
