from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import EmbedDistillLoss, MSELoss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("distance_metric", ["mse", "l2", "cosine", "MSELoss"])
def test_projection_accepts_student_precision(
    word_embeddings_model: SentenceTransformer, dtype: torch.dtype, distance_metric: str
) -> None:
    model = word_embeddings_model.to(dtype=dtype)
    model[0].emb_layer.weight.requires_grad_(True)
    if distance_metric == "MSELoss":
        loss = MSELoss(model, projection_dim=4)
    else:
        loss = EmbedDistillLoss(model, distance_metric=distance_metric, projection_dim=4)
    assert loss.projection.weight.dtype == torch.float32

    features = model.preprocess(["hello world", "sentence transformers"])
    labels = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    value = loss([features], labels)
    assert value.dtype == torch.float32
    assert torch.isfinite(value)
    value.backward()

    for parameter in (model[0].emb_layer.weight, loss.projection.weight, loss.projection.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
