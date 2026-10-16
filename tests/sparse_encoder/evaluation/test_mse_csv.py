from __future__ import annotations

import csv

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


def test_sparse_mse_csv_contains_numeric_values(tmp_path):
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    teacher = SparseEncoder(modules=[SparseStaticEmbedding(tokenizer, torch.ones(4))], device="cpu")
    student = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, torch.tensor([1.0, 1.0, 2.0, 2.0]))], device="cpu"
    )
    evaluator = SparseMSEEvaluator(["alpha", "beta"], ["alpha", "beta"], teacher_model=teacher)

    for epoch in [0, 1]:
        metrics = evaluator(student, output_path=str(tmp_path), epoch=epoch)
        assert metrics["negative_mse"] == pytest.approx(-25.0)

    with (tmp_path / evaluator.csv_file).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row in rows:
        assert float(row["MSE"]) == pytest.approx(-metrics["negative_mse"])
        assert float(row["active_dims"]) == 1.0
        assert float(row["sparsity_ratio"]) == 0.75
