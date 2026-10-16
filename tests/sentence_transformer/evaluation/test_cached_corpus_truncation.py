from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.modules import BoW


@pytest.mark.parametrize("truncate_on", ["evaluator", "model"])
@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("already_truncated", [False, True])
@pytest.mark.parametrize("corpus_chunk_size", [1, 3])
def test_cached_corpus_obeys_truncation(truncate_on, as_numpy, already_truncated, corpus_chunk_size):
    model = SentenceTransformer(modules=[BoW(["alpha", "beta", "gamma"])], device="cpu")
    corpus = {"a": "alpha gamma", "b": "beta", "c": "alpha alpha beta"}
    cached = model.encode_document(list(corpus.values()), convert_to_tensor=True)
    if already_truncated:
        cached = cached[:, :2]
    if as_numpy:
        cached = cached.numpy()
    original = cached.copy() if as_numpy else cached.clone()

    if truncate_on == "model":
        model.truncate_dim = 2
    evaluator = InformationRetrievalEvaluator(
        queries={"q": "alpha gamma"},
        corpus=corpus,
        relevant_docs={"q": {"a"}},
        truncate_dim=2 if truncate_on == "evaluator" else None,
        corpus_chunk_size=corpus_chunk_size,
        write_csv=False,
    )
    expected = evaluator(model)
    actual = evaluator(model, corpus_embeddings=cached)

    assert actual == pytest.approx(expected)
    assert actual["cosine_accuracy@1"] == 1.0
    if as_numpy:
        np.testing.assert_array_equal(cached, original)
    else:
        torch.testing.assert_close(cached, original)
