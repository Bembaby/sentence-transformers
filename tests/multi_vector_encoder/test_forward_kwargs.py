from __future__ import annotations

import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.base.modules import Module
from sentence_transformers.sentence_transformer.modules import WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


class ScaleTokens(Module):
    forward_kwargs = {"token_scale"}

    def forward(self, features, token_scale=1.0):
        features["token_embeddings"] = features["token_embeddings"] * token_scale
        return features

    def save(self, output_path, **kwargs):
        self.save_config(output_path)


@pytest.mark.parametrize("method", ["encode", "encode_query", "encode_document"])
@pytest.mark.parametrize("output_value", ["token_embeddings", None])
def test_encode_forwards_module_kwargs(method, output_value):
    words = WordEmbeddings(
        WhitespaceTokenizer(["PADDING_TOKEN", "alpha", "beta"], stop_words=[]),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    model = MultiVectorEncoder(modules=[words, ScaleTokens()], device="cpu")
    assert "token_scale" in model.get_model_kwargs()
    encode = getattr(model, method)
    texts = ["alpha beta", "beta"]
    baseline = encode(texts, output_value=output_value)
    scaled = encode(texts, output_value=output_value, token_scale=3.0)

    for original, actual in zip(baseline, scaled):
        if output_value is None:
            original = original["token_embeddings"]
            actual = actual["token_embeddings"]
        torch.testing.assert_close(actual, original * 3.0)
