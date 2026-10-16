from __future__ import annotations

import json

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings, WordWeights
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("safe_serialization", [True, False])
def test_word_weights_checkpoint_preserves_trained_embeddings(tmp_path, safe_serialization):
    vocab = ["PADDING_TOKEN", "alpha", "beta"]
    words = WordEmbeddings(
        WhitespaceTokenizer(vocab, stop_words=[]),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    weights = WordWeights(vocab, {"alpha": 1.0, "beta": 2.0})
    model = SentenceTransformer(modules=[words, weights, Pooling(2)], device="cpu")
    texts = ["alpha beta", "beta alpha alpha"]
    original = model.encode(texts, convert_to_tensor=True)

    model.train()
    optimizer = torch.optim.SGD(weights.parameters(), lr=0.5)
    loss = model(model.preprocess(texts))["sentence_embedding"][:, 0].sum()
    loss.backward()
    optimizer.step()
    expected = model.encode(texts, convert_to_tensor=True)
    assert not torch.allclose(original, expected)

    model.save(str(tmp_path), safe_serialization=safe_serialization, create_model_card=False)
    restored = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)

    torch.testing.assert_close(restored[1].emb_layer.weight, weights.emb_layer.weight)
    torch.testing.assert_close(restored.encode(texts, convert_to_tensor=True), expected)


def test_word_weights_loads_config_only_checkpoint(tmp_path):
    model = WordWeights(["alpha", "beta"], {"alpha": 2.0}, unknown_word_weight=0.5)
    (tmp_path / "config.json").write_text(json.dumps(model.get_config_dict()), encoding="utf-8")

    restored = WordWeights.load(str(tmp_path), local_files_only=True)

    torch.testing.assert_close(restored.emb_layer.weight, model.emb_layer.weight)
