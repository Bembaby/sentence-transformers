from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


@pytest.mark.parametrize("prompt", [None, "beta "])
def test_sparse_static_embedding_honors_and_saves_sequence_limit(tmp_path, prompt):
    backend = Tokenizer(
        models.WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32
    )
    model = SparseEncoder(modules=[SparseStaticEmbedding(tokenizer)], device="cpu")
    model.max_seq_length = 2
    texts = ["alpha beta gamma", "alpha"]

    features = model.preprocess(texts, prompt=prompt)
    assert features["input_ids"].shape[1] == 2
    assert 4 not in features["input_ids"]
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, float(bool(prompt)), 0.0]])
    actual = model.encode_query(texts, prompt=prompt, convert_to_sparse_tensor=False)
    torch.testing.assert_close(actual, expected)

    model.save(str(tmp_path), create_model_card=False)
    restored = SparseEncoder(str(tmp_path), device="cpu", local_files_only=True)
    assert restored.max_seq_length == 2
    torch.testing.assert_close(restored.encode_query(texts, prompt=prompt, convert_to_sparse_tensor=False), expected)
