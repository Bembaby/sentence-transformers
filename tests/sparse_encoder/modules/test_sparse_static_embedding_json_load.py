from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


@pytest.mark.parametrize("explicit_cache_folder", [False, True])
def test_sparse_static_embedding_loads_json_checkpoint(tmp_path, explicit_cache_folder):
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32
    )
    module_path = tmp_path / "query"
    module_path.mkdir()
    tokenizer.save_pretrained(module_path)
    idf_path = module_path / "idf.json"
    idf_path.write_text(json.dumps({"alpha": 2.5, "beta": 1.0}), encoding="utf-8")
    (module_path / "config.json").write_text(json.dumps({"path": str(idf_path), "frozen": True}), encoding="utf-8")

    loaded = SparseStaticEmbedding.load(
        str(tmp_path),
        subfolder="query",
        cache_folder=str(tmp_path / "cache") if explicit_cache_folder else None,
        local_files_only=True,
    )

    assert not loaded.weight.requires_grad
    model = SparseEncoder(modules=[loaded], device="cpu")
    expected = torch.tensor([[0.0, 0.0, 2.5, 1.0], [0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(model.encode_query(["alpha beta", "beta"], convert_to_sparse_tensor=False), expected)
