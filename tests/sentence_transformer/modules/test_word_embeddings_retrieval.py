from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings


@pytest.mark.parametrize("method", ["encode", "encode_query", "encode_document"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_hf_word_embeddings_retrieval_methods(method, batch_size):
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    words = WordEmbeddings(tokenizer, torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    model = SentenceTransformer(modules=[words, Pooling(2)], device="cpu")

    embeddings = getattr(model, method)(["alpha beta", "beta"], batch_size=batch_size, convert_to_tensor=True)

    torch.testing.assert_close(embeddings, torch.tensor([[0.5, 0.5], [0.0, 1.0]]))
