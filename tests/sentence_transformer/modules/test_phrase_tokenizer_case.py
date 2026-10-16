from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import PhraseTokenizer


@pytest.mark.parametrize("do_lower_case", [False, True])
def test_phrase_tokenizer_applies_configured_case_before_vocabulary_lookup(tmp_path, do_lower_case):
    pytest.importorskip("nltk")
    vocab = ["PADDING_TOKEN", "Cat", "cat", "New_York", "new_york"]
    tokenizer = PhraseTokenizer(vocab=vocab, stop_words=[], do_lower_case=do_lower_case)
    expected_ids = [2, 4] if do_lower_case else [1, 3]
    assert tokenizer.tokenize("Cat New York") == expected_ids

    words = WordEmbeddings(tokenizer, torch.eye(len(vocab)))
    model = SentenceTransformer(modules=[words, Pooling(len(vocab))], device="cpu")
    expected = torch.eye(len(vocab))[expected_ids].mean(dim=0)
    torch.testing.assert_close(model.encode("Cat New York", convert_to_tensor=True), expected)

    tokenizer.save(str(tmp_path))
    restored = PhraseTokenizer.load(str(tmp_path))
    assert restored.tokenize("Cat New York") == expected_ids
