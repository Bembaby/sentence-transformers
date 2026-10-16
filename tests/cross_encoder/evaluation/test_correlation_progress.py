from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator


@pytest.mark.parametrize("show_progress_bar", [False, True])
def test_correlation_evaluator_respects_progress_setting(tmp_path, capsys, show_progress_bar):
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    tokenizer.save_pretrained(tmp_path)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=4,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=1,
        )
    ).save_pretrained(tmp_path)
    model = CrossEncoder(str(tmp_path), device="cpu", local_files_only=True)
    evaluator = CrossEncoderCorrelationEvaluator(
        [("alpha", "beta"), ("beta", "alpha"), ("alpha beta", "beta alpha")],
        [0.0, 0.5, 1.0],
        batch_size=1,
        show_progress_bar=show_progress_bar,
        write_csv=False,
    )
    capsys.readouterr()

    evaluator(model)

    assert ("Batches:" in capsys.readouterr().err) is show_progress_bar
