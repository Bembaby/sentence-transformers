from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.base.evaluation import SequentialEvaluator
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator, TranslationEvaluator
from sentence_transformers.sentence_transformer.modules import BoW


@pytest.mark.parametrize("kind", ["binary", "translation"])
@pytest.mark.parametrize("name_kind", ["collision", "ordinary", "empty"])
def test_evaluator_name_can_match_metric_prefix(kind, name_kind):
    model = SentenceTransformer(modules=[BoW(["alpha", "beta"])], device="cpu")
    name = {"collision": "cosine" if kind == "binary" else "mean", "ordinary": "test", "empty": ""}[name_kind]
    if kind == "binary":
        evaluator = BinaryClassificationEvaluator(
            ["alpha", "beta", "alpha"], ["alpha", "beta", "beta"], [1, 1, 0], name=name, write_csv=False
        )
        metric = "cosine_ap"
    else:
        evaluator = TranslationEvaluator(["alpha", "beta"], ["alpha", "beta"], name=name, write_csv=False)
        metric = "mean_accuracy"
    expected_primary = f"{name}_{metric}" if name else metric
    sequence = SequentialEvaluator([evaluator])

    # Binary classification resets its primary metric each call; translation retains it.
    for _ in range(2):
        metrics = sequence(model)
        assert evaluator.primary_metric == expected_primary
        assert metrics[evaluator.primary_metric] == pytest.approx(1.0)
        assert metrics[sequence.primary_metric] == pytest.approx(1.0)


@pytest.fixture
def local_classifier(tmp_path):
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    tokenizer.save_pretrained(tmp_path)
    classifier = BertForSequenceClassification(
        BertConfig(
            vocab_size=4,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=1,
        )
    )
    # Tied predictions make reranked NDCG deterministic and distinct from the baseline.
    with torch.no_grad():
        classifier.classifier.weight.zero_()
        classifier.classifier.bias.zero_()
    classifier.save_pretrained(tmp_path)
    return CrossEncoder(str(tmp_path), device="cpu", local_files_only=True)


def test_prefixing_preserves_reranked_primary_metric_with_baseline_scores(local_classifier):
    evaluator = CrossEncoderRerankingEvaluator(
        [{"query": "alpha", "positive": ["alpha"], "documents": ["beta", "alpha"]}],
        at_k=1,
        name="base",
        write_csv=False,
    )
    sequence = SequentialEvaluator([evaluator])

    # Reranking prefixes model-card and returned metrics separately on every call.
    for _ in range(2):
        metrics = sequence(local_classifier)
        assert evaluator.primary_metric == "base_ndcg@1"
        assert metrics[evaluator.primary_metric] == pytest.approx(0.5)
        assert metrics["base_base_ndcg@1"] == pytest.approx(0.0)
        assert metrics[sequence.primary_metric] == pytest.approx(0.5)


def test_prefixing_disambiguates_precision_and_average_precision(local_classifier):
    evaluator = CrossEncoderClassificationEvaluator(
        [["alpha", "beta"], ["beta", "alpha"], ["alpha", "alpha"]],
        labels=[0, 1, 0],
        name="average",
        write_csv=False,
    )
    sequence = SequentialEvaluator([evaluator])

    # Classification resets its raw primary name, which also names another prefixed metric.
    for _ in range(2):
        metrics = sequence(local_classifier)
        assert evaluator.primary_metric == "average_average_precision"
        assert metrics[evaluator.primary_metric] == pytest.approx(1 / 3)
        assert metrics[sequence.primary_metric] == metrics["average_average_precision"]
