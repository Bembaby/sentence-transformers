from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import (
    BatchAllTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    ContrastiveTensionLoss,
    ContrastiveTensionLossInBatchNegatives,
)
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset

if not is_training_available():
    pytest.skip('Sentence Transformers was not installed with the ["train"] extra.', allow_module_level=True)


@pytest.mark.parametrize(
    "loss_cls", [BatchAllTripletLoss, BatchHardTripletLoss, BatchHardSoftMarginTripletLoss, BatchSemiHardTripletLoss]
)
@pytest.mark.parametrize("multi_dataset", [False, True])
def test_model_init_trains_the_returned_model(
    word_embeddings_model: SentenceTransformer, tmp_path, loss_cls, multi_dataset: bool
) -> None:
    word_embeddings_model[0].emb_layer.weight.requires_grad_(True)
    created = []

    def model_init():
        model = deepcopy(word_embeddings_model)
        created.append((model, model[0].emb_layer.weight.detach().clone()))
        return model

    dataset = Dataset.from_dict({"text": ["hello", "world", "sentence", "transformers"], "label": [0, 0, 1, 1]})
    trainer = SentenceTransformerTrainer(
        model_init=model_init,
        loss={"first": loss_cls, "second": loss_cls} if multi_dataset else loss_cls,
        train_dataset={"first": dataset, "second": dataset} if multi_dataset else dataset,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=4,
            learning_rate=0.1,
            optim="sgd",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            dataloader_pin_memory=False,
        ),
    )
    trainer.train()

    losses = list(trainer.loss.values()) if multi_dataset else [trainer.loss]
    assert all(loss.sentence_embedder is trainer.model for loss in losses)
    assert trainer.model is created[-1][0]
    assert not torch.equal(trainer.model[0].emb_layer.weight, created[-1][1])
    for old_model, initial_weight in created[:-1]:
        torch.testing.assert_close(old_model[0].emb_layer.weight, initial_weight, rtol=0, atol=0)


@pytest.mark.parametrize("loss_cls", [ContrastiveTensionLoss, ContrastiveTensionLossInBatchNegatives])
def test_contrastive_tension_model_init_trains_the_returned_model(
    word_embeddings_model: SentenceTransformer, tmp_path, loss_cls
) -> None:
    word_embeddings_model[0].emb_layer.weight.requires_grad_(True)
    created = []

    def model_init():
        model = deepcopy(word_embeddings_model)
        created.append((model, model[0].emb_layer.weight.detach().clone()))
        return model

    columns = {"text": ["hello", "world", "sentence", "transformers"]}
    if loss_cls is ContrastiveTensionLoss:
        columns["text_pair"] = ["hello", "sentence", "world", "transformers"]
    trainer = SentenceTransformerTrainer(
        model_init=model_init,
        loss=loss_cls,
        train_dataset=Dataset.from_dict(columns),
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=4,
            learning_rate=0.1,
            optim="sgd",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            dataloader_pin_memory=False,
        ),
    )
    independent_model = trainer.loss.model1
    independent_weight = independent_model[0].emb_layer.weight.detach().clone()

    trainer.train()

    assert trainer.loss.model2 is trainer.model
    assert trainer.model is created[-1][0]
    assert not torch.equal(trainer.model[0].emb_layer.weight, created[-1][1])
    for old_model, initial_weight in created[:-1]:
        torch.testing.assert_close(old_model[0].emb_layer.weight, initial_weight, rtol=0, atol=0)
    assert trainer.loss.model1 is independent_model
    assert independent_model[0].emb_layer.weight.data_ptr() != trainer.model[0].emb_layer.weight.data_ptr()
    assert not torch.equal(independent_model[0].emb_layer.weight, independent_weight)


def test_triplet_loss_uses_the_wrapped_model(word_embeddings_model: SentenceTransformer, tmp_path) -> None:
    class CountingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            return self.module(features)

    trainer = SentenceTransformerTrainer(
        model=word_embeddings_model,
        loss=BatchHardTripletLoss(word_embeddings_model),
        args=SentenceTransformerTrainingArguments(output_dir=str(tmp_path), report_to="none"),
    )
    wrapped = CountingWrapper(word_embeddings_model)
    trainer.model_wrapped = wrapped
    inputs = trainer.data_collator(
        [
            {"text": text, "label": label}
            for text, label in zip(["hello", "world", "sentence", "transformers"], [0, 0, 1, 1])
        ]
    )

    value = trainer.compute_loss(wrapped, inputs)
    assert torch.isfinite(value)
    assert trainer.loss.sentence_embedder is wrapped
    assert wrapped.calls == 1


@pytest.mark.parametrize("model_attribute", ["model", "sentence_embedder", "model2"])
def test_replacing_primary_loss_model_preserves_secondary_models(
    word_embeddings_model: SentenceTransformer, model_attribute: str
) -> None:
    loss = torch.nn.Module()
    setattr(loss, model_attribute, word_embeddings_model)
    loss.guide = deepcopy(word_embeddings_model)
    loss.model1 = deepcopy(word_embeddings_model)
    guide, independent_encoder = loss.guide, loss.model1
    replacement = deepcopy(word_embeddings_model)
    trainer = SentenceTransformerTrainer.__new__(SentenceTransformerTrainer)

    trainer.override_model_in_loss(loss, replacement)

    assert getattr(loss, model_attribute) is replacement
    assert loss.guide is guide
    assert loss.model1 is independent_encoder
