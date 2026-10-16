from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.multi_vector_encoder.interpretability import real_query_token_slice
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask


@pytest.fixture
def local_query_model(tmp_path):
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "query",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=32,
    )
    tokenizer.save_pretrained(tmp_path)
    BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
        )
    ).save_pretrained(tmp_path)
    transformer = Transformer(str(tmp_path))
    return MultiVectorEncoder(modules=[transformer, MultiVectorMask()], device="cpu")


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("attend", [False, True])
@pytest.mark.parametrize("prompt", [None, "query "])
def test_query_slice_excludes_left_padding_and_prefix_tokens(local_query_model, padding_side, attend, prompt):
    model = local_query_model
    model.tokenizer.padding_side = padding_side
    model[0].query_expansion = {"strategy": "fixed", "attend": attend, "length": 8}
    if prompt:
        model.prompts["query"] = prompt
    query = "alpha beta"
    token_slice = real_query_token_slice(model, query)
    output = model.encode_query(query, output_value=None)
    expected_ids = model.tokenizer(query, add_special_tokens=False)["input_ids"]

    assert output["input_ids"][token_slice].tolist() == expected_ids
    content_positions = torch.isin(output["input_ids"], torch.tensor(expected_ids))
    embeddings = model.encode_query(query)
    torch.testing.assert_close(embeddings[token_slice], output["token_embeddings"][content_positions])


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("expand", [False, True])
def test_query_slice_contains_only_untruncated_content(local_query_model, padding_side, expand):
    model = local_query_model
    model.tokenizer.padding_side = padding_side
    model.prompts["query"] = "query "
    if expand:
        model[0].query_expansion = {"strategy": "fixed", "length": 6}
    query = "alpha beta gamma delta epsilon zeta"
    token_slice = real_query_token_slice(model, query)
    output = model.encode_query(query, output_value=None)
    expected_text = "alpha beta gamma" if expand else query
    expected_ids = model.tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    assert output["input_ids"][token_slice].tolist() == expected_ids


def test_query_slice_with_flattened_preprocessing(local_query_model):
    from transformers import DataCollatorWithFlattening

    model = local_query_model
    transformer = model[0]
    transformer.can_flatten_inputs = True
    transformer._flatten_position_offset = 0
    transformer.data_collator = DataCollatorWithFlattening(
        return_seq_idx=True, return_flash_attn_kwargs=True, return_position_ids=True
    )
    query = "alpha beta"
    features = transformer.preprocess([query], task="query")
    assert "attention_mask" not in features

    token_slice = real_query_token_slice(model, query)

    assert (
        features["input_ids"][0][token_slice].tolist() == model.tokenizer(query, add_special_tokens=False)["input_ids"]
    )
