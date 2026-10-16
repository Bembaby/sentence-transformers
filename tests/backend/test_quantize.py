from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from transformers import BertConfig, BertModel, BertTokenizer

pytest.importorskip("optimum.intel.openvino")
pytest.importorskip("nncf")
pytest.importorskip("datasets")

from optimum.intel.openvino import OVQuantizationConfig, OVQuantizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.backend import export_static_quantized_openvino_model
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer


@pytest.mark.parametrize("as_dict", [True, False], ids=["dict", "config"])
def test_static_openvino_quantization_config(tmp_path: Path, monkeypatch, as_dict: bool) -> None:
    model_path = tmp_path / "model"
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "apple", "pear"])}
    BertTokenizer(vocab=vocab).save_pretrained(model_path)
    BertModel(
        BertConfig(
            vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16
        )
    ).save_pretrained(model_path)
    transformer = Transformer(str(model_path), backend="openvino", model_kwargs={"export": True})
    model = SentenceTransformer(modules=[transformer, Pooling(8)], device="cpu")
    output_path = tmp_path / "quantized"
    model.save_pretrained(output_path)

    calibration_path = tmp_path / "calibration"
    calibration_path.mkdir()
    (calibration_path / "train.jsonl").write_text(
        "\n".join(json.dumps({"sentence": text}) for text in ["apple pear", "pear apple", "apple", "pear"]),
        encoding="utf-8",
    )

    get_calibration_dataset = OVQuantizer.get_calibration_dataset
    sample_counts = []

    def record_calibration_dataset(self, *args, **kwargs):
        dataset = get_calibration_dataset(self, *args, **kwargs)
        sample_counts.append(len(dataset))
        return dataset

    monkeypatch.setattr(OVQuantizer, "get_calibration_dataset", record_calibration_dataset)
    config = {"num_samples": 2, "bits": 8, "weight_only": False} if as_dict else OVQuantizationConfig(num_samples=2)
    export_static_quantized_openvino_model(
        model,
        config,
        str(output_path),
        dataset_name=str(calibration_path),
        dataset_config_name="default",
        dataset_split="train",
        column_name="sentence",
    )

    assert sample_counts == [2]
    file_name = "openvino_model_qint8_quantized.xml"
    assert (output_path / "openvino" / file_name).is_file()
    assert (output_path / "openvino" / Path(file_name).with_suffix(".bin")).is_file()
    reloaded = SentenceTransformer(
        str(output_path), backend="openvino", model_kwargs={"file_name": file_name}, device="cpu"
    )
    embeddings = reloaded.encode(["apple", "pear"])
    assert embeddings.shape == (2, 8)
    assert np.isfinite(embeddings).all()
