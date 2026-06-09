import numpy as np
import torch
import pytest

from datasets import ABMDataset, MinMaxScaler, StandardScaler


def test_dataset_loads(tmp_data_dir):
    ds = ABMDataset(tmp_data_dir, load_images=False)
    assert len(ds) == 20
    assert ds.n_inputs == 2
    assert ds.n_outputs == 6


def test_dataset_item_shapes(tmp_data_dir):
    ds = ABMDataset(tmp_data_dir, load_images=False)
    item = ds[0]
    assert item["inputs"].shape == (2,)
    assert item["labels"].shape == (6,)
    assert "image" not in item


def test_dataset_raises_when_all_images_missing(tmp_data_dir):
    import pytest
    with pytest.raises(RuntimeError, match="No images found"):
        ABMDataset(tmp_data_dir, load_images=True)


def test_minmax_scaler_roundtrip():
    raw = np.random.rand(100, 2).astype(np.float32) * 1000
    scaler = MinMaxScaler().fit(raw)
    t = torch.tensor(raw[0])
    scaled = scaler(t)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0 + 1e-5
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered.numpy(), raw[0], atol=1e-4)


def test_standard_scaler_roundtrip():
    raw = np.random.rand(100, 6).astype(np.float32) * 500
    scaler = StandardScaler().fit(raw)
    t = torch.tensor(raw[5])
    scaled = scaler(t)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered.numpy(), raw[5], atol=1e-4)


def test_raw_arrays_shape(tmp_data_dir):
    ds = ABMDataset(tmp_data_dir, load_images=False)
    assert ds.raw_inputs().shape == (20, 2)
    assert ds.raw_labels().shape == (20, 6)
