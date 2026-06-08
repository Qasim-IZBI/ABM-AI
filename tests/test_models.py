import torch
import pytest

from models import (
    MLP, MLPConfig,
    CGAN, CGANConfig,
    ImageRegressor, ImageRegressorConfig,
    MultiModalImReg, MultiModalImRegConfig,
    MultiModalCGAN, MultiModalCGANConfig,
    build_model,
)


# ── MLP ──────────────────────────────────────────────────────────────────────

def test_mlp_forward():
    model = MLP(MLPConfig(n_inputs=2, n_outputs=6, hidden_dims=[64, 32]))
    out = model(torch.randn(4, 2))
    assert out.shape == (4, 6)


def test_mlp_config_dict():
    model = MLP(MLPConfig(n_inputs=2, n_outputs=6))
    d = model.config_dict()
    assert d["n_inputs"] == 2 and d["n_outputs"] == 6


# ── cGAN — components ────────────────────────────────────────────────────────

@pytest.fixture
def small_cfg():
    return CGANConfig(n_inputs=2, noise_dim=16, image_size=64, ngf=16, ndf=16)


def test_cgan_generator_shape(small_cfg):
    model = CGAN(small_cfg)
    fake = model.G(torch.randn(3, 2))
    assert fake.shape == (3, 3, 64, 64)


def test_cgan_generator_range(small_cfg):
    model = CGAN(small_cfg)
    fake = model.G(torch.randn(3, 2))
    assert fake.min().item() >= -1.0 - 1e-5
    assert fake.max().item() <=  1.0 + 1e-5


def test_cgan_discriminator_shape(small_cfg):
    model = CGAN(small_cfg)
    out = model.D(torch.randn(3, 3, 64, 64), torch.randn(3, 2))
    assert out.shape[0] == 3 and out.shape[1] == 1


def test_cgan_custom_noise(small_cfg):
    model = CGAN(small_cfg)
    cond  = torch.randn(2, 2)
    noise = torch.zeros(2, small_cfg.noise_dim)
    out1  = model.G(cond, noise)
    out2  = model.G(cond, noise)
    assert torch.allclose(out1, out2), "Same noise should give identical output"


# ── cGAN — training interface ────────────────────────────────────────────────

@pytest.fixture
def fake_batch(small_cfg):
    return {"inputs": torch.randn(2, 2), "image": torch.randn(2, 3, 64, 64)}


def test_compute_generator_loss(small_cfg, fake_batch):
    model = CGAN(small_cfg)
    loss_G, logs, visuals = model.compute_generator_loss(fake_batch)
    assert loss_G.item() > 0
    assert {"loss_G", "adv_G", "l1"} <= logs.keys()
    assert visuals["fake"].shape == (2, 3, 64, 64)
    assert not visuals["fake"].requires_grad   # must be detached for D step


def test_compute_discriminator_loss(small_cfg, fake_batch):
    model = CGAN(small_cfg)
    _, _, visuals = model.compute_generator_loss(fake_batch)
    loss_D, logs = model.compute_discriminator_loss(fake_batch, visuals)
    assert loss_D.item() > 0
    assert "loss_D" in logs


# ── build_model ───────────────────────────────────────────────────────────────

def test_build_mlp():
    model = build_model("mlp", n_inputs=2, n_outputs=6, hidden_dims=[32])
    assert model(torch.randn(3, 2)).shape == (3, 6)


def test_build_cgan():
    model = build_model("cgan", n_inputs=2, noise_dim=16, image_size=64, ngf=16, ndf=16)
    assert model.generate(torch.randn(2, 2)).shape == (2, 3, 64, 64)


# ── ImageRegressor ────────────────────────────────────────────────────────────

@pytest.fixture
def imreg_cfg():
    return ImageRegressorConfig(n_inputs=2, image_size=64, ngf=16)


def test_imreg_forward_shape(imreg_cfg):
    model = ImageRegressor(imreg_cfg)
    out = model(torch.randn(3, 2))
    assert out.shape == (3, 3, 64, 64)


def test_imreg_output_range(imreg_cfg):
    model = ImageRegressor(imreg_cfg)
    out = model(torch.randn(3, 2))
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <=  1.0 + 1e-5


def test_imreg_deterministic(imreg_cfg):
    model = ImageRegressor(imreg_cfg)
    model.eval()
    cond = torch.randn(2, 2)
    with torch.no_grad():
        assert torch.allclose(model(cond), model(cond)), \
            "Same input must always produce the same image"


def test_imreg_has_no_discriminator(imreg_cfg):
    model = ImageRegressor(imreg_cfg)
    assert not hasattr(model, "discriminator_parameters")
    assert not hasattr(model, "compute_discriminator_loss")


def test_imreg_generator_loss(imreg_cfg):
    model = ImageRegressor(imreg_cfg)
    batch = {"inputs": torch.randn(2, 2), "image": torch.randn(2, 3, 64, 64)}
    loss, logs, visuals = model.compute_generator_loss(batch)
    assert loss.item() > 0
    assert {"loss", "l1", "mse"} <= logs.keys()
    assert visuals["fake"].shape == (2, 3, 64, 64)
    assert not visuals["fake"].requires_grad


def test_build_imreg():
    model = build_model("imreg", n_inputs=2, image_size=64, ngf=16)
    assert model.generate(torch.randn(2, 2)).shape == (2, 3, 64, 64)


# ── MultiModalImReg ───────────────────────────────────────────────────────────

@pytest.fixture
def mmimreg_cfg():
    return MultiModalImRegConfig(n_inputs=2, n_outputs=6, image_size=64, ngf=16,
                                  hidden_dims=[32])


def test_mmimreg_forward_shapes(mmimreg_cfg):
    model = MultiModalImReg(mmimreg_cfg)
    img, num = model(torch.randn(3, 2))
    assert img.shape == (3, 3, 64, 64)
    assert num.shape == (3, 6)


def test_mmimreg_image_range(mmimreg_cfg):
    model = MultiModalImReg(mmimreg_cfg)
    img, _ = model(torch.randn(3, 2))
    assert img.min().item() >= -1.0 - 1e-5
    assert img.max().item() <=  1.0 + 1e-5


def test_mmimreg_deterministic(mmimreg_cfg):
    model = MultiModalImReg(mmimreg_cfg)
    model.eval()
    cond = torch.randn(2, 2)
    with torch.no_grad():
        img1, num1 = model(cond)
        img2, num2 = model(cond)
    assert torch.allclose(img1, img2) and torch.allclose(num1, num2)


def test_mmimreg_has_no_discriminator(mmimreg_cfg):
    model = MultiModalImReg(mmimreg_cfg)
    assert not hasattr(model, "discriminator_parameters")
    assert not hasattr(model, "compute_discriminator_loss")


def test_mmimreg_generator_loss(mmimreg_cfg):
    model  = MultiModalImReg(mmimreg_cfg)
    batch  = {"inputs": torch.randn(2, 2),
              "image":  torch.randn(2, 3, 64, 64),
              "labels": torch.randn(2, 6)}
    loss, logs, visuals = model.compute_generator_loss(batch)
    assert loss.item() > 0
    assert {"loss", "l1", "mse_img", "mse_reg"} <= logs.keys()
    assert visuals["fake"].shape == (2, 3, 64, 64)
    assert visuals["pred_num"].shape == (2, 6)
    assert not visuals["fake"].requires_grad


def test_build_mmimreg():
    model = build_model("mmimreg", n_inputs=2, n_outputs=6, image_size=64, ngf=16)
    img, num = model(torch.randn(2, 2))
    assert img.shape == (2, 3, 64, 64) and num.shape == (2, 6)


# ── MultiModalCGAN ────────────────────────────────────────────────────────────

@pytest.fixture
def mmcgan_cfg():
    return MultiModalCGANConfig(n_inputs=2, n_outputs=6, noise_dim=8,
                                 image_size=64, ngf=16, ndf=16, hidden_dims=[32])


def test_mmcgan_forward_shapes(mmcgan_cfg):
    model = MultiModalCGAN(mmcgan_cfg)
    img, num = model(torch.randn(3, 2))
    assert img.shape == (3, 3, 64, 64)
    assert num.shape == (3, 6)


def test_mmcgan_image_range(mmcgan_cfg):
    model = MultiModalCGAN(mmcgan_cfg)
    img, _ = model(torch.randn(3, 2))
    assert img.min().item() >= -1.0 - 1e-5
    assert img.max().item() <=  1.0 + 1e-5


def test_mmcgan_numerical_is_deterministic(mmcgan_cfg):
    model = MultiModalCGAN(mmcgan_cfg)
    model.eval()
    cond = torch.randn(2, 2)
    with torch.no_grad():
        num1 = model.predict(cond)
        num2 = model.predict(cond)
    assert torch.allclose(num1, num2), "Regression head must be deterministic"


def test_mmcgan_image_is_stochastic(mmcgan_cfg):
    model = MultiModalCGAN(mmcgan_cfg)
    cond  = torch.randn(2, 2)
    img1  = model.generate(cond)
    img2  = model.generate(cond)
    assert not torch.allclose(img1, img2), "Different noise → different images"


def test_mmcgan_fixed_noise_is_deterministic(mmcgan_cfg):
    model = MultiModalCGAN(mmcgan_cfg)
    cond  = torch.randn(2, 2)
    noise = torch.zeros(2, mmcgan_cfg.noise_dim)
    img1  = model.generate(cond, noise)
    img2  = model.generate(cond, noise)
    assert torch.allclose(img1, img2)


def test_mmcgan_generator_loss(mmcgan_cfg):
    model  = MultiModalCGAN(mmcgan_cfg)
    batch  = {"inputs": torch.randn(2, 2),
              "image":  torch.randn(2, 3, 64, 64),
              "labels": torch.randn(2, 6)}
    loss_G, logs, visuals = model.compute_generator_loss(batch)
    assert loss_G.item() > 0
    assert {"loss_G", "adv_G", "l1", "mse_reg"} <= logs.keys()
    assert not visuals["fake"].requires_grad


def test_mmcgan_discriminator_loss(mmcgan_cfg):
    model  = MultiModalCGAN(mmcgan_cfg)
    batch  = {"inputs": torch.randn(2, 2),
              "image":  torch.randn(2, 3, 64, 64),
              "labels": torch.randn(2, 6)}
    _, _, visuals = model.compute_generator_loss(batch)
    loss_D, logs = model.compute_discriminator_loss(batch, visuals)
    assert loss_D.item() > 0 and "loss_D" in logs


def test_build_mmcgan():
    model = build_model("mmcgan", n_inputs=2, n_outputs=6,
                        noise_dim=8, image_size=64, ngf=16, ndf=16)
    img, num = model(torch.randn(2, 2))
    assert img.shape == (2, 3, 64, 64) and num.shape == (2, 6)


def test_build_model_filters_unknown_kwargs():
    # MLP should silently ignore cGAN-specific kwargs
    model = build_model("mlp", n_inputs=2, n_outputs=6,
                        hidden_dims=[32], noise_dim=999, lambda_l1=99.0)
    assert model(torch.randn(3, 2)).shape == (3, 6)
