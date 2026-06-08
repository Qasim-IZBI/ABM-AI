import dataclasses

from .mlp     import MLP,               MLPConfig
from .cgan    import CGAN,              CGANConfig
from .imreg   import ImageRegressor,    ImageRegressorConfig
from .mmimreg import MultiModalImReg,   MultiModalImRegConfig
from .mmcgan  import MultiModalCGAN,    MultiModalCGANConfig

MODELS = {
    "mlp":     (MLP,             MLPConfig),
    "cgan":    (CGAN,            CGANConfig),
    "imreg":   (ImageRegressor,  ImageRegressorConfig),
    "mmimreg": (MultiModalImReg, MultiModalImRegConfig),
    "mmcgan":  (MultiModalCGAN,  MultiModalCGANConfig),
}

# Models that generate an image output — require load_images=True during training
IMAGE_GENERATION_MODELS = {"cgan", "imreg", "mmcgan", "mmimreg"}

# Models that produce a numerical output — require label scaler during training
NUMERICAL_PREDICTION_MODELS = {"mlp", "mmcgan", "mmimreg"}


def build_model(name: str, n_inputs: int, n_outputs: int = 0, **kwargs):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS)}")
    cls, cfg_cls = MODELS[name]
    valid    = {f.name for f in dataclasses.fields(cfg_cls)}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    cfg = cfg_cls(n_inputs=n_inputs, n_outputs=n_outputs, **filtered)
    return cls(cfg)


__all__ = [
    "MLP", "MLPConfig",
    "CGAN", "CGANConfig",
    "ImageRegressor", "ImageRegressorConfig",
    "MultiModalImReg", "MultiModalImRegConfig",
    "MultiModalCGAN", "MultiModalCGANConfig",
    "MODELS", "IMAGE_GENERATION_MODELS", "NUMERICAL_PREDICTION_MODELS", "build_model",
]
