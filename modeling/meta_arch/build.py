import torch
from fvcore.common.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
def build_model(cfg):
    """
        build the whole model architecture, defined by configs.MODEL.META_ARCHITECTURE
    """
    name = cfg.training.model.meta_arch
    model = META_ARCH_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg.training.model.device))

    return model