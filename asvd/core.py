from math import isfinite
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Two linear projections populated by compress or load."""

    def __init__(self, layer, rank):
        super().__init__()
        self.down = nn.Parameter(layer.weight.new_empty(rank, layer.in_features))
        self.up = nn.Parameter(layer.weight.new_empty(layer.out_features, rank))
        self.bias = nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None

    def forward(self, input):
        return F.linear(F.linear(input, self.down), self.up, self.bias)


class LowRankConv2d(nn.Module):
    """Spatial projection followed by a 1x1 projection, with zero padding."""

    def __init__(self, layer, rank):
        super().__init__()
        self.down = nn.Parameter(layer.weight.new_empty(rank, layer.in_channels, *layer.kernel_size))
        self.up = nn.Parameter(layer.weight.new_empty(layer.out_channels, rank, 1, 1))
        self.bias = nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation

    def forward(self, input):
        x = F.conv2d(input, self.down, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return F.conv2d(x, self.up, self.bias)


_LAYERS = {nn.Linear: LowRankLinear, nn.Conv2d: LowRankConv2d}
_TYPES = {"linear": LowRankLinear, "conv2d": LowRankConv2d}


def _selected(model, selector, rank, min_saving):
    selected = {}
    for name, layer in model.named_modules():
        if not name or type(layer) not in _LAYERS:
            continue
        if type(layer) is nn.Conv2d and (layer.groups != 1 or layer.padding_mode != "zeros"):
            continue
        if selector is not None and not selector(name, layer):
            continue
        original = layer.weight.numel()
        if original == 0:
            continue
        rows = layer.weight.shape[0]
        factorized = rank * (rows + original // rows)
        if factorized < original and (original - factorized) / original >= min_saving:
            selected[name] = layer
    return selected


def _channel_magnitude(x, layer):
    if type(layer) is nn.Linear:
        return x.detach().abs().reshape(-1, x.shape[-1]).mean(0, dtype=torch.float32)
    dims = (0, 2, 3) if x.ndim == 4 else (1, 2)
    return x.detach().abs().mean(dim=dims, dtype=torch.float32)


def _forward(model, batch):
    return model(batch)


def _calibrate(model, selected, samples, forward):
    stats = {}
    handles = []
    for name, layer in selected.items():
        rows = layer.weight.shape[0]
        channels = layer.in_features if type(layer) is nn.Linear else layer.in_channels
        stats[name] = [layer.weight.new_zeros(rows, dtype=torch.float32),
                       layer.weight.new_zeros(channels, dtype=torch.float32), 0]

        def hook(module, inputs, kwargs, output, record=stats[name]):
            record[0].add_(_channel_magnitude(output, module))
            input = inputs[0] if inputs else kwargs["input"]
            record[1].add_(_channel_magnitude(input, module))
            record[2] += 1

        handles.append(layer.register_forward_hook(hook, with_kwargs=True))

    training = [(module, module.training) for module in model.modules()]
    batches = 0
    try:
        model.eval()
        with torch.inference_mode():
            for batch in samples:
                forward(model, batch)
                batches += 1
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training:
            module.training = was_training
    if batches == 0:
        raise ValueError("Calibration data is empty.")
    missing = [name for name, (_, _, n) in stats.items() if n == 0]
    if missing:
        raise ValueError(f"Calibration did not execute selected layers: {missing[:8]}")
    return {name: (left / n, right / n) for name, (left, right, n) in stats.items()}


def _factors(layer, rank, magnitudes, max_ratio):
    weight = layer.weight.detach().flatten(1).float()
    left, right = magnitudes
    left = left.clamp_min(left.max() / max_ratio).add_(1e-6)
    right = right.clamp_min(right.max() / max_ratio).add_(1e-6)
    if type(layer) is nn.Conv2d:
        right = right.repeat_interleave(layer.kernel_size[0] * layer.kernel_size[1])
    weight = (left[:, None] * weight).mul_(right[None, :])
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    a = (u[:, :rank] * s[:rank]).div_(left[:, None])
    b = vh[:rank] / right[None, :]
    return a, b


def _replace(model, name, replacement):
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name), child, replacement)


@torch.no_grad()
def compress(model, rank, calibration, selector=None, forward=_forward,
             min_saving=0.0, max_ratio=1000.0):
    """Replace eligible child layers in-place with activation-weighted factors.

    Every selected layer must run on the nonempty calibration iterable.
    Inputs default to model(batch); use forward(model, batch) for other signatures.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("rank must be a positive integer.")
    if not 0 <= min_saving < 1:
        raise ValueError("min_saving must be in [0, 1).")
    if not isfinite(max_ratio) or max_ratio <= 0:
        raise ValueError("max_ratio must be finite and positive.")
    if calibration is None:
        raise ValueError("ASVD requires calibration data.")
    selected = _selected(model, selector, rank, min_saving)
    if not selected:
        raise ValueError("No eligible layers selected at this rank and min_saving.")
    stats = _calibrate(model, selected, calibration, forward)
    prepared = []
    for name, layer in selected.items():
        a, b = _factors(layer, rank, stats.pop(name), max_ratio)
        replacement = _LAYERS[type(layer)](layer, rank)
        replacement.down.copy_(b.reshape_as(replacement.down))
        replacement.up.copy_(a.reshape_as(replacement.up))
        replacement.train(layer.training)
        prepared.append((name, layer, replacement))
    manifest = []
    for name, layer, replacement in prepared:
        _replace(model, name, replacement)
        manifest.append({"name": name, "rank": rank,
                         "type": "linear" if type(layer) is nn.Linear else "conv2d"})
    return manifest


def save(model, manifest, path):
    """Save factors and the manifest needed to reconstruct the compressed model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": manifest, "state_dict": model.state_dict()}, path)


def load(model, path, map_location="cpu"):
    """Load into the matching base architecture, already on the desired device/dtype."""
    bundle = torch.load(path, map_location=map_location, weights_only=True)
    for entry in bundle["layers"]:
        layer = model.get_submodule(entry["name"])
        replacement = _TYPES[entry["type"]](layer, entry["rank"])
        replacement.train(layer.training)
        _replace(model, entry["name"], replacement)
    model.load_state_dict(bundle["state_dict"])
    return bundle["layers"]
