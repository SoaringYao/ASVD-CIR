import argparse
import importlib.util
import io
import random
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from asvd import compress, load, save


PROJECTIONS = {
    ("WindowAttention", "qkv"), ("WindowAttention", "proj"),
    ("OCAB", "qkv"), ("OCAB", "proj"),
    ("Attention", "qkv"), ("Attention", "project_out"),
    ("FeedForward", "project_in"), ("FeedForward", "project_out"),
}


def promptcir_selector(model):
    def select(name, layer):
        parent_name, _, child = name.rpartition(".")
        parent_type = type(model.get_submodule(parent_name)).__name__
        return ((parent_type, child) in PROJECTIONS or
                "prompt" in name.lower() and
                parent_type == "Mlp" and child in ("fc1", "fc2"))
    return select


def create_model(model_file, device="cpu"):
    path = Path(model_file).resolve()
    spec = importlib.util.spec_from_file_location("promptcir_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calculate_mask = module.calculate_mask
    module.calculate_mask = cache(lambda size: calculate_mask(size).to(device))
    return module.MyModel(decoder=True)


def image_tensor(image):
    pixels = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().div_(255)


def calibration_images(directory, device, limit, image_size, seed):
    if limit < 1 or image_size < 32 or image_size % 32:
        raise ValueError("samples must be positive and image-size a positive multiple of 32.")
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in
                   (".png", ".jpg", ".jpeg", ".bmp"))
    if not paths:
        raise ValueError(f"No calibration images found in {directory}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    for path in paths[:limit]:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=rng.randint(10, 70))
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                yield image_tensor(compressed).to(device)


@torch.inference_mode()
def capture(model, sample):
    """Capture a fixed tile and reuse its output buffer on replay."""
    static_input = sample.clone(memory_format=torch.contiguous_format)
    stream = torch.cuda.Stream(device=sample.device)
    stream.wait_stream(torch.cuda.current_stream(sample.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(static_input)
    torch.cuda.current_stream(sample.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_output = model(static_input)

    def run(x):
        static_input.copy_(x)
        graph.replay()
        return static_output

    return run


@torch.inference_mode()
def restore(model, tensor, tile=256, overlap=64, cuda_graph=False):
    _, _, height, width = tensor.shape
    if min(height, width) < 32:
        raise ValueError("Input image must be at least 32x32.")
    if tile < 32 or tile % 32 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be a positive multiple of 32 and 0 <= overlap < tile.")
    if cuda_graph and not tensor.is_cuda:
        raise ValueError("CUDA Graph inference requires a CUDA tensor.")
    padding = (0, (-width) % 32, 0, (-height) % 32)
    padded = F.pad(tensor, padding, mode="reflect") if any(padding) else tensor
    _, _, h, w = padded.shape
    tile = min(tile, h, w)
    if cuda_graph:
        model = capture(model, padded[..., :tile, :tile])
    if h == tile and w == tile:
        return model(padded)[..., :height, :width].clamp(0, 1)
    stride = tile - min(overlap, tile - 32)

    def starts(length):
        return [*range(0, length - tile, stride), length - tile]

    total = torch.zeros_like(padded)
    count = padded.new_zeros((1, 1, h, w))
    rows, columns = starts(h), starts(w)
    for y in rows:
        for x in columns:
            result = model(padded[..., y:y + tile, x:x + tile])
            total[..., y:y + tile, x:x + tile] += result
            count[..., y:y + tile, x:x + tile] += 1
    return total.div_(count)[..., :height, :width].clamp_(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Optional PromptCIR demonstration for ASVD")
    parser.add_argument("--model-file", default="PromptCIR/model.py")
    parser.add_argument("--device", default="cuda")
    commands = parser.add_subparsers(dest="command", required=True)

    conversion = commands.add_parser("compress", help="Create one compressed checkpoint")
    conversion.add_argument("--checkpoint", required=True)
    conversion.add_argument("--calibration", required=True)
    conversion.add_argument("--output", required=True)
    conversion.add_argument("--rank", type=int, default=64)
    conversion.add_argument("--samples", type=int, default=8)
    conversion.add_argument("--seed", type=int, default=325)
    conversion.add_argument("--image-size", type=int, default=128)

    inference = commands.add_parser("infer", help="Run one model on one image")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--tile", type=int, default=256)
    inference.add_argument("--overlap", type=int, default=64)
    inference.add_argument("--cuda-graph", action="store_true", help="Capture fixed tiles for CUDA replay")

    args = parser.parse_args()
    device = torch.device(args.device)
    model = create_model(args.model_file, device).to(device).eval()

    if args.command == "compress":
        original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(original, strict=True)
        selector = promptcir_selector(model)
        images = calibration_images(args.calibration, device, args.samples,
                                    args.image_size, args.seed)
        manifest = compress(model, args.rank, calibration=images,
                            selector=selector, min_saving=0.25)
        save(model, manifest, args.output)
        print(f"Saved {len(manifest)} factorized layers to {args.output}")
    else:
        load(model, args.checkpoint)
        with Image.open(args.input) as image:
            tensor = image_tensor(image).to(device)
        result = restore(model, tensor, args.tile, args.overlap, args.cuda_graph)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        array = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        Image.fromarray(array).save(output)
        print(f"Saved restored image to {output}")


if __name__ == "__main__":
    main()
