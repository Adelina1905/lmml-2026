import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from challenge_model import (
    IMAGE_FORMAT,
    IMAGE_MODE,
    MAX_OUTPUT_BYTES,
    MAX_PIXEL_DELTA,
    MIN_TARGET_CONFIDENCE,
    TARGET_CLASS_ID,
    load_model,
    normalize_batch,
    pil_to_tensor,
    predict_tensor,
    tensor_to_rgb_image,
)

MODEL_PATH = "classifier/mobilenet_v3_large-5c1a4163.pth"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.png")
    parser.add_argument("--output", default="test-output.png")
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    torch.set_num_threads(2)
    arguments = parse_arguments()

    image = load_image(arguments.input)
    print(f"Loaded {arguments.input}: size {image.size}, mode {image.mode}")

    x = pil_to_tensor(image)
    print(f"Tensor shape: {tuple(x.shape)}, values from {x.min().item():.3f} to {x.max().item():.3f}")

    model = load_model(MODEL_PATH)
    prediction = predict_tensor(model, x)
    print(f"Model sees: {prediction.class_name} (class {prediction.class_id}), "
          f"confidence {prediction.confidence:.4f}")
    print(f"Moped confidence: {prediction.target_confidence:.6f}")
    
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    
    for name, parameter in model.named_parameters():
        print(name, tuple(parameter.shape))
            
    target = torch.tensor([TARGET_CLASS_ID])
    delta = torch.zeros_like(x, requires_grad=True)

    logits = model(normalize_batch(x + delta))
    loss = F.cross_entropy(logits, target)
    loss.backward()
    
    gradient = delta.grad
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient shape: {tuple(gradient.shape)}")
    print(f"Gradient of first 5 red values: {gradient[0, 0, 0, :5]}")
    print(f"Values to push up:   {(gradient < 0).float().mean().item():.0%}")
    print(f"Values to push down: {(gradient > 0).float().mean().item():.0%}")

    budget = 1 / 255
    step_size = budget / 4

    for step in range(200):
        with torch.no_grad():
            rounded = torch.round(delta * 255) / 255
            rounded = (x + rounded).clamp(0, 1) - x
            moped = model(normalize_batch(x + rounded)).softmax(dim=1)[0, TARGET_CLASS_ID].item()
        print(f"step {step:3d}   moped after rounding {moped:.4f}")

        if moped >= 0.995:
            print("Reached the target!")
            break

        logits = model(normalize_batch(x + delta))
        loss = F.cross_entropy(logits, target)
        delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-budget, budget)
            delta.copy_((x + delta).clamp(0, 1) - x)

    result = tensor_to_rgb_image(x + rounded)
    result.save(arguments.output, format=IMAGE_FORMAT)
    print(f"Saved {arguments.output}")

    with Image.open(arguments.output) as saved:
        format_ok = (
            saved.format == IMAGE_FORMAT
            and saved.mode == IMAGE_MODE
            and saved.size == image.size
            and os.path.getsize(arguments.output) <= MAX_OUTPUT_BYTES
        )
        saved_pixels = np.array(saved, dtype=np.int16)
        final = predict_tensor(model, pil_to_tensor(saved))

    original_pixels = np.array(image, dtype=np.int16)
    max_change = int(np.abs(saved_pixels - original_pixels).max())

    checks = {
        "image_valid": format_ok,
        "pixel_budget": max_change <= MAX_PIXEL_DELTA,
        "predicted_class": final.class_id == TARGET_CLASS_ID,
        "target_confidence": final.target_confidence >= MIN_TARGET_CONFIDENCE,
    }
    print(f"Saved image is: {final.class_name}, moped confidence {final.target_confidence:.4f}")
    print(f"Maximum pixel change: {max_change}")
    for name, passed in checks.items():
        print(f"  {name:18} {'PASS' if passed else 'FAIL'}")
    print(f"All checks pass: {all(checks.values())}")

if __name__ == "__main__":
    main()
