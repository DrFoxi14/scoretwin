"""Small synthetic patch model for ranking notehead proposals.

The model is deliberately modest: a softmax linear classifier over normalized
16x16 crops. It is a reproducible baseline, not a claim of human-like music
understanding. The held-out score measures only synthetic patch classification.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LABELS = ("filled_head", "hollow_head", "artifact")
FEATURE_SIZE = 16


def _synthetic_patch(label: str, rng: np.random.Generator, size: int = 24) -> np.ndarray:
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)

    for _ in range(int(rng.integers(0, 3))):
        y = int(rng.integers(5, size - 4))
        draw.line((0, y, size - 1, y), fill=int(rng.integers(0, 100)), width=int(rng.integers(1, 3)))

    if label in ("filled_head", "hollow_head"):
        cx = int(rng.integers(size // 2 - 3, size // 2 + 4))
        cy = int(rng.integers(size // 2 - 3, size // 2 + 4))
        rx = int(rng.integers(4, 8))
        ry = int(rng.integers(3, 6))
        box = (cx - rx, cy - ry, cx + rx, cy + ry)
        if label == "filled_head":
            draw.ellipse(box, fill=int(rng.integers(0, 70)))
        else:
            draw.ellipse(box, fill=255, outline=int(rng.integers(0, 70)), width=2)
        if rng.random() < 0.62:
            side = cx + rx - 1 if rng.random() < 0.5 else cx - rx + 1
            stem_top = max(1, cy - int(rng.integers(7, 14)))
            stem_bottom = min(size - 2, cy + int(rng.integers(7, 14)))
            draw.line((side, stem_top, side, stem_bottom), fill=int(rng.integers(0, 90)), width=1)
        if rng.random() < 0.18:
            draw.ellipse((cx + rx + 2, cy - 1, cx + rx + 4, cy + 1), fill=0)
    else:
        for _ in range(int(rng.integers(1, 5))):
            x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
            x1, y1 = int(rng.integers(0, size)), int(rng.integers(0, size))
            draw.line((x0, y0, x1, y1), fill=int(rng.integers(0, 145)), width=int(rng.integers(1, 3)))
        for _ in range(int(rng.integers(2, 10))):
            x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
            draw.point((x, y), fill=int(rng.integers(0, 180)))

    if rng.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.0, 0.65))))
    arr = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0.0, float(rng.uniform(0.0, 12.0)), arr.shape)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def patch_features(gray_crop: np.ndarray) -> np.ndarray:
    if gray_crop.size == 0:
        gray_crop = np.full((FEATURE_SIZE, FEATURE_SIZE), 255, dtype=np.uint8)
    gray_crop = np.asarray(gray_crop, dtype=np.uint8)
    resized = cv2.resize(gray_crop, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    return (1.0 - resized.astype(np.float32).reshape(-1) / 255.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -30, 30))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def train_proposal_model(
    samples_per_class: int = 400,
    epochs: int = 45,
    seed: int = 19,
) -> tuple[dict, dict]:
    """Train and evaluate on independently generated synthetic patches."""
    if samples_per_class < 10:
        raise ValueError("samples_per_class must be at least 10.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    targets: list[int] = []
    holdout_features: list[np.ndarray] = []
    holdout_targets: list[int] = []

    for label_index, label in enumerate(LABELS):
        for _ in range(samples_per_class):
            features.append(patch_features(_synthetic_patch(label, rng)))
            targets.append(label_index)
        for _ in range(max(20, samples_per_class // 4)):
            holdout_features.append(patch_features(_synthetic_patch(label, rng)))
            holdout_targets.append(label_index)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    x_hold = np.asarray(holdout_features, dtype=np.float32)
    y_hold = np.asarray(holdout_targets, dtype=np.int64)
    x = np.concatenate([np.ones((len(x), 1), dtype=np.float32), x], axis=1)
    x_hold = np.concatenate([np.ones((len(x_hold), 1), dtype=np.float32), x_hold], axis=1)

    model_rng = np.random.default_rng(seed + 1)
    weights = model_rng.normal(0.0, 0.01, (x.shape[1], len(LABELS))).astype(np.float32)
    one_hot = np.eye(len(LABELS), dtype=np.float32)[y]
    learning_rate = 1.4
    regularization = 0.0005
    batch_size = min(96, len(x))

    for epoch in range(epochs):
        order = rng.permutation(len(x))
        rate = learning_rate * (0.985 ** epoch)
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            xb, yb = x[batch], one_hot[batch]
            probabilities = _softmax(xb @ weights)
            gradient = (xb.T @ (probabilities - yb)) / len(batch)
            gradient[1:] += regularization * weights[1:]
            weights -= rate * gradient

    probabilities = _softmax(x_hold @ weights)
    accuracy = float(np.mean(probabilities.argmax(axis=1) == y_hold))
    metrics = {
        "seed": seed,
        "train_n": int(len(y)),
        "holdout_n": int(len(y_hold)),
        "labels": list(LABELS),
        "synthetic_holdout_accuracy": accuracy,
        "epochs": epochs,
    }
    return {"weights": weights, "labels": LABELS}, metrics


def save_model(model: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, weights=np.asarray(model["weights"], dtype=np.float32),
                        labels=np.asarray(model["labels"], dtype=str))


def load_model(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        labels = tuple(str(item) for item in data["labels"].tolist())
        weights = np.asarray(data["weights"], dtype=np.float32)
    if weights.shape != (FEATURE_SIZE * FEATURE_SIZE + 1, len(labels)):
        raise ValueError("Proposal model has an incompatible weight shape.")
    if labels != LABELS:
        raise ValueError("Proposal model labels do not match this implementation.")
    return {"weights": weights, "labels": labels}


def predict_patch(model: dict, gray_crop: np.ndarray) -> dict:
    features = patch_features(gray_crop)
    features = np.concatenate([np.ones(1, dtype=np.float32), features])
    probabilities = _softmax(features[None, :] @ np.asarray(model["weights"], dtype=np.float32))[0]
    return {label: float(probabilities[i]) for i, label in enumerate(model["labels"])}


_DEFAULT_MODEL: dict | None = None


def get_default_model() -> dict:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL, _ = train_proposal_model(samples_per_class=120, epochs=24, seed=19)
    return _DEFAULT_MODEL
