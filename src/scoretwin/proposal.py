"""Small synthetic patch model for ranking notehead proposals.

The model is deliberately modest: a softmax linear classifier over normalized
16x16 crops. It is a reproducible baseline, not a claim of human-like music
understanding. The held-out score measures only synthetic patch classification.
"""

from __future__ import annotations

from pathlib import Path
import json

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
    return {"weights": weights, "labels": LABELS, "training_metrics": metrics}, metrics


def save_model(model: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = model.get("training_metrics", {})
    np.savez_compressed(
        output,
        weights=np.asarray(model["weights"], dtype=np.float32),
        labels=np.asarray(model["labels"], dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def load_model(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        labels = tuple(str(item) for item in data["labels"].tolist())
        weights = np.asarray(data["weights"], dtype=np.float32)
        metadata = {}
        if "metadata_json" in data.files:
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except (ValueError, TypeError, json.JSONDecodeError):
                metadata = {}
    if weights.shape != (FEATURE_SIZE * FEATURE_SIZE + 1, len(labels)):
        raise ValueError("Proposal model has an incompatible weight shape.")
    if labels != LABELS:
        raise ValueError("Proposal model labels do not match this implementation.")
    return {"weights": weights, "labels": labels, "training_metrics": metadata}


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


def get_default_model_metrics() -> dict:
    """Return training metadata saved with the cached synthetic patch model."""
    return dict(get_default_model().get("training_metrics", {}))


IDSR_LABELS = ("filled_head", "hollow_head", "sharp", "flat", "natural", "artifact")


def _synthetic_idsr_patch_and_offset(
    label: str,
    rng: np.random.Generator,
    size: int = 28,
    spacing: float = 16.0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Generate a synthetic glyph crop with known sub-pixel offset (dx_sp, dy_sp) in staff spaces."""
    image = np.full((size, size), 255, dtype=np.float32)
    # Random staff lines crossing background
    for _ in range(int(rng.integers(1, 3))):
        y_line = int(rng.integers(4, size - 4))
        cv2.line(image, (0, y_line), (size - 1, y_line), float(rng.uniform(35.0, 115.0)), 1, cv2.LINE_AA)

    # True center offset relative to patch center (size/2, size/2) within [-0.28, +0.28] sp
    dx_sp = float(rng.uniform(-0.26, 0.26))
    dy_sp = float(rng.uniform(-0.26, 0.26))
    cx = size / 2.0 + dx_sp * spacing
    cy = size / 2.0 + dy_sp * spacing
    icx, icy = int(round(cx)), int(round(cy))
    darkness = float(rng.uniform(10.0, 65.0))

    if label in ("filled_head", "hollow_head"):
        rx = max(4, int(round(0.42 * spacing + rng.uniform(-1.0, 1.0))))
        ry = max(3, int(round(0.28 * spacing + rng.uniform(-0.8, 0.8))))
        angle = float(rng.uniform(-24.0, -8.0))
        thick = -1 if label == "filled_head" else max(1, int(round(0.11 * spacing)))
        cv2.ellipse(image, (icx, icy), (rx, ry), angle, 0, 360, darkness, thick, cv2.LINE_AA)
        if rng.random() < 0.65:
            sx = icx + rx - 1 if rng.random() < 0.5 else icx - rx + 1
            cv2.line(image, (sx, max(1, icy - 10)), (sx, min(size - 2, icy + 10)), darkness + 15.0, 1, cv2.LINE_AA)
    elif label == "sharp":
        w, h = max(6, int(round(0.55 * spacing))), max(12, int(round(1.35 * spacing)))
        x1, x2 = icx - w // 3, icx + w // 3
        y1, y2 = icy - h // 5, icy + h // 5
        cv2.line(image, (x1, icy - h // 2), (x1, icy + h // 2), darkness, 1, cv2.LINE_AA)
        cv2.line(image, (x2, icy - h // 2), (x2, icy + h // 2), darkness, 1, cv2.LINE_AA)
        cv2.line(image, (icx - w // 2, y1 + 1), (icx + w // 2, y1 - 1), darkness, 2, cv2.LINE_AA)
        cv2.line(image, (icx - w // 2, y2 + 1), (icx + w // 2, y2 - 1), darkness, 2, cv2.LINE_AA)
    elif label == "flat":
        w, h = max(6, int(round(0.50 * spacing))), max(12, int(round(1.35 * spacing)))
        xb = icx - w // 3
        cv2.line(image, (xb, icy - h // 2), (xb, icy + h // 2), darkness, 1, cv2.LINE_AA)
        cv2.ellipse(image, (xb + 2, icy + h // 5), (max(2, w // 2), max(2, h // 5)), -15.0, -90, 110, darkness, 2, cv2.LINE_AA)
    elif label == "natural":
        w, h = max(6, int(round(0.50 * spacing))), max(12, int(round(1.35 * spacing)))
        x1, x2 = icx - w // 3, icx + w // 3
        y1, y2 = icy - h // 5, icy + h // 5
        cv2.line(image, (x1, icy - h // 2), (x1, icy + h // 4), darkness, 1, cv2.LINE_AA)
        cv2.line(image, (x2, icy - h // 4), (x2, icy + h // 2), darkness, 1, cv2.LINE_AA)
        cv2.line(image, (x1, y1), (x2, y1 - 1), darkness, 2, cv2.LINE_AA)
        cv2.line(image, (x1, y2), (x2, y2 - 1), darkness, 2, cv2.LINE_AA)
    else:
        dx_sp, dy_sp = 0.0, 0.0
        for _ in range(int(rng.integers(1, 5))):
            p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.line(image, p0, p1, float(rng.uniform(40.0, 170.0)), int(rng.integers(1, 3)), cv2.LINE_AA)

    sigma_blur = float(rng.uniform(0.2, 0.65))
    image = cv2.GaussianBlur(image, (0, 0), sigma_blur)
    noise = rng.normal(0.0, float(rng.uniform(1.0, 9.0)), image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255).astype(np.uint8), (dx_sp, dy_sp)


def idsr_patch_features(gray_crop: np.ndarray) -> np.ndarray:
    """Extract intensity + spatial gradient + radial moment features for 6-class + offset prediction."""
    if gray_crop.size == 0:
        gray_crop = np.full((FEATURE_SIZE, FEATURE_SIZE), 255, dtype=np.uint8)
    resized = cv2.resize(np.asarray(gray_crop, dtype=np.uint8), (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    ink = 1.0 - resized.astype(np.float32) / 255.0
    gx = cv2.Sobel(ink, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(ink, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    gx_pool = cv2.resize(np.abs(gx), (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    gy_pool = cv2.resize(np.abs(gy), (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)

    # First and second spatial moments of ink for sub-pixel offset sensitivity
    coords = np.linspace(-0.5, 0.5, FEATURE_SIZE, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    mass = float(np.sum(ink)) + 1e-6
    mx = float(np.sum(ink * xx)) / mass
    my = float(np.sum(ink * yy)) / mass
    mxx = float(np.sum(ink * (xx - mx) ** 2)) / mass
    myy = float(np.sum(ink * (yy - my) ** 2)) / mass
    mxy = float(np.sum(ink * (xx - mx) * (yy - my))) / mass
    center_ring = float(np.mean(ink[5:11, 5:11])) - float(np.mean(ink[6:10, 6:10]))
    moments = np.asarray([mx, my, mxx, myy, mxy, center_ring, mass / float(FEATURE_SIZE * FEATURE_SIZE)], dtype=np.float32)
    return np.concatenate([ink.reshape(-1), gx_pool, gy_pool, moments])


def train_idsr_proposal_model(
    samples_per_class: int = 320,
    epochs: int = 55,
    seed: int = 2026,
    spacing: float = 16.0,
) -> tuple[dict, dict]:
    """Train the multi-task IDSR proposal model (6 classes + sub-pixel center offset regression) and verify the <=0.3 sp contract."""
    if samples_per_class < 20:
        raise ValueError("samples_per_class must be at least 20.")
    rng = np.random.default_rng(seed)
    x_train, y_train, off_train = [], [], []
    x_hold, y_hold, off_hold = [], [], []

    for label_idx, label in enumerate(IDSR_LABELS):
        for _ in range(samples_per_class):
            patch, offset = _synthetic_idsr_patch_and_offset(label, rng, spacing=spacing)
            x_train.append(idsr_patch_features(patch))
            y_train.append(label_idx)
            off_train.append(offset)
        for _ in range(max(30, samples_per_class // 4)):
            patch, offset = _synthetic_idsr_patch_and_offset(label, rng, spacing=spacing)
            x_hold.append(idsr_patch_features(patch))
            y_hold.append(label_idx)
            off_hold.append(offset)

    X = np.asarray(x_train, dtype=np.float32)
    Y = np.asarray(y_train, dtype=np.int64)
    O = np.asarray(off_train, dtype=np.float32)
    Xh = np.asarray(x_hold, dtype=np.float32)
    Yh = np.asarray(y_hold, dtype=np.int64)
    Oh = np.asarray(off_hold, dtype=np.float32)

    X_aug = np.concatenate([np.ones((len(X), 1), dtype=np.float32), X], axis=1)
    Xh_aug = np.concatenate([np.ones((len(Xh), 1), dtype=np.float32), Xh], axis=1)

    # 1. Train softmax classification head
    w_rng = np.random.default_rng(seed + 7)
    W_cls = w_rng.normal(0.0, 0.01, (X_aug.shape[1], len(IDSR_LABELS))).astype(np.float32)
    one_hot = np.eye(len(IDSR_LABELS), dtype=np.float32)[Y]
    lr = 1.25
    reg = 4e-4
    batch_size = min(128, len(X_aug))
    for epoch in range(epochs):
        order = rng.permutation(len(X_aug))
        rate = lr * (0.985 ** epoch)
        for start in range(0, len(X_aug), batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X_aug[idx], one_hot[idx]
            probs = _softmax(xb @ W_cls)
            grad = (xb.T @ (probs - yb)) / len(idx)
            grad[1:] += reg * W_cls[1:]
            W_cls -= rate * grad

    # 2. Fit closed-form Ridge regression head for sub-pixel offset (dx_sp, dy_sp) on non-artifact glyphs
    glyph_mask = Y != IDSR_LABELS.index("artifact")
    Xg = X_aug[glyph_mask]
    Og = O[glyph_mask]
    ridge_lambda = 1.5
    eye = np.eye(Xg.shape[1], dtype=np.float32)
    eye[0, 0] = 0.0
    W_off = np.linalg.solve(Xg.T @ Xg + ridge_lambda * eye, Xg.T @ Og).astype(np.float32)

    # Evaluate on holdout set
    probs_hold = _softmax(Xh_aug @ W_cls)
    acc = float(np.mean(probs_hold.argmax(axis=1) == Yh))

    hold_glyph_mask = Yh != IDSR_LABELS.index("artifact")
    pred_off = Xh_aug[hold_glyph_mask] @ W_off
    true_off = Oh[hold_glyph_mask]
    raw_err_sp = np.hypot(true_off[:, 0], true_off[:, 1])
    residual_err_sp = np.hypot(true_off[:, 0] - pred_off[:, 0], true_off[:, 1] - pred_off[:, 1])
    mean_err_sp = float(np.mean(residual_err_sp))
    p95_err_sp = float(np.percentile(residual_err_sp, 95))
    contract_fraction = float(np.mean(residual_err_sp <= 0.30))

    metrics = {
        "seed": seed,
        "train_n": int(len(Y)),
        "holdout_n": int(len(Yh)),
        "labels": list(IDSR_LABELS),
        "holdout_6class_accuracy": round(acc, 4),
        "holdout_raw_offset_mean_sp": round(float(np.mean(raw_err_sp)), 4),
        "holdout_raw_offset_p95_sp": round(float(np.percentile(raw_err_sp, 95)), 4),
        "holdout_refined_offset_mean_sp": round(mean_err_sp, 4),
        "holdout_refined_offset_p95_sp": round(p95_err_sp, 4),
        "fraction_within_0_30_sp_contract": round(contract_fraction, 4),
        "satisfies_0_30_sp_p95_contract": bool(p95_err_sp <= 0.30),
    }
    model = {
        "kind": "idsr-multitask-6class-plus-offset-ridge",
        "weights_cls": W_cls,
        "weights_off": W_off,
        "labels": IDSR_LABELS,
        "training_metrics": metrics,
    }
    return model, metrics


def predict_idsr_patch(model: dict, gray_crop: np.ndarray) -> dict:
    """Predict 6-class probabilities and sub-pixel center offset (dx_sp, dy_sp)."""
    feats = idsr_patch_features(gray_crop)
    aug = np.concatenate([np.ones(1, dtype=np.float32), feats])[None, :]
    probs = _softmax(aug @ np.asarray(model["weights_cls"], dtype=np.float32))[0]
    offset = (aug @ np.asarray(model["weights_off"], dtype=np.float32))[0]
    label_probs = {label: float(probs[i]) for i, label in enumerate(model["labels"])}
    best_label = max(label_probs, key=label_probs.get)
    return {
        "label": best_label,
        "confidence": float(label_probs[best_label]),
        "probabilities": label_probs,
        "offset_sp": [float(np.clip(offset[0], -0.35, 0.35)), float(np.clip(offset[1], -0.35, 0.35))],
    }


_DEFAULT_IDSR_MODEL: dict | None = None


def get_default_idsr_model() -> dict:
    global _DEFAULT_IDSR_MODEL
    if _DEFAULT_IDSR_MODEL is None:
        _DEFAULT_IDSR_MODEL, _ = train_idsr_proposal_model(samples_per_class=180, epochs=45, seed=2026)
    return _DEFAULT_IDSR_MODEL

