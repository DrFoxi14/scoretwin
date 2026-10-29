"""Deterministic synthetic score pages for regression tests and proposal training."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _pitch_from_step(step: int) -> dict:
    letters = ["C", "D", "E", "F", "G", "A", "B"]
    absolute_index = 2 + int(step)  # E4 is the bottom line of treble clef.
    octave_delta, letter_index = divmod(absolute_index, 7)
    return {"step": letters[letter_index], "octave": 4 + octave_delta}


def generate_synthetic_page(
    seed: int = 3,
    width: int = 1120,
    height: int = 400,
    staff_count: int = 2,
    notes_per_staff: int = 9,
    spacing: int = 16,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """Generate a page with exact notehead-center ground truth.

    The page is synthetic and deliberately limited to treble-staff geometry. It
    supports regression testing; it is not a substitute for a real-page benchmark.
    """
    if width < 500 or spacing < 8 or staff_count < 1 or notes_per_staff < 1:
        raise ValueError("Invalid synthetic page dimensions or score density.")
    max_notes = int((width - 160) // 88) + 1
    if notes_per_staff > max_notes:
        raise ValueError(f"notes_per_staff must be <= {max_notes} for width={width}.")
    needed_height = 52 + (staff_count - 1) * 170 + 4 * spacing + 70
    height = max(height, needed_height)
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 255, dtype=np.uint8)
    truth: list[dict] = []
    x_start = 110
    x_gap = 88
    top_start = 52
    staff_gap = 170

    for staff_index in range(staff_count):
        top = top_start + staff_index * staff_gap
        lines = [top + line_index * spacing for line_index in range(5)]
        for y in lines:
            cv2.line(image, (36, y), (width - 36, y), 0, 1, cv2.LINE_AA)

        for bar_number in range(1, notes_per_staff // 4 + 1):
            x_bar = x_start + bar_number * 4 * x_gap - x_gap // 2
            if 3 < x_bar < width - 4:
                cv2.line(image, (x_bar, lines[0]), (x_bar, lines[-1]), 0, 1, cv2.LINE_AA)

        for note_index in range(notes_per_staff):
            x = x_start + note_index * x_gap
            diatonic_step = ((note_index * 3 + seed + staff_index) % 9) - 2
            y = int(round(lines[-1] - diatonic_step * spacing / 2))
            filled = (note_index + staff_index) % 3 != 0
            rx = max(4, int(round(spacing * 0.58)))
            ry = max(3, int(round(spacing * 0.36)))
            angle = float(rng.uniform(-25.0, 25.0))
            thickness = -1 if filled else max(1, int(round(spacing * 0.12)))
            cv2.ellipse(image, (x, y), (rx, ry), angle, 0, 360, 0, thickness, cv2.LINE_AA)

            if (note_index + staff_index) % 2 == 0:
                stem_x = x + rx - 1
                stem_end = max(4, y - int(round(3.1 * spacing)))
                cv2.line(image, (stem_x, stem_end), (stem_x, y), 0, 1, cv2.LINE_AA)
            else:
                stem_x = x - rx + 1
                stem_end = min(height - 4, y + int(round(3.1 * spacing)))
                cv2.line(image, (stem_x, y), (stem_x, stem_end), 0, 1, cv2.LINE_AA)

            truth.append({
                "id": f"gt-s{staff_index}-n{note_index}",
                "staff_index": staff_index,
                "center": [x, y],
                "diatonic_step_from_bottom_line": diatonic_step,
                "pitch": _pitch_from_step(diatonic_step),
                "head_type": "filled_head" if filled else "hollow_head",
            })

    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), float(blur_sigma))
    if noise_sigma > 0:
        noise = rng.normal(0.0, float(noise_sigma), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, truth


def write_synthetic_page(path: str | Path, **kwargs) -> list[dict]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image, truth = generate_synthetic_page(**kwargs)
    if not cv2.imwrite(str(output), image):
        raise IOError(f"Could not write synthetic page to {output}")
    return truth
