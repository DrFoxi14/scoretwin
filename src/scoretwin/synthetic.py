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


def _draw_accidental_glyph(
    canvas_tau: np.ndarray,
    kind: str,
    cx: float,
    cy: float,
    spacing: float,
    absorbance: float = 1.4,
) -> list[int]:
    """Draw a sharp, flat, or natural into an optical-depth canvas and return bbox."""
    sp = float(spacing)
    w = max(8, int(round(0.95 * sp)))
    h = max(14, int(round(2.1 * sp)))
    x0 = max(0, int(round(cx - w / 2.0)))
    y0 = max(0, int(round(cy - h / 2.0)))
    x1 = min(canvas_tau.shape[1], x0 + w)
    y1 = min(canvas_tau.shape[0], y0 + h)
    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    pw, ph = x1 - x0, y1 - y0
    if pw < 4 or ph < 6:
        return [x0, y0, pw, ph]

    thick = max(1, int(round(0.10 * sp)))
    bar_thick = max(1, int(round(0.16 * sp)))
    if kind == "sharp":
        vx1, vx2 = int(round(0.32 * pw)), int(round(0.68 * pw))
        cv2.line(patch, (vx1, int(0.08 * ph)), (vx1, int(0.92 * ph)), absorbance, thick)
        cv2.line(patch, (vx2, int(0.08 * ph)), (vx2, int(0.92 * ph)), absorbance, thick)
        hy1, hy2 = int(round(0.38 * ph)), int(round(0.62 * ph))
        cv2.line(patch, (1, hy1 + 2), (pw - 2, hy1 - 2), absorbance * 1.1, bar_thick)
        cv2.line(patch, (1, hy2 + 2), (pw - 2, hy2 - 2), absorbance * 1.1, bar_thick)
    elif kind == "flat":
        vx = int(round(0.26 * pw))
        cv2.line(patch, (vx, int(0.06 * ph)), (vx, int(0.90 * ph)), absorbance, thick)
        cv2.ellipse(
            patch,
            (int(round(0.54 * pw)), int(round(0.68 * ph))),
            (max(2, int(round(0.30 * pw))), max(2, int(round(0.19 * ph)))),
            -15.0,
            -90,
            110,
            absorbance * 1.05,
            bar_thick,
        )
    elif kind == "natural":
        vx1, vx2 = int(round(0.30 * pw)), int(round(0.70 * pw))
        hy1, hy2 = int(round(0.38 * ph)), int(round(0.62 * ph))
        cv2.line(patch, (vx1, int(0.08 * ph)), (vx1, int(0.70 * ph)), absorbance, thick)
        cv2.line(patch, (vx2, int(0.30 * ph)), (vx2, int(0.92 * ph)), absorbance, thick)
        cv2.line(patch, (vx1, hy1), (vx2, hy1 - 2), absorbance * 1.1, bar_thick)
        cv2.line(patch, (vx1, hy2), (vx2, hy2 - 2), absorbance * 1.1, bar_thick)

    canvas_tau[y0:y1, x0:x1] += patch
    return [x0, y0, pw, ph]


def generate_extended_synthetic_page(
    seed: int = 101,
    width: int = 1240,
    system_count: int = 2,
    staves_per_system: int = 2,
    measures_per_system: int = 2,
    spacing: int = 16,
    hand: float = 0.0,
    noise_sigma: float = 2.0,
    blur_sigma: float = 0.45,
    illumination_gradient: float = 0.06,
) -> tuple[np.ndarray, dict]:
    """Generate a rich Beer-Lambert synthetic score with chords, voices, accidentals, beams, and systems.

    Uses subtractive optical density composition (Beer-Lambert law):
        I(x, y) = I_0(x, y) * exp(-G_psf * sum_v tau_v(x, y)) + noise
    so overlapping primitives (notehead on staffline, stem on notehead) add in
    optical depth tau rather than clipping to a binary OR mask.
    """
    if system_count < 1 or staves_per_system < 1 or spacing < 10:
        raise ValueError("Invalid system_count, staves_per_system, or spacing.")
    hand = float(np.clip(hand, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    staff_count = system_count * staves_per_system
    intra_system_gap = int(round(7.5 * spacing))
    inter_system_gap = int(round(10.5 * spacing))
    top_margin = int(round(3.5 * spacing))

    staff_tops: list[int] = []
    cur_y = top_margin
    for sys_idx in range(system_count):
        for st_in_sys in range(staves_per_system):
            staff_tops.append(cur_y)
            cur_y += 4 * spacing + (intra_system_gap if st_in_sys < staves_per_system - 1 else inter_system_gap)
    height = max(420, cur_y + int(round(2.5 * spacing)))

    tau_canvas = np.zeros((height, width), dtype=np.float32)
    truth_notes: list[dict] = []
    truth_accidentals: list[dict] = []
    truth_beams: list[dict] = []
    truth_relations: list[dict] = []
    truth_staffs: list[dict] = []

    x_left = 56
    x_right = width - 48
    usable_w = x_right - x_left
    measure_w = usable_w // measures_per_system

    alter_map = {"sharp": 1, "flat": -1, "natural": 0}
    acc_Sequence = ("sharp", "flat", "natural")

    for staff_index in range(staff_count):
        sys_idx = staff_index // staves_per_system
        staff_num = (staff_index % staves_per_system) + 1
        top = staff_tops[staff_index]
        lines = [top + li * spacing for li in range(5)]
        barlines = [x_left + m * measure_w for m in range(1, measures_per_system)]
        truth_staffs.append({
            "index": staff_index,
            "system_index": sys_idx,
            "staff_number": staff_num,
            "lines": lines,
            "spacing": float(spacing),
            "barlines": barlines,
            "clef": {"sign": "G", "line": 2} if staff_num == 1 else {"sign": "F", "line": 4},
        })

        # Draw 5 staff lines in optical depth tau
        for y in lines:
            line_layer = np.zeros_like(tau_canvas)
            cv2.line(line_layer, (x_left - 16, y), (x_right, y), 0.95, 1, cv2.LINE_AA)
            tau_canvas += line_layer

        # Draw barlines
        for x_bar in barlines:
            bar_layer = np.zeros_like(tau_canvas)
            cv2.line(bar_layer, (x_bar, lines[0]), (x_bar, lines[-1]), 1.15, 1, cv2.LINE_AA)
            tau_canvas += bar_layer

        # Left system connector if first staff in system
        if staff_num == 1 and staves_per_system > 1:
            bot_staff_top = staff_tops[staff_index + staves_per_system - 1]
            bot_y = bot_staff_top + 4 * spacing
            conn_layer = np.zeros_like(tau_canvas)
            cv2.line(conn_layer, (x_left - 16, lines[0]), (x_left - 16, bot_y), 1.25, 2, cv2.LINE_AA)
            tau_canvas += conn_layer

        # Populate each measure with 4 event columns:
        # Column 0: 2-note chord (Voice 1, up-stem)
        # Column 1: Accidental + quarter note (Voice 1) AND simultaneous Voice 2 (down-stem)
        # Column 2 & 3: Beamed eighth-note pair (Voice 1, up-stem connected by a beam)
        for m_idx in range(measures_per_system):
            m_x0 = x_left + m_idx * measure_w + int(round(2.4 * spacing))
            col_step_x = (measure_w - int(round(4.2 * spacing))) // 4

            def _add_head_and_stem(
                col_x: int,
                step: int,
                filled: bool,
                stem_dir: str | None,
                voice: int,
                note_type: str,
                chord_id: str | None = None,
                accidental: str | None = None,
                dot: bool = False,
            ) -> dict:
                jx = float(rng.normal(0.0, 0.35 * hand))
                jy = float(rng.normal(0.0, 0.25 * hand))
                cx = float(col_x) + jx
                cy = float(lines[-1]) - step * (spacing / 2.0) + jy
                rx = max(4, int(round(spacing * 0.58)))
                ry = max(3, int(round(spacing * 0.36)))
                angle = -18.0 + float(rng.uniform(-8.0, 8.0)) * (0.3 + 0.7 * hand)
                alpha_ink = float(rng.uniform(1.35, 1.75))

                head_layer = np.zeros_like(tau_canvas)
                thick = -1 if filled else max(1, int(round(spacing * 0.14)))
                cv2.ellipse(
                    head_layer,
                    (int(round(cx)), int(round(cy))),
                    (rx, ry),
                    angle,
                    0,
                    360,
                    alpha_ink,
                    thick,
                    cv2.LINE_AA,
                )
                tau_canvas[:] += head_layer

                stem_info = None
                if stem_dir == "up":
                    sx = int(round(cx + rx - 1))
                    sy_end = max(4, int(round(cy - 3.1 * spacing)))
                    stem_layer = np.zeros_like(tau_canvas)
                    cv2.line(stem_layer, (sx, sy_end), (sx, int(round(cy))), alpha_ink * 0.9, 1, cv2.LINE_AA)
                    tau_canvas[:] += stem_layer
                    stem_info = {"x": sx, "y_top": sy_end, "y_bot": int(round(cy)), "tip_y": sy_end, "dir": "up"}
                elif stem_dir == "down":
                    sx = int(round(cx - rx + 1))
                    sy_end = min(height - 4, int(round(cy + 3.1 * spacing)))
                    stem_layer = np.zeros_like(tau_canvas)
                    cv2.line(stem_layer, (sx, int(round(cy))), (sx, sy_end), alpha_ink * 0.9, 1, cv2.LINE_AA)
                    tau_canvas[:] += stem_layer
                    stem_info = {"x": sx, "y_top": int(round(cy)), "y_bot": sy_end, "tip_y": sy_end, "dir": "down"}

                pitch = _pitch_from_step(step)
                note_id = f"gt-s{staff_index}-m{m_idx}-n{len(truth_notes)}"
                if accidental is not None:
                    acc_cx = cx - rx - 0.85 * spacing
                    acc_bbox = _draw_accidental_glyph(tau_canvas, accidental, acc_cx, cy, spacing, alpha_ink * 0.95)
                    pitch["alter"] = alter_map[accidental]
                    acc_id = f"gt-acc-{len(truth_accidentals)}"
                    truth_accidentals.append({
                        "id": acc_id,
                        "accidental_type": accidental,
                        "alter": alter_map[accidental],
                        "staff_index": staff_index,
                        "target_note_id": note_id,
                        "bbox": acc_bbox,
                    })
                    truth_relations.append({"type": "accidental_of", "source": acc_id, "target": note_id})

                if dot:
                    dot_layer = np.zeros_like(tau_canvas)
                    dx_dot = int(round(cx + rx + 0.45 * spacing))
                    dy_dot = int(round(cy - (0.22 * spacing if step % 2 == 0 else 0.0)))
                    cv2.circle(dot_layer, (dx_dot, dy_dot), max(2, int(round(0.14 * spacing))), alpha_ink, -1, cv2.LINE_AA)
                    tau_canvas[:] += dot_layer

                note_rec = {
                    "id": note_id,
                    "staff_index": staff_index,
                    "system_index": sys_idx,
                    "measure_index": m_idx,
                    "center": [round(cx, 3), round(cy, 3)],
                    "diatonic_step_from_bottom_line": step,
                    "pitch": pitch,
                    "head_type": "filled_head" if filled else "hollow_head",
                    "voice": voice,
                    "type": note_type,
                    "chord_id": chord_id,
                    "accidental": accidental,
                    "dots": 1 if dot else 0,
                    "optical_absorbance": round(alpha_ink, 4),
                    "_stem": stem_info,
                }
                truth_notes.append(note_rec)
                return note_rec

            # Col 0: Two-note chord in Voice 1
            x0_col = m_x0
            chord_id = f"gt-chord-s{staff_index}-m{m_idx}"
            base_step = 2 + ((seed + staff_index + m_idx) % 3)
            n_c1 = _add_head_and_stem(x0_col, base_step, True, None, 1, "quarter", chord_id=chord_id)
            n_c2 = _add_head_and_stem(x0_col, base_step + 2, True, "up", 1, "quarter", chord_id=chord_id)
            # Extend stem of n_c2 down to n_c1
            if n_c2["_stem"] is not None:
                sx = n_c2["_stem"]["x"]
                cv2.line(tau_canvas, (sx, int(round(n_c2["center"][1]))), (sx, int(round(n_c1["center"][1]))), 1.4, 1, cv2.LINE_AA)
            truth_relations.append({"type": "chord_member_with", "source": n_c2["id"], "target": n_c1["id"]})

            # Col 1: Polyphonic two-voice event with accidental on upper voice
            x1_col = m_x0 + col_step_x
            acc_kind = acc_Sequence[(staff_index + m_idx + seed) % len(acc_Sequence)]
            n_v1 = _add_head_and_stem(x1_col, 5, False, "up", 1, "half", accidental=acc_kind, dot=(m_idx == 0))
            n_v2 = _add_head_and_stem(x1_col, 0, False, "down", 2, "half")
            truth_relations.append({"type": "simultaneous_voice_counterpart", "source": n_v2["id"], "target": n_v1["id"]})

            # Col 2 & Col 3: Beamed eighth notes in Voice 1
            x2_col = m_x0 + 2 * col_step_x
            x3_col = m_x0 + 3 * col_step_x
            step_b1 = 3 + (m_idx % 2)
            step_b2 = 4 + (m_idx % 2)
            n_b1 = _add_head_and_stem(x2_col, step_b1, True, "up", 1, "eighth")
            n_b2 = _add_head_and_stem(x3_col, step_b2, True, "up", 1, "eighth")
            if n_b1["_stem"] and n_b2["_stem"]:
                bx0, by0 = n_b1["_stem"]["x"], n_b1["_stem"]["tip_y"]
                bx1, by1 = n_b2["_stem"]["x"], n_b2["_stem"]["tip_y"]
                beam_layer = np.zeros_like(tau_canvas)
                beam_thick = max(3, int(round(0.32 * spacing)))
                cv2.line(beam_layer, (bx0, by0 + beam_thick // 2), (bx1, by1 + beam_thick // 2), 1.65, beam_thick, cv2.LINE_AA)
                tau_canvas[:] += beam_layer
                beam_id = f"gt-beam-s{staff_index}-m{m_idx}"
                truth_beams.append({
                    "id": beam_id,
                    "staff_index": staff_index,
                    "note_ids": [n_b1["id"], n_b2["id"]],
                    "beam_count": 1,
                })

    # Clean internal helper field
    for n in truth_notes:
        n.pop("_stem", None)

    # Apply PSF blur to optical depth tau, then Beer-Lambert transmission I = I_0 * exp(-tau)
    if blur_sigma > 0:
        tau_blurred = cv2.GaussianBlur(tau_canvas, (0, 0), float(blur_sigma))
    else:
        tau_blurred = tau_canvas

    # Smooth spatial illumination field I_0(x, y)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    i0_field = 252.0 * (1.0 - float(illumination_gradient) * (0.45 * xx + 0.35 * yy + 0.20 * (xx ** 2 + yy ** 2)))
    i0_field = np.clip(i0_field, 200.0, 255.0)

    transmitted = i0_field * np.exp(-tau_blurred)
    if noise_sigma > 0:
        transmitted = transmitted + rng.normal(0.0, float(noise_sigma), transmitted.shape).astype(np.float32)

    image = np.clip(np.round(transmitted), 0, 255).astype(np.uint8)
    ground_truth = {
        "seed": seed,
        "hand": hand,
        "spacing": spacing,
        "system_count": system_count,
        "staves_per_system": staves_per_system,
        "measures_per_system": measures_per_system,
        "staffs": truth_staffs,
        "notes": truth_notes,
        "accidentals": truth_accidentals,
        "beams": truth_beams,
        "relations": truth_relations,
        "optical_depth_max": float(tau_canvas.max()),
        "illumination_min_max": [float(i0_field.min()), float(i0_field.max())],
    }
    return image, ground_truth

