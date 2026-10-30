"""Extended musical notation detection and structural inference for ScoreTwin / IDSR.

Provides pixel-level detection and geometric-symbolic inference for:
- Adaptive multi-strip staff-system detection (tolerant to faint ink and scan skew);
- Multi-staff system grouping (grand staff / multi-system layout) and clef inference;
- Accidentals (sharp, flat, natural) preceding noteheads and in key signatures;
- Beams, flags, and augmentation dots;
- Chord grouping, polyphonic multi-voice separation (up-stem vs down-stem streams),
  and symbolic rhythm/duration inference.
"""

from __future__ import annotations

from collections import defaultdict
import math
import cv2
import numpy as np

from .imaging import StaffSystem, binarize


def detect_staff_systems_adaptive(
    gray: np.ndarray,
    max_skew_deg: float = 3.5,
) -> tuple[list[StaffSystem], np.ndarray, np.ndarray]:
    """Detect five-line staff systems via strip-local 5-tooth Dirac comb filtering.

    Handles faint scans (where global Otsu thresholding drops staff lines),
    uneven vertical illumination (via local RMS normalization), and moderately
    skewed/curled scans (by evaluating the 5-tooth comb inside narrow vertical
    strips before skew alignment).
    """
    if gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    height, width = gray.shape
    binary_otsu = binarize(gray)

    # Vertical black-hat emphasizes thin horizontal dark lines (1..4 px thick)
    # independent of low contrast or uneven background illumination.
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    bh_thresh_val = max(6.0, float(np.percentile(blackhat, 88.0)))
    binary_bh = (blackhat >= bh_thresh_val).astype(np.uint8) * 255

    # Local adaptive threshold captures faint ink strokes missed by global Otsu.
    block_size = max(15, (min(height, width) // 24) | 1)
    binary_adapt = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        6,
    )
    binary = cv2.bitwise_or(binary_otsu, cv2.bitwise_and(binary_adapt, binary_bh))

    # Build strip-normalized horizontal line signal
    signal = 0.70 * blackhat.astype(np.float64) + 0.30 * (255 - gray).astype(np.float64)
    n_strips = max(8, min(16, width // 40))
    strip_w = max(1, width // n_strips)
    profs: list[np.ndarray] = []
    xs: list[float] = []
    s_start = 2 if n_strips >= 10 else 1
    s_end = (n_strips - 2) if n_strips >= 10 else (n_strips - 1)
    for s in range(s_start, s_end):
        x0 = s * strip_w
        x1 = width if s == n_strips - 1 else (s + 1) * strip_w
        if x1 - x0 < 6:
            continue
        p = signal[:, x0:x1].mean(axis=1)
        bg = cv2.GaussianBlur(p.reshape(-1, 1), (1, 45), 0).ravel()
        hp = np.maximum(0.0, p - 0.75 * bg)
        scale = np.sqrt(cv2.GaussianBlur(np.square(hp).reshape(-1, 1), (1, 121), 0).ravel()) + 3.0
        profs.append(hp / scale)
        xs.append(0.5 * (x0 + x1))

    if not profs:
        return [], binary, np.zeros_like(binary)

    y_idx = np.arange(height, dtype=np.float64)
    min_sp = 4.2
    max_sp = min(32.0, max(20.0, height / 5.5))

    # 1. Estimate dominant staff spacing s_dom across strips
    best_s = 10.0
    best_sc = -1.0
    top_k_rows = max(4, min(30, height // 25))
    for s in np.linspace(min_sp, max_sp, 120):
        tot = 0.0
        for p in profs:
            c = sum(np.interp(y_idx + k * s, y_idx, p, left=0.0, right=0.0) for k in range(5)) / 5.0
            sp_c = sum(np.interp(y_idx + (k + 0.5) * s, y_idx, p, left=0.0, right=0.0) for k in range(4)) / 4.0
            net = np.maximum(0.0, c - 0.80 * sp_c)
            tot += float(np.sum(np.sort(net)[-top_k_rows:]))
        if tot > best_sc:
            best_sc = tot
            best_s = float(s)

    # 2. Evaluate 5-tooth Dirac comb envelope inside each narrow strip
    strip_combs: list[np.ndarray] = []
    for p in profs:
        env = np.zeros(height, dtype=np.float64)
        for s in np.linspace(0.90 * best_s, 1.10 * best_s, 11):
            c = sum(np.interp(y_idx + k * s, y_idx, p, left=0.0, right=0.0) for k in range(5)) / 5.0
            sp_c = sum(np.interp(y_idx + (k + 0.5) * s, y_idx, p, left=0.0, right=0.0) for k in range(4)) / 4.0
            env = np.maximum(env, np.maximum(0.0, c - 0.80 * sp_c))
        strip_combs.append(env)

    # 3. Align strip comb envelopes across candidate skew angles
    cx = width / 2.0
    best_comb = np.zeros(height, dtype=np.float64)
    for deg in np.linspace(-max_skew_deg, max_skew_deg, 45):
        slope = math.tan(math.radians(deg))
        acc = np.zeros(height, dtype=np.float64)
        for sc_p, sx in zip(strip_combs, xs):
            acc += np.interp(y_idx + slope * (sx - cx), y_idx, sc_p, left=0.0, right=0.0)
        acc /= len(strip_combs)
        best_comb = np.maximum(best_comb, acc)

    max_comb = float(np.max(best_comb)) if best_comb.size else 0.0
    if max_comb < 0.32:
        return [], binary, np.zeros_like(binary)

    thresh = max(0.38, max_comb * 0.35)
    peaks: list[int] = []
    min_dist = int(round(6.5 * best_s))
    margin_y = max(4, int(round(0.05 * height)))
    for y in np.argsort(best_comb)[::-1]:
        y_int = int(y)
        if best_comb[y_int] < thresh:
            break
        if y_int < margin_y or y_int + 4 * best_s >= height - margin_y:
            continue
        if all(abs(y_int - py) >= min_dist for py in peaks):
            peaks.append(y_int)
    peaks.sort()

    # 4. Refine each 5-line staff's exact line rows and build restricted staff_mask
    seg_width = max(16, min(42, width // 24))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (seg_width, 1))
    raw_horiz_mask = cv2.bitwise_or(
        cv2.morphologyEx(binary_bh, cv2.MORPH_OPEN, horiz_kernel),
        cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, horiz_kernel),
    )
    staff_mask = np.zeros_like(binary)
    candidate_staffs: list[StaffSystem] = []
    mean_prof = np.mean(profs, axis=0)
    horiz_row_counts = (raw_horiz_mask > 0).sum(axis=1)

    for idx, top_y in enumerate(peaks):
        # Refine local spacing in [0.92*best_s, 1.08*best_s] around top_y
        local_best_s = best_s
        local_best_val = -1.0
        for s_try in np.linspace(0.92 * best_s, 1.08 * best_s, 17):
            val = sum(
                float(np.interp(top_y + k * s_try, y_idx, mean_prof, left=0.0, right=0.0))
                for k in range(5)
            )
            if val > local_best_val:
                local_best_val = val
                local_best_s = float(s_try)

        snapped_lines: list[int] = []
        for k in range(5):
            y_est = int(round(top_y + k * local_best_s))
            w0 = max(0, y_est - 2)
            w1 = min(height, y_est + 3)
            if w1 > w0 and np.max(horiz_row_counts[w0:w1]) > 0:
                ly = w0 + int(np.argmax(horiz_row_counts[w0:w1]))
            else:
                ly = y_est
            snapped_lines.append(ly)

        lines = tuple(snapped_lines)
        spacing = float(np.mean(np.diff(lines)))
        candidate_staffs.append(
            StaffSystem(index=idx, lines=lines, spacing=spacing)
        )
        for ly in lines:
            y0 = max(0, ly - 2)
            y1 = min(height, ly + 3)
            staff_mask[y0:y1, :] = raw_horiz_mask[y0:y1, :]

    return candidate_staffs, binary, staff_mask


def _make_accidental_templates(spacing: float) -> dict[str, np.ndarray]:
    """Synthesize normalized templates for sharp, flat, and natural at staff scale."""
    sp = max(8.0, float(spacing))
    w = max(9, int(round(1.1 * sp)))
    h = max(15, int(round(2.4 * sp)))
    templates: dict[str, np.ndarray] = {}

    # Sharp (#): two vertical lines + two slanted crossbars
    sharp = np.zeros((h, w), dtype=np.float32)
    x1, x2 = int(round(0.32 * w)), int(round(0.68 * w))
    cv2.line(sharp, (x1, int(0.1 * h)), (x1, int(0.9 * h)), 1.0, max(1, int(round(0.1 * sp))))
    cv2.line(sharp, (x2, int(0.1 * h)), (x2, int(0.9 * h)), 1.0, max(1, int(round(0.1 * sp))))
    y1, y2 = int(round(0.38 * h)), int(round(0.62 * h))
    cv2.line(sharp, (1, y1 + 2), (w - 2, y1 - 2), 1.0, max(1, int(round(0.16 * sp))))
    cv2.line(sharp, (1, y2 + 2), (w - 2, y2 - 2), 1.0, max(1, int(round(0.16 * sp))))
    templates["sharp"] = sharp

    # Flat (b): tall left vertical stem + lower-right loop
    flat = np.zeros((h, w), dtype=np.float32)
    xb = int(round(0.28 * w))
    cv2.line(flat, (xb, int(0.08 * h)), (xb, int(0.88 * h)), 1.0, max(1, int(round(0.11 * sp))))
    cv2.ellipse(
        flat,
        (int(round(0.52 * w)), int(round(0.68 * h))),
        (max(2, int(round(0.28 * w))), max(2, int(round(0.18 * h)))),
        -15.0,
        -90,
        110,
        1.0,
        max(1, int(round(0.13 * sp))),
    )
    templates["flat"] = flat

    # Natural: left vertical upper-heavy + right vertical lower-heavy + two bars
    nat = np.zeros((h, w), dtype=np.float32)
    cv2.line(nat, (x1, int(0.10 * h)), (x1, int(0.72 * h)), 1.0, max(1, int(round(0.1 * sp))))
    cv2.line(nat, (x2, int(0.28 * h)), (x2, int(0.90 * h)), 1.0, max(1, int(round(0.1 * sp))))
    cv2.line(nat, (x1, y1), (x2, y1 - 2), 1.0, max(1, int(round(0.15 * sp))))
    cv2.line(nat, (x1, y2), (x2, y2 - 2), 1.0, max(1, int(round(0.15 * sp))))
    templates["natural"] = nat

    return templates


def classify_accidental_patch(patch_binary: np.ndarray, spacing: float) -> tuple[str | None, float]:
    """Classify a binary foreground crop as sharp, flat, natural, or None."""
    if patch_binary.size == 0 or np.count_nonzero(patch_binary) < 4:
        return None, 0.0
    templates = _make_accidental_templates(spacing)
    th, tw = next(iter(templates.values())).shape
    resized = cv2.resize(
        (patch_binary > 0).astype(np.float32),
        (tw, th),
        interpolation=cv2.INTER_AREA,
    )
    norm_patch = resized - float(np.mean(resized))
    denom_patch = float(np.linalg.norm(norm_patch))
    if denom_patch <= 1e-6:
        return None, 0.0

    # Structural features to disambiguate sharp vs flat vs natural
    top_half = float(np.sum(resized[: th // 2, :]))
    bot_half = float(np.sum(resized[th // 2 :, :]))
    left_half = float(np.sum(resized[:, : tw // 2]))
    right_half = float(np.sum(resized[:, tw // 2 :]))
    top_right = float(np.sum(resized[: int(0.42 * th), int(0.55 * tw) :]))
    bot_left = float(np.sum(resized[int(0.68 * th) :, : int(0.45 * tw)]))

    scores: dict[str, float] = {}
    for name, tmpl in templates.items():
        norm_t = tmpl - float(np.mean(tmpl))
        corr = float(np.sum(norm_patch * norm_t) / (denom_patch * float(np.linalg.norm(norm_t)) + 1e-8))
        scores[name] = corr

    # Apply morphological priors:
    # Flat has very little ink in top-right and strong bottom/left weight
    if top_right < 0.08 * (top_half + bot_half) and bot_half > 1.15 * top_half:
        scores["flat"] += 0.22
    # Sharp has symmetric top/bottom and left/right ink distribution
    if 0.72 <= (top_half / max(bot_half, 1e-4)) <= 1.38 and 0.72 <= (left_half / max(right_half, 1e-4)) <= 1.38:
        scores["sharp"] += 0.16
    # Natural has empty bottom-left or top-right relative to sharp
    if bot_left < 0.06 * (top_half + bot_half) and top_right < 0.09 * (top_half + bot_half):
        scores["natural"] += 0.18

    best_name = max(scores, key=scores.get)
    best_score = float(scores[best_name])
    if best_score < 0.24:
        return None, best_score
    return best_name, min(0.99, round(best_score, 4))


def detect_accidentals(
    gray: np.ndarray,
    binary: np.ndarray,
    staff_mask: np.ndarray,
    staffs: list[StaffSystem],
    notes: list[dict],
) -> list[dict]:
    """Detect accidentals immediately preceding noteheads and attach pitch alterations."""
    if not staffs or not notes:
        return []
    clean = cv2.subtract(binary, staff_mask)
    height, width = clean.shape
    detected_accidentals: list[dict] = []

    # Sort notes left-to-right per staff to prevent overlapping search windows
    notes_by_staff: dict[int, list[dict]] = defaultdict(list)
    for note in notes:
        notes_by_staff[int(note["staff_index"])].append(note)

    alter_map = {"sharp": 1, "flat": -1, "natural": 0}
    reclassified_note_ids: set[str] = set()

    for staff_index, staff_notes in notes_by_staff.items():
        if staff_index >= len(staffs):
            continue
        staff = staffs[staff_index]
        sp = float(staff.spacing)
        staff_notes.sort(key=lambda n: float(n["center"][0]))

        # Pass 1: Check if any stemless note candidate immediately to the left of a notehead
        # is actually a tall accidental glyph whose central bars were proposed as a notehead.
        for idx in range(len(staff_notes) - 1):
            cand = staff_notes[idx]
            target = staff_notes[idx + 1]
            if cand["id"] in reclassified_note_ids or cand.get("stem_ids"):
                continue
            cx_c, cy_c = float(cand["center"][0]), float(cand["center"][1])
            cx_t, cy_t = float(target["center"][0]), float(target["center"][1])
            if 0.65 * sp <= (cx_t - cx_c) <= 2.10 * sp and abs(cy_t - cy_c) <= 0.85 * sp:
                # Inspect the full vertical window around cand in `clean`
                wx0 = max(0, int(round(cx_c - 0.65 * sp)))
                wx1 = min(width, int(round(cx_c + 0.65 * sp)))
                wy0 = max(0, int(round(cy_c - 1.35 * sp)))
                wy1 = min(height, int(round(cy_c + 1.35 * sp)))
                w_roi = clean[wy0:wy1, wx0:wx1]
                ys, xs = np.nonzero(w_roi)
                if len(xs) >= 8:
                    bh = int(ys.max() - ys.min() + 1)
                    bw = int(xs.max() - xs.min() + 1)
                    if bh >= 1.35 * sp and 0.30 * sp <= bw <= 1.35 * sp:
                        reclassified_note_ids.add(str(cand["id"]))

        active_staff_notes = [n for n in staff_notes if str(n["id"]) not in reclassified_note_ids]

        for idx, note in enumerate(active_staff_notes):
            cx, cy = float(note["center"][0]), float(note["center"][1])
            rx = float(note["bbox"][2]) / 2.0
            # Search window to the left of the notehead, bounded by any previous true note
            x_right = int(round(cx - rx - max(1.0, 0.12 * sp)))
            x_left = int(round(cx - rx - 1.75 * sp))
            if idx > 0:
                prev_cx, prev_cy = float(active_staff_notes[idx - 1]["center"][0]), float(active_staff_notes[idx - 1]["center"][1])
                if abs(prev_cx - cx) > 0.45 * sp:
                    x_left = max(x_left, int(round(prev_cx + float(active_staff_notes[idx - 1]["bbox"][2]) / 2.0 + 2)))
                elif abs(prev_cy - cy) < 1.2 * sp:
                    # Chord notehead at same x: only allow one accidental at same vertical band
                    pass
            y_top = max(0, int(round(cy - 1.35 * sp)))
            y_bot = min(height, int(round(cy + 1.35 * sp)))
            x_left = max(0, x_left)
            x_right = min(width, x_right)
            if x_right - x_left < max(4, int(0.35 * sp)) or y_bot - y_top < max(6, int(0.8 * sp)):
                continue

            roi = clean[y_top:y_bot, x_left:x_right]
            ink_pixels = int(np.count_nonzero(roi))
            if ink_pixels < max(6, int(0.12 * sp * sp)) or ink_pixels > int(1.6 * sp * sp):
                continue

            # Find tight bounding box of foreground in ROI
            ys, xs = np.nonzero(roi)
            if len(xs) == 0:
                continue
            bx0, bx1 = int(xs.min()), int(xs.max()) + 1
            by0, by1 = int(ys.min()), int(ys.max()) + 1
            bw, bh = bx1 - bx0, by1 - by0
            if not (0.30 * sp <= bw <= 1.45 * sp and 0.95 * sp <= bh <= 3.1 * sp):
                continue
            # Exclude tall stems (very narrow width)
            if bw <= max(2, int(round(0.28 * sp))) and bh >= 1.8 * sp:
                continue

            patch = roi[by0:by1, bx0:bx1]
            acc_kind, score = classify_accidental_patch(patch, sp)
            if acc_kind is None:
                continue

            abs_x = x_left + bx0
            abs_y = y_top + by0
            acc_id = f"accidental-{len(detected_accidentals):04d}"
            acc_obj = {
                "id": acc_id,
                "kind": "accidental",
                "accidental_type": acc_kind,
                "alter": alter_map[acc_kind],
                "staff_index": staff_index,
                "target_note_id": note["id"],
                "bbox": [abs_x, abs_y, bw, bh],
                "center": [round(abs_x + bw / 2.0, 3), round(abs_y + bh / 2.0, 3)],
                "confidence": score,
            }
            detected_accidentals.append(acc_obj)

            # Attach explicitly only when not already annotated
            if note.get("accidental") is None:
                note["accidental"] = acc_kind
                note["pitch"] = {**note["pitch"], "alter": alter_map[acc_kind]}
                note["accidental_confidence"] = score

    if reclassified_note_ids:
        notes[:] = [n for n in notes if str(n["id"]) not in reclassified_note_ids]

    return detected_accidentals


def detect_beams_and_dots(
    binary: np.ndarray,
    staff_mask: np.ndarray,
    staffs: list[StaffSystem],
    notes: list[dict],
    stems: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Detect thick beam segments connecting stems and augmentation dots after noteheads."""
    if not staffs:
        return [], []
    clean = cv2.subtract(binary, staff_mask)
    height, width = clean.shape
    beams: list[dict] = []
    dots: list[dict] = []

    notes_by_id = {str(n["id"]): n for n in notes}
    stems_by_staff: dict[int, list[dict]] = defaultdict(list)
    for stem in stems:
        if stem.get("associated_note_ids"):
            stems_by_staff[int(stem["staff_index"])].append(stem)

    # 1. Detect beams between adjacent stems on the same staff
    for staff_index, staff_stems in stems_by_staff.items():
        if staff_index >= len(staffs):
            continue
        sp = float(staffs[staff_index].spacing)
        staff_stems.sort(key=lambda s: float(s["center"][0]))
        for i in range(len(staff_stems) - 1):
            s1, s2 = staff_stems[i], staff_stems[i + 1]
            x1 = float(s1["center"][0])
            x2 = float(s2["center"][0])
            if not (0.5 * sp <= (x2 - x1) <= 8.5 * sp):
                continue

            # Determine stem free tips (away from attached noteheads)
            def _stem_tip(stem_obj: dict) -> tuple[float, float, str]:
                bx, by, bw, bh = [float(v) for v in stem_obj["bbox"]]
                n_ys = [
                    float(notes_by_id[nid]["center"][1])
                    for nid in stem_obj.get("associated_note_ids", [])
                    if nid in notes_by_id
                ]
                mean_ny = float(np.mean(n_ys)) if n_ys else (by + bh / 2.0)
                if (by + bh / 2.0) < mean_ny:
                    return float(stem_obj["center"][0]), by, "up"
                return float(stem_obj["center"][0]), by + bh, "down"

            tx1, ty1, dir1 = _stem_tip(s1)
            tx2, ty2, dir2 = _stem_tip(s2)
            if dir1 != dir2 or abs(ty2 - ty1) > 2.5 * sp:
                continue

            # Sample strip between (tx1, ty1) and (tx2, ty2)
            n_samples = max(8, int(round(x2 - x1)))
            xs = np.linspace(tx1, tx2, n_samples)
            ys = np.linspace(ty1, ty2, n_samples)
            band_half = max(2, int(round(0.45 * sp)))
            coverage = 0
            thickness_samples = []
            for sx, sy in zip(xs, ys):
                ix = int(round(sx))
                iy0 = max(0, int(round(sy)) - band_half)
                iy1 = min(height, int(round(sy)) + band_half + 1)
                if 0 <= ix < width and iy1 > iy0:
                    col = clean[iy0:iy1, ix] > 0
                    c_sum = int(np.sum(col))
                    if c_sum >= max(2, int(round(0.18 * sp))):
                        coverage += 1
                        thickness_samples.append(c_sum)

            if coverage >= 0.72 * n_samples:
                mean_thick = float(np.mean(thickness_samples)) if thickness_samples else 2.0
                beam_count = 2 if mean_thick >= 0.62 * sp else 1
                beam_id = f"beam-{len(beams):04d}"
                beams.append({
                    "id": beam_id,
                    "kind": "beam",
                    "staff_index": staff_index,
                    "stem_ids": [s1["id"], s2["id"]],
                    "note_ids": list(dict.fromkeys(s1["associated_note_ids"] + s2["associated_note_ids"])),
                    "beam_count": beam_count,
                    "direction": dir1,
                    "bbox": [int(min(tx1, tx2)), int(min(ty1, ty2) - mean_thick / 2.0), int(abs(tx2 - tx1)), int(abs(ty2 - ty1) + mean_thick)],
                    "geometry": {
                        "x0_px": round(tx1, 3),
                        "y0_px": round(ty1, 3),
                        "x1_px": round(tx2, 3),
                        "y1_px": round(ty2, 3),
                        "thickness_px_estimate": round(mean_thick, 3),
                        "beam_count": beam_count,
                        "sub_beams": [[int(min(tx1, tx2)), int(min(ty1, ty2)), int(abs(tx2 - tx1)), int(max(2, mean_thick))]],
                    },
                })

    # 2. Direct Optical Beam Detection from clean foreground connected components
    # Captures printed/engraved beams even when stems are slanted, broken or unassociated
    optical_beam_candidates: list[dict] = []
    for staff in staffs:
        staff_index = int(staff.index)
        sp = float(staff.spacing)
        top_limit = max(0, int(round(staff.top - 3.5 * sp)))
        bottom_limit = min(height, int(round(staff.bottom + 3.5 * sp)))
        roi = clean[top_limit:bottom_limit, :]
        num_l, _labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
        for comp_i in range(1, num_l):
            bx, by, bw, bh, area = stats[comp_i]
            cx, cy = centroids[comp_i]
            min_thick = max(6, int(round(0.30 * sp)))
            if (
                bw >= 1.35 * sp
                and bw <= 25.0 * sp
                and bh >= min_thick
                and bh <= 2.5 * sp
                and (bw / max(1, bh)) >= 1.6
                and area >= int(0.20 * sp * sp)
            ):
                optical_beam_candidates.append({
                    "staff_index": staff_index,
                    "bbox": [int(bx), int(by + top_limit), int(bw), int(bh)],
                    "center": [float(cx), float(cy + top_limit)],
                    "thickness": int(bh),
                    "area": int(area),
                })

    # Group parallel double beams (sixteenth notes: 2 bare paralele)
    skip_opt = set()
    for i, b1 in enumerate(optical_beam_candidates):
        if i in skip_opt:
            continue
        sp = float(staffs[b1["staff_index"]].spacing)
        x1_min, y1_min, w1, h1 = b1["bbox"]
        x1_max = x1_min + w1
        y1_max = y1_min + h1

        paired_beam = None
        for j, b2 in enumerate(optical_beam_candidates):
            if j <= i or j in skip_opt or b2["staff_index"] != b1["staff_index"]:
                continue
            x2_min, y2_min, w2, h2 = b2["bbox"]
            x2_max = x2_min + w2
            overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            if overlap_x >= 0.40 * min(w1, w2) and abs(b2["center"][1] - b1["center"][1]) <= 1.65 * sp:
                paired_beam = (j, b2)
                break

        if paired_beam:
            j, b2 = paired_beam
            skip_opt.add(j)
            comb_x0 = min(x1_min, b2["bbox"][0])
            comb_x1 = max(x1_max, b2["bbox"][0] + b2["bbox"][2])
            comb_y0 = min(y1_min, b2["bbox"][1])
            comb_y1 = max(y1_max, b2["bbox"][1] + b2["bbox"][3])
            sub_b = [b1["bbox"], b2["bbox"]]
            sub_b.sort(key=lambda b: b[1])
            beam_count = 2
            mean_thick = float(max(h1, b2["thickness"]))
            y0_cen = float(min(b1["center"][1], b2["center"][1]))
            y1_cen = float(max(b1["center"][1], b2["center"][1]))
            box = [comb_x0, comb_y0, comb_x1 - comb_x0, comb_y1 - comb_y0]
        else:
            beam_count = 2 if b1["thickness"] >= 0.60 * sp else 1
            mean_thick = float(h1)
            comb_x0, comb_x1 = x1_min, x1_max
            y0_cen = y1_cen = float(b1["center"][1])
            sub_b = [b1["bbox"]]
            box = b1["bbox"]

        # Check if already covered by an existing stem-to-stem beam
        already_covered = False
        for ex_b in beams:
            if ex_b["staff_index"] == b1["staff_index"]:
                ex_x0 = ex_b["geometry"]["x0_px"]
                ex_x1 = ex_b["geometry"]["x1_px"]
                if max(0, min(comb_x1, ex_x1) - max(comb_x0, ex_x0)) >= 0.50 * min(comb_x1 - comb_x0, ex_x1 - ex_x0):
                    already_covered = True
                    ex_b["beam_count"] = max(ex_b["beam_count"], beam_count)
                    ex_b["geometry"]["beam_count"] = ex_b["beam_count"]
                    ex_b["geometry"]["sub_beams"] = sub_b
                    break

        if not already_covered:
            matched_notes = [
                str(n["id"]) for n in notes
                if int(n["staff_index"]) == b1["staff_index"]
                and (comb_x0 - 0.5 * sp) <= float(n["center"][0]) <= (comb_x1 + 0.5 * sp)
            ]
            beam_id = f"beam-{len(beams):04d}"
            beams.append({
                "id": beam_id,
                "kind": "beam",
                "staff_index": b1["staff_index"],
                "stem_ids": [],
                "note_ids": matched_notes,
                "beam_count": beam_count,
                "direction": "up" if y0_cen < staffs[b1["staff_index"]].top else "down",
                "bbox": box,
                "geometry": {
                    "x0_px": round(float(comb_x0), 3),
                    "y0_px": round(y0_cen, 3),
                    "x1_px": round(float(comb_x1), 3),
                    "y1_px": round(y1_cen, 3),
                    "thickness_px_estimate": round(mean_thick, 3),
                    "beam_count": beam_count,
                    "sub_beams": sub_b,
                },
            })

    # 3. Detect augmentation dots immediately to the right of noteheads
    # Suppress false dots that collide with detected beams or stems
    for note in notes:
        staff_index = int(note["staff_index"])
        if staff_index >= len(staffs):
            continue
        sp = float(staffs[staff_index].spacing)
        cx, cy = float(note["center"][0]), float(note["center"][1])
        rx = float(note["bbox"][2]) / 2.0
        x0 = max(0, int(round(cx + rx + 0.15 * sp)))
        x1 = min(width, int(round(cx + rx + 0.95 * sp)))
        y0 = max(0, int(round(cy - 0.55 * sp)))
        y1 = min(height, int(round(cy + 0.55 * sp)))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        roi = clean[y0:y1, x0:x1]
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
        for comp in range(1, num_labels):
            bx, by, bw, bh, area = [int(v) for v in stats[comp]]
            min_dot = max(2, int(round(0.015 * sp * sp)))
            max_dot = max(6, int(round(0.18 * sp * sp)))
            if min_dot <= area <= max_dot and 0.5 <= (bw / max(bh, 1)) <= 2.0 and max(bw, bh) <= max(4, int(round(0.45 * sp))):
                dcx = x0 + float(centroids[comp][0])
                dcy = y0 + float(centroids[comp][1])

                # Discard if inside or adjacent to any beam bounding box
                in_beam = False
                for b in beams:
                    if b["staff_index"] != staff_index:
                        continue
                    b_box = b.get("bbox") or [b["geometry"]["x0_px"], b["geometry"]["y0_px"] - 4, b["geometry"]["x1_px"] - b["geometry"]["x0_px"], 8]
                    if (b_box[0] - 0.5 * sp) <= dcx <= (b_box[0] + b_box[2] + 0.5 * sp) and (b_box[1] - 0.5 * sp) <= dcy <= (b_box[1] + b_box[3] + 0.5 * sp):
                        in_beam = True
                        break
                if in_beam:
                    continue

                dot_id = f"dot-{len(dots):04d}"
                dots.append({
                    "id": dot_id,
                    "kind": "dot",
                    "staff_index": staff_index,
                    "target_note_id": note["id"],
                    "center": [round(dcx, 3), round(dcy, 3)],
                    "bbox": [x0 + bx, y0 + by, bw, bh],
                })
                note["dots"] = int(note.get("dots", 0)) + 1
                break

    return beams, dots


def infer_chords_voices_and_durations(
    staffs: list[StaffSystem],
    notes: list[dict],
    stems: list[dict],
    beams: list[dict],
    barlines_by_staff: dict[int, list[int]] | None = None,
) -> dict:
    """Infer chord groupings, polyphonic voice streams (voice 1 vs 2), and durations."""
    barlines_by_staff = barlines_by_staff or {}
    stems_by_id = {str(s["id"]): s for s in stems}
    beam_count_by_stem: dict[str, int] = defaultdict(int)
    beam_count_by_note: dict[str, int] = defaultdict(int)
    for beam in beams:
        b_count = int(beam.get("beam_count", 1))
        for sid in beam.get("stem_ids", []):
            beam_count_by_stem[str(sid)] = max(beam_count_by_stem[str(sid)], b_count)
        for nid in beam.get("note_ids", []):
            beam_count_by_note[str(nid)] = max(beam_count_by_note[str(nid)], b_count)

    # Determine stem direction for each note
    for note in notes:
        sids = note.get("stem_ids", [])
        cy = float(note["center"][1])
        stem_dir = None
        max_beams = beam_count_by_note.get(str(note["id"]), 0)
        for sid in sids:
            stem = stems_by_id.get(str(sid))
            if stem is None:
                continue
            max_beams = max(max_beams, beam_count_by_stem.get(str(sid), 0))
            sy = float(stem["center"][1])
            stem_dir = "up" if sy < cy else "down"
        note["stem_direction"] = stem_dir
        note["beam_count"] = max_beams

        # Refined duration hypothesis with beam & dot evidence
        dots_n = int(note.get("dots", 0))
        if note.get("shape_class") == "hollow_head":
            if sids:
                base_type = "half"
                cand = "dotted_half_note_candidate" if dots_n else "half_note_candidate"
                basis = "hollow_notehead_with_stem"
            else:
                base_type = "whole"
                cand = "whole_note_candidate"
                basis = "hollow_notehead_without_detected_stem"
        elif max_beams >= 2:
            base_type = "16th"
            cand = "dotted_sixteenth_note_candidate" if dots_n else "sixteenth_note_candidate"
            basis = "filled_notehead_with_double_beam"
        elif max_beams == 1:
            base_type = "eighth"
            cand = "dotted_eighth_note_candidate" if dots_n else "eighth_note_candidate"
            basis = "filled_notehead_with_single_beam"
        elif sids:
            base_type = "quarter"
            cand = "dotted_quarter_note_candidate" if dots_n else "quarter_or_shorter_candidate"
            basis = "filled_notehead_with_stem; beam_and_flag_count_unknown"
        else:
            base_type = "quarter"
            cand = "unresolved"
            basis = "filled_notehead_without_detected_stem"

        note["inferred_note_type"] = base_type
        note["duration_hypothesis"] = {
            "candidate": cand,
            "inferred_type": base_type,
            "beam_count": max_beams,
            "dots": dots_n,
            "basis": basis,
            "confidence_calibrated": False,
            "used_for_musicxml_duration": False,
        }

    # Group notes by (staff_index, measure_index) to infer chords and voices
    notes_by_staff_measure: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for note in notes:
        s_idx = int(note["staff_index"])
        x = float(note["center"][0])
        bars = sorted(float(b) for b in barlines_by_staff.get(s_idx, []))
        m_idx = sum(1 for b in bars if x > b)
        note["inferred_measure_index"] = m_idx
        notes_by_staff_measure[(s_idx, m_idx)].append(note)

    chord_relations: list[dict] = []
    voice_relations: list[dict] = []
    chord_counter = 0

    for (s_idx, _m_idx), m_notes in notes_by_staff_measure.items():
        if s_idx >= len(staffs) or not m_notes:
            continue
        sp = float(staffs[s_idx].spacing)
        m_notes.sort(key=lambda n: (float(n["center"][0]), float(n["center"][1])))

        # Cluster simultaneous x-columns (within 0.42 * sp or sharing a stem)
        columns: list[list[dict]] = []
        for note in m_notes:
            placed = False
            nx = float(note["center"][0])
            nsids = set(note.get("stem_ids", []))
            for col in columns:
                col_x = float(np.mean([float(c["center"][0]) for c in col]))
                col_sids = {sid for c in col for sid in c.get("stem_ids", [])}
                if (nsids and (nsids & col_sids)) or abs(nx - col_x) <= max(2.5, 0.36 * sp):
                    col.append(note)
                    placed = True
                    break
            if not placed:
                columns.append([note])

        # Determine if this measure is polyphonic (2 voices):
        # A column with opposing stem directions OR multiple columns where upper/lower streams split
        has_two_voices = False
        for col in columns:
            dirs = {c.get("stem_direction") for c in col if c.get("stem_direction") is not None}
            if len(dirs) >= 2:
                has_two_voices = True
                break
            # Or two vertically separated noteheads in the same column that do NOT share a stem
            # and have distinct stems or > 2.2 sp vertical separation
            if len(col) >= 2:
                shared = set(col[0].get("stem_ids", []))
                for other in col[1:]:
                    shared &= set(other.get("stem_ids", []))
                y_span = max(float(c["center"][1]) for c in col) - min(float(c["center"][1]) for c in col)
                if not shared and y_span >= 1.8 * sp and any(c.get("stem_ids") for c in col):
                    has_two_voices = True
                    break

        for col in columns:
            # Sort column from highest pitch (smallest y) to lowest pitch (largest y)
            col.sort(key=lambda c: float(c["center"][1]))
            if len(col) == 1:
                only = col[0]
                if only.get("voice") is None:
                    if has_two_voices and only.get("stem_direction") == "down":
                        only["inferred_voice"] = 2
                    else:
                        only["inferred_voice"] = 1
            else:
                # Check if all notes in the column share a stem or form a single chord
                dirs = {c.get("stem_direction") for c in col if c.get("stem_direction") is not None}
                shared_stems = set(col[0].get("stem_ids", []))
                for other in col[1:]:
                    shared_stems &= set(other.get("stem_ids", []))

                if len(dirs) >= 2 or (has_two_voices and not shared_stems and len(col) == 2 and abs(float(col[1]["center"][1]) - float(col[0]["center"][1])) >= 1.8 * sp):
                    # Split column into Voice 1 (upper / up-stem) and Voice 2 (lower / down-stem)
                    v1_group = []
                    v2_group = []
                    mid_y = 0.5 * (float(col[0]["center"][1]) + float(col[-1]["center"][1]))
                    for c in col:
                        if c.get("stem_direction") == "up":
                            v1_group.append(c)
                        elif c.get("stem_direction") == "down":
                            v2_group.append(c)
                        elif float(c["center"][1]) <= mid_y:
                            v1_group.append(c)
                        else:
                            v2_group.append(c)
                    if not v1_group and v2_group:
                        v1_group.append(v2_group.pop(0))
                    if not v2_group and len(v1_group) >= 2:
                        v2_group.append(v1_group.pop(-1))
                    for c in v1_group:
                        c["inferred_voice"] = 1
                    for c in v2_group:
                        c["inferred_voice"] = 2
                        voice_relations.append({
                            "type": "simultaneous_voice_counterpart",
                            "source": c["id"],
                            "target": v1_group[0]["id"],
                        })
                    for v_group in (v1_group, v2_group):
                        if len(v_group) >= 2:
                            chord_counter += 1
                            cid = f"chord-{chord_counter:04d}"
                            for member in v_group:
                                member["inferred_chord_id"] = cid
                            for member in v_group[1:]:
                                chord_relations.append({
                                    "type": "chord_member_with",
                                    "source": member["id"],
                                    "target": v_group[0]["id"],
                                })
                else:
                    # Single chord in one voice
                    chord_counter += 1
                    cid = f"chord-{chord_counter:04d}"
                    v_num = 2 if (has_two_voices and dirs == {"down"}) else 1
                    for member in col:
                        member["inferred_voice"] = v_num
                        member["inferred_chord_id"] = cid
                    for member in col[1:]:
                        chord_relations.append({
                            "type": "chord_member_with",
                            "source": member["id"],
                            "target": col[0]["id"],
                        })

    return {
        "chord_count": chord_counter,
        "chord_relations": chord_relations,
        "voice_relations": voice_relations,
    }


def group_staffs_into_systems(
    gray: np.ndarray,
    binary: np.ndarray,
    staffs: list[StaffSystem],
) -> list[dict]:
    """Group five-line staves into multi-staff systems and infer historical or modern clefs."""
    if not staffs:
        return []

    from .historical_clefs import detect_clef_glyph_at_staff_start

    n = len(staffs)
    if n == 1:
        staff0 = staffs[0]
        clef_det = detect_clef_glyph_at_staff_start(
            gray,
            binary,
            staff_lines_y=staff0.lines,
            spacing=staff0.spacing,
        )
        if clef_det.get("detected") and clef_det.get("confidence", 0.0) >= 0.75:
            clef = {
                "sign": clef_det["sign"],
                "line": clef_det["line"],
                "clef_id": clef_det.get("clef_id"),
                "detected_visually": True,
                "confidence": clef_det["confidence"],
                "bbox_px": clef_det.get("bbox_px"),
            }
        else:
            clef = {"sign": "G", "line": 2, "clef_id": "treble"}
        return [{
            "staff_index": 0,
            "system_index": 0,
            "staff_number_in_system": 1,
            "inferred_clef": clef,
        }]

    # Check vertical gaps between consecutive staves and left-edge connector ink
    gaps = [float(staffs[i + 1].top - staffs[i].bottom) for i in range(n - 1)]
    med_sp = float(np.median([s.spacing for s in staffs]))

    # Check if staves form a 4-staff choral system (SATB: Sopran, Alto, Tenor, Bas)
    is_four_staff_satb = False
    if n == 4:
        even_gaps = [gaps[i] for i in range(0, len(gaps), 2)]
        odd_gaps = [gaps[i] for i in range(1, len(gaps), 2)]
        is_alternating = bool(odd_gaps and np.median(even_gaps) < 0.88 * np.median(odd_gaps))

        x_left_band = binary[staffs[0].top : staffs[3].bottom + 1, : max(10, gray.shape[1] // 6)]
        col_cov = (x_left_band > 0).mean(axis=0) if x_left_band.size else np.array([0.0])
        # SATB choir scores have a continuous left bracket/barline spanning all 4 staves without alternating inter-system gaps
        if float(np.max(col_cov)) >= 0.72 and not is_alternating:
            is_four_staff_satb = True

    # Check if staves pair up into 2-staff systems (grand staff / choir system)
    paired_two_staff = False
    if not is_four_staff_satb and n % 2 == 0 and n >= 2:
        if n == 2:
            # Check if left barline/brace connects the two staves
            x_left_band = binary[staffs[0].top : staffs[1].bottom + 1, : max(10, gray.shape[1] // 6)]
            col_cov = (x_left_band > 0).mean(axis=0)
            if float(np.max(col_cov)) >= 0.78 or gaps[0] <= 7.5 * med_sp:
                paired_two_staff = True
        else:
            even_gaps = [gaps[i] for i in range(0, len(gaps), 2)]
            odd_gaps = [gaps[i] for i in range(1, len(gaps), 2)]
            # In choral hymnals, lyrics placed between Soprano/Alto and Tenor/Bass can make intra-system gaps
            # larger than inter-system gaps, or vice-versa. Both exhibit clear alternating gap structures.
            if odd_gaps and (np.median(even_gaps) < 0.88 * np.median(odd_gaps) or np.median(odd_gaps) < 0.88 * np.median(even_gaps)):
                paired_two_staff = True
            elif float(np.max(gaps) - np.min(gaps)) <= 2.5 * med_sp:
                # Evenly spaced grand-staff systems (common in hymnals/choir scores)
                paired_two_staff = True

    satb_names = ["Sopran", "Alto", "Tenor", "Bas"]
    metadata: list[dict] = []
    for idx, _staff in enumerate(staffs):
        if is_four_staff_satb:
            sys_idx = idx // 4
            staff_num = (idx % 4) + 1
            voice_name = satb_names[staff_num - 1]
            part_id = f"P{staff_num}"
            part_name = voice_name
            satb_default_clefs = [
                {"sign": "G", "line": 2, "clef_id": "treble"},
                {"sign": "G", "line": 2, "clef_id": "treble"},
                {"sign": "G", "line": 2, "clef_id": "treble"},
                {"sign": "F", "line": 4, "clef_id": "bass"},
            ]
            default_clef = satb_default_clefs[staff_num - 1]
        elif paired_two_staff:
            sys_idx = idx // 2
            staff_num = (idx % 2) + 1
            voice_name = "Sopran / Alto" if staff_num == 1 else "Tenor / Bas"
            part_id = f"P{staff_num}"
            part_name = voice_name
            default_clef = {"sign": "G", "line": 2, "clef_id": "treble"} if staff_num == 1 else {"sign": "F", "line": 4, "clef_id": "bass"}
        else:
            sys_idx = idx
            staff_num = 1
            voice_name = f"Voce {idx + 1}"
            part_id = f"P{idx + 1}"
            part_name = voice_name
            default_clef = {"sign": "G", "line": 2, "clef_id": "treble"}

        clef_det = detect_clef_glyph_at_staff_start(
            gray,
            binary,
            staff_lines_y=_staff.lines,
            spacing=_staff.spacing,
        )
        if clef_det.get("detected") and clef_det.get("confidence", 0.0) >= 0.75:
            clef = {
                "sign": clef_det["sign"],
                "line": clef_det["line"],
                "clef_id": clef_det.get("clef_id"),
                "detected_visually": True,
                "confidence": clef_det["confidence"],
                "bbox_px": clef_det.get("bbox_px"),
            }
        else:
            clef = default_clef

        metadata.append({
            "staff_index": idx,
            "system_index": sys_idx,
            "staff_number_in_system": staff_num,
            "voice_name": voice_name,
            "part_id": part_id,
            "part_name": part_name,
            "is_satb_choral": is_four_staff_satb,
            "inferred_clef": clef,
        })
    return metadata
