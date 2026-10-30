"""Universal Historical Clef Engine for IDSR (Identic Digital Score Reconstruction).

Implements the finite mathematical grammar of Western staff notation across all 10 canonical
historical and modern clefs (15th - 21st century):
- French Violin Clef (G on line 1)
- Treble Clef (G on line 2)
- Soprano Clef (C on line 1)
- Mezzo-Soprano Clef (C on line 2)
- Alto Clef (C on line 3)
- Tenor Clef (C on line 4)
- Baritone Clef in C (C on line 5)
- Baritone Clef in F (F on line 3)
- Bass Clef (F on line 4)
- Sub-Bass Clef (F on line 5)

Mathematical Invariant:
Lines are indexed 1..5 from bottom to top. Step s = 0 is line 1, s = 8 is line 5.
Any clef (Sign S, Line L) is an exact anchor bijection mapping step s to an absolute
diatonic index D(s) = D_anchor + (s - 2 * (L - 1)).
Absolute index D yields octave = D // 7, step = DIATONIC[D % 7] with zero ambiguity.
"""

from __future__ import annotations

import math
from typing import Any
import cv2
import numpy as np


DIATONIC_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]

# Base diatonic indices (C0 = 0, C4 / Middle C = 28)
ANCHOR_DIATONIC = {
    "G": 32,  # G4 (32 = 7 * 4 + 4)
    "C": 28,  # C4 / Middle C (28 = 7 * 4 + 0)
    "F": 24,  # F3 (24 = 7 * 3 + 3)
}

HISTORICAL_CLEF_DEFINITIONS: dict[str, dict[str, Any]] = {
    "french_violin": {
        "id": "french_violin",
        "name": "French Violin Clef",
        "sign": "G",
        "line": 1,
        "anchor_diatonic": 32,
        "bottom_line_pitch": ("G", 4),
        "typical_repertoire": "French Baroque (Lully, Rameau, Couperin), violin parts",
        "plausible_diatonic_range": (30, 52),  # E4 to D7
    },
    "treble": {
        "id": "treble",
        "name": "Treble Clef",
        "sign": "G",
        "line": 2,
        "anchor_diatonic": 32,
        "bottom_line_pitch": ("E", 4),
        "typical_repertoire": "Standard classical/modern treble, violin, flute, soprano",
        "plausible_diatonic_range": (28, 50),  # C4 to B6
    },
    "soprano": {
        "id": "soprano",
        "name": "Soprano Clef",
        "sign": "C",
        "line": 1,
        "anchor_diatonic": 28,
        "bottom_line_pitch": ("C", 4),
        "typical_repertoire": "Renaissance/Baroque vocal soprano, Bach cantatas, Palestrina",
        "plausible_diatonic_range": (27, 43),  # B3 to A5
    },
    "mezzo_soprano": {
        "id": "mezzo_soprano",
        "name": "Mezzo-Soprano Clef",
        "sign": "C",
        "line": 2,
        "anchor_diatonic": 28,
        "bottom_line_pitch": ("A", 3),
        "typical_repertoire": "Renaissance/Baroque vocal mezzo-soprano, second cantus",
        "plausible_diatonic_range": (25, 41),  # G3 to F5
    },
    "alto": {
        "id": "alto",
        "name": "Alto Clef",
        "sign": "C",
        "line": 3,
        "anchor_diatonic": 28,
        "bottom_line_pitch": ("F", 3),
        "typical_repertoire": "Viola, vocal alto, alto trombone, English madrigals",
        "plausible_diatonic_range": (23, 39),  # E3 to D5
    },
    "tenor": {
        "id": "tenor",
        "name": "Tenor Clef",
        "sign": "C",
        "line": 4,
        "anchor_diatonic": 28,
        "bottom_line_pitch": ("D", 3),
        "typical_repertoire": "Cello, bassoon, trombone upper register, vocal tenor",
        "plausible_diatonic_range": (20, 36),  # B2 to A4
    },
    "baritone_c": {
        "id": "baritone_c",
        "name": "Baritone Clef (C)",
        "sign": "C",
        "line": 5,
        "anchor_diatonic": 28,
        "bottom_line_pitch": ("B", 2),
        "typical_repertoire": "16th-century vocal baritone / quintus parts",
        "plausible_diatonic_range": (18, 34),  # G2 to F4
    },
    "baritone_f": {
        "id": "baritone_f",
        "name": "Baritone Clef (F)",
        "sign": "F",
        "line": 3,
        "anchor_diatonic": 24,
        "bottom_line_pitch": ("B", 2),
        "typical_repertoire": "Baroque vocal baritone (Telemann, C.P.E. Bach)",
        "plausible_diatonic_range": (18, 34),  # G2 to F4
    },
    "bass": {
        "id": "bass",
        "name": "Bass Clef",
        "sign": "F",
        "line": 4,
        "anchor_diatonic": 24,
        "bottom_line_pitch": ("G", 2),
        "typical_repertoire": "Standard modern bass, cello, double bass, bass voice",
        "plausible_diatonic_range": (15, 32),  # D2 to G4
    },
    "sub_bass": {
        "id": "sub_bass",
        "name": "Sub-Bass Clef",
        "sign": "F",
        "line": 5,
        "anchor_diatonic": 24,
        "bottom_line_pitch": ("E", 2),
        "typical_repertoire": "Franco-Flemish polyphony (Ockeghem, Josquin contra-tenor bassus)",
        "plausible_diatonic_range": (13, 29),  # B1 to E4
    },
}


def diatonic_step_to_pitch(
    step: int,
    clef_sign: str = "G",
    clef_line: int = 2,
) -> tuple[str, int, int]:
    """Map a staff step s (relative to line 1 = 0) to (letter, octave, diatonic_index).

    Exact bijection:
    D(s) = D_anchor(sign) + (s - 2 * (clef_line - 1))
    """
    sign_upper = clef_sign.upper()
    anchor_diat = ANCHOR_DIATONIC.get(sign_upper, 32)
    clef_line_clamped = max(1, min(5, int(clef_line)))

    # Step s_anchor corresponding to the clef's anchor line
    s_anchor = 2 * (clef_line_clamped - 1)
    abs_diatonic = anchor_diat + (int(step) - s_anchor)

    octave, letter_idx = divmod(abs_diatonic, 7)
    letter = DIATONIC_LETTERS[letter_idx]
    return letter, int(octave), int(abs_diatonic)


def staff_step_and_pitch_with_clef(
    cy: float,
    staff_bottom: float,
    staff_spacing: float,
    clef_sign: str = "G",
    clef_line: int = 2,
) -> tuple[int, str, int, int]:
    """Compute (step, letter, octave, abs_diatonic_index) for a notehead center cy.

    step is measured from the bottom staff line (line 1 = 0, line 5 = 8).
    """
    sp = float(staff_spacing) if staff_spacing > 0.0 else 16.0
    continuous_step = (float(staff_bottom) - float(cy)) / (sp / 2.0)
    step = int(round(continuous_step))
    letter, octave, abs_diat = diatonic_step_to_pitch(step, clef_sign=clef_sign, clef_line=clef_line)
    return step, letter, octave, abs_diat


def identify_clef_definition(sign: str, line: int) -> dict[str, Any]:
    """Lookup standard historical clef metadata by sign ('G', 'C', 'F') and line (1..5)."""
    s_clean = sign.upper()
    l_clean = int(line)
    for c_id, c_data in HISTORICAL_CLEF_DEFINITIONS.items():
        if c_data["sign"] == s_clean and c_data["line"] == l_clean:
            return c_data
    return {
        "id": f"custom_{s_clean.lower()}{l_clean}",
        "name": f"Custom {s_clean}-Clef on Line {l_clean}",
        "sign": s_clean,
        "line": l_clean,
        "anchor_diatonic": ANCHOR_DIATONIC.get(s_clean, 28),
        "bottom_line_pitch": diatonic_step_to_pitch(0, s_clean, l_clean)[:2],
        "typical_repertoire": "Non-standard historical or transposed clef",
        "plausible_diatonic_range": (15, 45),
    }


def detect_clef_glyph_at_staff_start(
    gray: np.ndarray,
    binary: np.ndarray,
    staff_lines_y: list[float],
    spacing: float,
    x_start: float | None = None,
) -> dict[str, Any]:
    """Detect visual clef glyph at the left margin of a staff system.

    Distinguishes:
    - G-clef: Tall vertical ascender/descender (> 4.8 * sp), central loop circling line L (L=2 or L=1).
    - F-clef: Backward curve with two distinct dots flanking line L (L=4 or L=3).
    - C-clef: Two vertical bars/lobes with central indent/cusp pointing at line L (L in 1..5).
    """
    h_img, w_img = gray.shape[:2]
    sp = float(spacing) if spacing > 4.0 else 16.0
    lines_sorted = sorted(float(y) for y in staff_lines_y)
    if len(lines_sorted) < 2:
        return {"detected": False, "sign": "G", "line": 2, "confidence": 0.5, "reason": "Insufficient staff lines"}

    top_line = lines_sorted[0]
    bot_line = lines_sorted[-1]
    n_lines = len(lines_sorted)

    # Determine left edge of staff
    if x_start is None:
        # Scan binary ink profile along staff lines to find staff start
        staff_band = binary[max(0, int(top_line - sp)): min(h_img, int(bot_line + sp)), :]
        col_sums = np.sum(staff_band > 0, axis=0)
        active_cols = np.flatnonzero(col_sums >= max(2, int(0.6 * n_lines)))
        x_left = float(active_cols[0]) if len(active_cols) else 0.0
    else:
        x_left = float(x_start)

    # ROI for clef: extends from x_left to x_left + 4.2 * sp, vertically [top_line - 2.5*sp, bot_line + 2.5*sp]
    x0 = max(0, int(round(x_left)))
    x1 = min(w_img, int(round(x_left + 4.2 * sp)))
    y0 = max(0, int(round(top_line - 2.5 * sp)))
    y1 = min(h_img, int(round(bot_line + 2.5 * sp)))

    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return {"detected": False, "sign": "G", "line": 2, "confidence": 0.5, "reason": "Invalid ROI"}

    roi_bin = binary[y0:y1, x0:x1].copy()
    roi_gray = gray[y0:y1, x0:x1]

    # Suppress thin horizontal staff lines with morphological opening so they don't merge with vertical barlines,
    # while preserving vertical strokes, cusps, and crossbars of clef glyphs intact.
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.2 * sp), 1))
    staff_lines_roi = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, line_kernel)
    roi_clean = cv2.subtract(roi_bin, staff_lines_roi)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    # Find connected components inside ROI
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_clean, 8)
    if count <= 1:
        return {"detected": False, "sign": "G", "line": 2, "confidence": 0.5, "reason": "No ink in clef ROI"}

    # Filter for components with significant vertical height AND width (clefs are wide, barlines are narrow)
    min_clef_h = 2.2 * sp
    min_clef_w = max(4, int(0.55 * sp))
    valid_clef_comps = []
    for comp_idx in range(1, count):
        bx, by, bw, bh, area = [int(v) for v in stats[comp_idx]]
        if bh >= min_clef_h and bw >= min_clef_w and area >= 0.8 * sp * sp:
            valid_clef_comps.append((comp_idx, bx, by, bw, bh, area, centroids[comp_idx]))

    if not valid_clef_comps:
        # Fallback only if largest area component has non-trivial width (not a thin barline)
        largest_idx = max(range(1, count), key=lambda idx: stats[idx][4])
        bx, by, bw, bh, area = [int(v) for v in stats[largest_idx]]
        if bw >= min_clef_w:
            valid_clef_comps = [(largest_idx, bx, by, bw, bh, area, centroids[largest_idx])]
        else:
            return {"detected": False, "sign": "G", "line": 2, "confidence": 0.5, "reason": "No clef glyph detected"}

    # Choose leftmost prominent component as primary clef glyph
    primary = min(valid_clef_comps, key=lambda c: c[1])
    c_idx, bx, by, bw, bh, area, (cx_local, cy_local) = primary
    cx_global = x0 + cx_local
    cy_global = y0 + cy_local
    total_h = float(bh)

    comp_mask = (labels == c_idx).astype(np.uint8)

    # 1. Test for G-clef (Treble / French Violin):
    # Very tall (typically >= 5.0 * sp), with a wide central loop (bw >= 1.35 * sp)
    # extends substantially above top line and below bottom line
    extends_above = (y0 + by) < (top_line - 1.2 * sp)
    extends_below = (y0 + by + bh) > (bot_line + 1.2 * sp)
    if total_h >= 4.8 * sp and bw >= 1.35 * sp and (extends_above or extends_below):
        # Determine anchor line: locate the central loop (widest horizontal portion)
        row_counts = np.sum(comp_mask, axis=1)
        widest_row = int(np.argmax(row_counts))
        loop_y_global = y0 + by + widest_row
        # Find closest staff line
        closest_line_idx = min(range(len(lines_sorted)), key=lambda l_idx: abs(lines_sorted[l_idx] - loop_y_global))
        # Line number from bottom (1 to 5)
        clef_line = len(lines_sorted) - closest_line_idx
        clef_line = 2 if clef_line not in (1, 2) else clef_line
        clef_id = "french_violin" if clef_line == 1 else "treble"
        return {
            "detected": True,
            "sign": "G",
            "line": clef_line,
            "clef_id": clef_id,
            "confidence": 0.94,
            "bbox_px": [x0 + bx, y0 + by, bw, bh],
            "center_px": [round(float(cx_global), 2), round(float(cy_global), 2)],
            "historical_info": HISTORICAL_CLEF_DEFINITIONS[clef_id],
        }

    # 2. Test for F-clef (Bass / Baritone F):
    # Features two dots stacked vertically adjacent to the curved head
    # Check for dots in binary image to the right of primary component
    dot_roi_x0 = bx + bw - 2
    dot_roi_x1 = min(roi_bin.shape[1], bx + bw + int(round(1.8 * sp)))
    two_dots_found = False
    f_dot_line = 4
    if dot_roi_x1 > dot_roi_x0:
        dot_strip = roi_bin[by:by + bh, dot_roi_x0:dot_roi_x1]
        n_dots, _, dot_stats, dot_centroids = cv2.connectedComponentsWithStats(dot_strip, 8)
        small_dots = []
        for d_i in range(1, n_dots):
            d_area = dot_stats[d_i][4]
            d_w, d_h = dot_stats[d_i][2], dot_stats[d_i][3]
            if 4 <= d_area <= 1.2 * sp * sp and max(d_w, d_h) <= 1.2 * sp:
                small_dots.append(dot_centroids[d_i][1])
        if len(small_dots) >= 2:
            small_dots.sort()
            # Mean y of the two dots is the anchor line
            dot_mid_y = y0 + by + float(np.mean(small_dots[:2]))
            closest_idx = min(range(len(lines_sorted)), key=lambda l_idx: abs(lines_sorted[l_idx] - dot_mid_y))
            f_dot_line = len(lines_sorted) - closest_idx
            two_dots_found = True

    if two_dots_found or (total_h >= 2.8 * sp and total_h < 4.8 * sp and cy_global < (top_line + bot_line) / 2):
        line_cand = f_dot_line if two_dots_found else 4
        clef_line = max(3, min(5, line_cand))
        clef_id = "baritone_f" if clef_line == 3 else ("sub_bass" if clef_line == 5 else "bass")
        return {
            "detected": True,
            "sign": "F",
            "line": clef_line,
            "clef_id": clef_id,
            "confidence": 0.92 if two_dots_found else 0.82,
            "bbox_px": [x0 + bx, y0 + by, bw, bh],
            "center_px": [round(float(cx_global), 2), round(float(cy_global), 2)],
            "historical_info": HISTORICAL_CLEF_DEFINITIONS[clef_id],
        }

    # 3. Test for C-clef (Soprano, Mezzo, Alto, Tenor, Baritone C):
    # Two vertical uprights connected by a central cusp pointing at line L.
    # C-clef glyphs require substantial height and width (at least 3.0 * sp high and 0.7 * sp wide)
    if total_h >= 3.0 * sp and total_h <= 4.8 * sp and bw >= 0.7 * sp:
        row_widths = np.sum(comp_mask, axis=1)
        active_rows = np.nonzero(row_widths)[0]
        if len(active_rows):
            mid_row_in_roi = int(np.mean(active_rows))
            cusp_y_global = y0 + mid_row_in_roi
        else:
            cusp_y_global = cy_global

        closest_line_idx = min(range(len(lines_sorted)), key=lambda l_idx: abs(lines_sorted[l_idx] - cusp_y_global))
        c_line = len(lines_sorted) - closest_line_idx
        c_line = max(1, min(5, c_line))

        c_map = {1: "soprano", 2: "mezzo_soprano", 3: "alto", 4: "tenor", 5: "baritone_c"}
        clef_id = c_map.get(c_line, "alto")

        return {
            "detected": True,
            "sign": "C",
            "line": c_line,
            "clef_id": clef_id,
            "confidence": 0.88,
            "bbox_px": [x0 + bx, y0 + by, bw, bh],
            "center_px": [round(float(cx_global), 2), round(float(cy_global), 2)],
            "historical_info": HISTORICAL_CLEF_DEFINITIONS[clef_id],
        }

    return {
        "detected": False,
        "sign": "G",
        "line": 2,
        "confidence": 0.5,
        "reason": "No confident clef glyph matched visual criteria; defaulting to system structural role",
    }


CLEF_PRIOR_PENALTY: dict[str, float] = {
    "treble": 0.0,
    "bass": 0.0,
    "alto": 0.15,
    "tenor": 0.20,
    "soprano": 0.25,
    "mezzo_soprano": 0.35,
    "baritone_f": 0.60,
    "french_violin": 0.70,
    "baritone_c": 0.80,
    "sub_bass": 0.85,
}

VOICE_TARGET_DIATONIC: dict[str, float] = {
    "soprano": 33.0,
    "mezzo_soprano": 31.0,
    "alto": 29.0,
    "tenor": 26.0,
    "baritone": 23.0,
    "bass": 20.0,
}


def disambiguate_clef_by_syntactic_constraints(
    staff_notes: list[dict],
    candidate_clefs: list[dict[str, Any]] | None = None,
    preferred_voice: str | None = None,
) -> dict[str, Any]:
    """Resolve clef ambiguity using the finite syntactic grammar of Western staff notation.

    Axiomatic principle:
    Historical scribes and printers chose the clef specifically to minimize ledger lines
    and maintain the melodic ambitus within the 5 staff lines (s in [0, 8]).
    Ledger penalty = sum(max(0, s - 8) + max(0, -s)).
    The clef with the minimal ledger penalty and best tessitura alignment is chosen deterministically.
    """
    if candidate_clefs is None:
        clef_pool = list(HISTORICAL_CLEF_DEFINITIONS.values())
    else:
        clef_pool = candidate_clefs

    if not staff_notes:
        # Default to standard treble
        return {
            "chosen_clef": HISTORICAL_CLEF_DEFINITIONS["treble"],
            "confidence": 0.5,
            "method": "default_empty_staff",
            "evaluations": [],
        }

    steps = [int(note.get("diatonic_step_from_bottom_line", 0)) for note in staff_notes]

    evaluations: list[dict[str, Any]] = []
    for c_def in clef_pool:
        c_sign = c_def["sign"]
        c_line = int(c_def["line"])
        min_range, max_range = c_def.get("plausible_diatonic_range", (15, 50))

        ledger_penalty = 0
        range_violations = 0
        diatonic_indices = []

        for s in steps:
            _, _, d_idx = diatonic_step_to_pitch(s, clef_sign=c_sign, clef_line=c_line)
            diatonic_indices.append(d_idx)
            # Ledger penalty: distance beyond staff bounds [0, 8]
            if s > 8:
                ledger_penalty += (s - 8)
            elif s < 0:
                ledger_penalty += (-s)
            # Physiological vocal/instrument range check
            if d_idx < min_range or d_idx > max_range:
                range_violations += 1

        mean_d = float(np.mean(diatonic_indices)) if diatonic_indices else 28.0
        # Score formulation: lower penalty is better
        score = float(ledger_penalty) + 3.0 * float(range_violations) + CLEF_PRIOR_PENALTY.get(c_def["id"], 1.0)
        if preferred_voice and preferred_voice.lower() in VOICE_TARGET_DIATONIC:
            target = VOICE_TARGET_DIATONIC[preferred_voice.lower()]
            score += 0.8 * abs(mean_d - target)

        evaluations.append({
            "clef": c_def,
            "score": round(score, 3),
            "ledger_penalty": ledger_penalty,
            "range_violations": range_violations,
            "mean_diatonic_index": mean_d,
        })

    evaluations.sort(key=lambda item: item["score"])
    best = evaluations[0]

    return {
        "chosen_clef": best["clef"],
        "confidence": round(1.0 / (1.0 + 0.1 * best["score"]), 3),
        "method": "finite_syntactic_constraint_satisfaction",
        "evaluations": evaluations,
    }
