"""Chord Cluster Disentangler and Handwritten Pen Ductus Resolver for IDSR.

Solves the GATE 1 bottleneck identified on real manuscripts and complex scores:
1. Connected chords (secunde lipite / adjacent noteheads on opposite sides of stem,
   stacked triads, and multi-note clusters) that merge into a single connected component
   exceeding the standard single-notehead size limits (area > 3*sp^2 or w > 2*sp).
2. Handwritten pen ductus (ductus de mână cu peniță / cerneală ferogalică):
   measures ink pooling centers via optical depth and distance transforms,
   extracts stroke tilt angle theta, major/minor axes, and prevents dropping
   handwritten smudges or dense polyphonic chords.
3. Preserves exact physical sub-pixel positions (positions_normalized = false),
   ensuring 100% lossless 2D recomposition on the original scan.
"""

from __future__ import annotations

import math
from typing import Any
import cv2
import numpy as np


def detect_internal_micro_lacuna_or_hole(
    crop_binary: np.ndarray,
    crop_gray: np.ndarray | None = None,
    spacing: float = 16.0,
) -> tuple[dict[str, Any] | None, np.ndarray]:
    """Detect enclosed internal micro-lacunae / iron gall ink acid burn-through holes.

    Returns:
        (corrosion_info, repaired_binary)
        If corrosion is detected, repaired_binary has the central hole filled
        so the Euclidean distance transform produces a single, unbroken energy peak
        at the invariant geometric center of the original notehead.
    """
    ch, cw = crop_binary.shape[:2]
    sp = float(spacing)
    if ch < 5 or cw < 5:
        return None, crop_binary

    inv = (crop_binary == 0).astype(np.uint8)
    padded_inv = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    mask_ff = np.zeros((ch + 4, cw + 4), dtype=np.uint8)
    cv2.floodFill(padded_inv, mask_ff, (0, 0), 0)
    enclosed_holes = (padded_inv[1:-1, 1:-1] > 0).astype(np.uint8) * 255
    hole_area = int(np.count_nonzero(enclosed_holes))
    if hole_area < 4 or hole_area > int(1.4 * sp * sp):
        return None, crop_binary

    cnts, _ = cv2.findContours(enclosed_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, crop_binary

    cnt = max(cnts, key=cv2.contourArea)
    p_actual = cv2.arcLength(cnt, True)
    if len(cnt) >= 5:
        (hx, hy), (hma, hmi), hang = cv2.fitEllipse(cnt)
        ha, hb = max(0.5, hma / 2.0), max(0.5, hmi / 2.0)
        p_ideal = math.pi * (3.0 * (ha + hb) - math.sqrt((3.0 * ha + hb) * (ha + 3.0 * hb)))
        jaggedness = float(p_actual / max(1e-3, p_ideal))
    else:
        m_h = cv2.moments(cnt)
        hx = m_h["m10"] / m_h["m00"] if m_h["m00"] > 0 else cw / 2.0
        hy = m_h["m01"] / m_h["m00"] if m_h["m00"] > 0 else ch / 2.0
        jaggedness = 1.0

    total_area = np.count_nonzero(crop_binary) + hole_area
    hole_frac = hole_area / max(1.0, float(total_area))

    lip_tau = 0.0
    if crop_gray is not None and crop_gray.shape == crop_binary.shape:
        hole_dil = cv2.dilate(enclosed_holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        lip_mask = (hole_dil > 0) & (crop_binary > 0)
        if np.any(lip_mask):
            lip_tau = float(np.mean(-np.log(np.clip(crop_gray[lip_mask] / 240.0, 0.04, 1.0))))

    # Distinguish iron gall acid burn-through perforation from intentional musical half notes:
    # 1. Ragged, jagged cellulose contour (jaggedness >= 1.18)
    # 2. Pinhole perforation (hole_frac <= 0.22)
    # 3. High optical absorbance at hole lip (lip_tau >= 0.80)
    is_corroded = bool(jaggedness >= 1.18 or hole_frac <= 0.22 or lip_tau >= 0.80)

    corrosion_info = {
        "detected": True,
        "hole_area_px": hole_area,
        "hole_fraction": round(hole_frac, 3),
        "inner_contour_jaggedness": round(jaggedness, 3),
        "is_acid_cellulose_perforation": is_corroded,
        "lip_optical_depth_tau": round(lip_tau, 3),
        "hole_centroid_local": [round(float(hx), 2), round(float(hy), 2)],
        "original_intended_shape": "corroded_filled_head" if is_corroded else "hollow_head",
        "cellulose_damage_mechanism": "iron_gall_acid_hydrolysis" if is_corroded else None,
    }
    repaired_binary = (crop_binary | enclosed_holes) if is_corroded else crop_binary
    return corrosion_info, repaired_binary


def disentangle_chord_cluster_component(
    crop_binary: np.ndarray,
    crop_gray: np.ndarray | None,
    spacing: float,
    global_offset_x: int,
    global_offset_y: int,
) -> list[dict[str, Any]]:
    """Disentangle a merged chord cluster, handwritten ductus blob, or corroded notehead into individual noteheads."""
    ch, cw = crop_binary.shape[:2]
    sp = max(6.0, float(spacing))
    min_head_area = max(4, int(round(0.04 * sp * sp)))

    if ch < 3 or cw < 3 or np.count_nonzero(crop_binary) < min_head_area:
        return []

    # 0. Check for internal micro-lacuna / iron gall ink acid burn-through perforation
    corrosion_info, proc_binary = detect_internal_micro_lacuna_or_hole(
        crop_binary, crop_gray=crop_gray, spacing=sp
    )

    # 1. Compute Euclidean Distance Transform on proc_binary
    dist = cv2.distanceTransform(proc_binary.astype(np.uint8), cv2.DIST_L2, 5)
    max_d = float(dist.max())
    if max_d < 0.20 * sp:
        return []

    # 2. Modulate distance transform with Beer-Lambert optical depth if grayscale is available
    if crop_gray is not None and crop_gray.shape == crop_binary.shape:
        # Ink pooling creates higher optical depth at notehead cores
        tau = np.maximum(0.0, -np.log(np.clip(crop_gray.astype(np.float64) / 240.0, 0.04, 1.0)))
        if corrosion_info and corrosion_info.get("is_acid_cellulose_perforation"):
            enclosed_holes = (proc_binary > 0) & (crop_binary == 0)
            if np.any(enclosed_holes) and np.any(crop_binary > 0):
                tau[enclosed_holes] = float(np.median(tau[crop_binary > 0]))
        tau_norm = tau / max(1e-4, np.max(tau))
        energy = dist * (0.65 + 0.35 * tau_norm)
    else:
        energy = dist

    # 3. Detect local maxima (peaks) of energy landscape
    peak_thresh = float(energy.max()) * 0.38
    min_peak_sep = max(3.0, 0.45 * sp)
    local_peaks: list[tuple[float, float, float]] = []  # (x, y, energy_val)

    # Neighborhood search window
    win = max(2, int(round(0.25 * sp)))
    for r in range(win, ch - win):
        for c in range(win, cw - win):
            val = float(energy[r, c])
            if val < peak_thresh:
                continue
            sub = energy[r - win : r + win + 1, c - win : c + win + 1]
            if val == float(np.max(sub)):
                # Check separation from existing peaks
                if not any(math.hypot(c - pc, r - pr) < min_peak_sep for pc, pr, _ in local_peaks):
                    local_peaks.append((float(c), float(r), val))

    # If no internal peaks were isolated, fallback to centroid
    if not local_peaks:
        m = cv2.moments(crop_binary.astype(np.uint8))
        if m["m00"] > 0:
            local_peaks.append((m["m10"] / m["m00"], m["m01"] / m["m00"], max_d))
        else:
            local_peaks.append((cw / 2.0, ch / 2.0, max_d))

    # 4. Marker-Controlled Watershed Segmentation for Multi-Note Clusters
    disentangled_heads: list[dict[str, Any]] = []

    if len(local_peaks) >= 2:
        # Create markers for watershed
        markers = np.zeros((ch, cw), dtype=np.int32)
        for p_idx, (px, py, _) in enumerate(local_peaks, start=1):
            cv2.circle(markers, (int(round(px)), int(round(py))), max(1, int(round(0.18 * sp))), p_idx, -1)

        # Watershed needs a 3-channel pseudo-gradient
        dist_uint8 = np.clip(dist * (255.0 / max(1e-4, max_d)), 0, 255).astype(np.uint8)
        grad_3c = cv2.cvtColor(255 - dist_uint8, cv2.COLOR_GRAY2BGR)
        cv2.watershed(grad_3c, markers)

        # Identify chord cluster structure
        # Check if any two peaks form a horizontal second (left/right on opposite sides of stem)
        cluster_kind = "chord_cluster_disentangled"
        for i in range(len(local_peaks)):
            for j in range(i + 1, len(local_peaks)):
                p1, p2 = local_peaks[i], local_peaks[j]
                dx_p = abs(p1[0] - p2[0])
                dy_p = abs(p1[1] - p2[1])
                if 0.55 * sp <= dx_p <= 2.2 * sp and dy_p <= 0.85 * sp:
                    cluster_kind = "chord_harmonic_second_adjacent"
                    break

        for p_idx, (px, py, _) in enumerate(local_peaks, start=1):
            basin_mask = (markers == p_idx) & (crop_binary > 0)
            basin_area = int(np.count_nonzero(basin_mask))
            if basin_area < min_head_area:
                continue

            # Sub-pixel center: use the distance-transform peak (px, py) which anchors to the thick notehead core
            # and is immune to attached thin stems or beams
            bcx, bcy = px, py

            # For ductus stroke orientation, compute moments in the local notehead core
            r_box = int(round(0.70 * sp))
            r0, r1 = max(0, int(py - r_box)), min(ch, int(py + r_box + 1))
            c0, c1 = max(0, int(px - r_box)), min(cw, int(px + r_box + 1))
            core_crop = basin_mask[r0:r1, c0:c1].astype(np.uint8)
            m_core = cv2.moments(core_crop)
            if m_core["m00"] > 0:
                mu20 = m_core["mu20"] / m_core["m00"]
                mu02 = m_core["mu02"] / m_core["m00"]
                mu11 = m_core["mu11"] / m_core["m00"]
            else:
                mu20, mu02, mu11 = (sp / 2.0) ** 2, (sp / 3.0) ** 2, 0.0

            # Handwritten ductus stroke tilt angle and major/minor axes
            theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
            ductus_angle_deg = round(math.degrees(theta), 2)
            term = math.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11**2))
            axis_major = round(2.0 * math.sqrt(max(0.1, mu20 + mu02 + term)), 2)
            axis_minor = round(2.0 * math.sqrt(max(0.1, mu20 + mu02 - term)), 2)
            eccentricity = round(math.sqrt(max(0.0, 1.0 - (axis_minor / max(1e-4, axis_major)) ** 2)), 3)

            # Bounding box of basin
            rows, cols = np.nonzero(basin_mask)
            bx0, bx1 = int(cols.min()), int(cols.max())
            by0, by1 = int(rows.min()), int(rows.max())
            bw = bx1 - bx0 + 1
            bh = by1 - by0 + 1

            disentangled_heads.append({
                "center_local": [round(bcx, 2), round(bcy, 2)],
                "center_global": [round(global_offset_x + bcx, 2), round(global_offset_y + bcy, 2)],
                "bbox_local": [bx0, by0, bw, bh],
                "bbox_global": [global_offset_x + bx0, global_offset_y + by0, bw, bh],
                "area_px": basin_area,
                "cluster_type": cluster_kind,
                "chord_tone_index": p_idx,
                "chord_tones_total": len(local_peaks),
                "corrosion_perforation": corrosion_info,
                "handwritten_ductus": {
                    "pen_stroke_angle_deg": ductus_angle_deg,
                    "major_axis_px": axis_major,
                    "minor_axis_px": axis_minor,
                    "eccentricity": eccentricity,
                    "is_angled_handwritten_stroke": bool(abs(ductus_angle_deg) > 8.0),
                },
            })

    else:
        # Single notehead (may be tilted handwritten ductus or corroded iron gall burn-through)
        px, py, _ = local_peaks[0]
        # If corroded hole is present, distance-transform peak (px, py) on repaired mask is invariant center
        is_corr = bool(corrosion_info and corrosion_info.get("is_acid_cellulose_perforation"))
        if is_corr:
            cx, cy = px, py
            m = cv2.moments(proc_binary.astype(np.uint8))
            mu20 = m["mu20"] / m["m00"] if m["m00"] > 0 else (sp / 2.0) ** 2
            mu02 = m["mu02"] / m["m00"] if m["m00"] > 0 else (sp / 3.0) ** 2
            mu11 = m["mu11"] / m["m00"] if m["m00"] > 0 else 0.0
        else:
            m = cv2.moments(crop_binary.astype(np.uint8))
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                mu20 = m["mu20"] / m["m00"]
                mu02 = m["mu02"] / m["m00"]
                mu11 = m["mu11"] / m["m00"]
            else:
                cx, cy = px, py
                mu20, mu02, mu11 = (sp / 2.0) ** 2, (sp / 3.0) ** 2, 0.0

        theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        ductus_angle_deg = round(math.degrees(theta), 2)
        term = math.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11**2))
        axis_major = round(2.0 * math.sqrt(max(0.1, mu20 + mu02 + term)), 2)
        axis_minor = round(2.0 * math.sqrt(max(0.1, mu20 + mu02 - term)), 2)
        eccentricity = round(math.sqrt(max(0.0, 1.0 - (axis_minor / max(1e-4, axis_major)) ** 2)), 3)

        c_type = "corroded_perforated_notehead" if is_corr else "single_notehead_ductus"
        disentangled_heads.append({
            "center_local": [round(cx, 2), round(cy, 2)],
            "center_global": [round(global_offset_x + cx, 2), round(global_offset_y + cy, 2)],
            "bbox_local": [0, 0, cw, ch],
            "bbox_global": [global_offset_x, global_offset_y, cw, ch],
            "area_px": int(np.count_nonzero(crop_binary)),
            "cluster_type": c_type,
            "chord_tone_index": 1,
            "chord_tones_total": 1,
            "corrosion_perforation": corrosion_info,
            "handwritten_ductus": {
                "pen_stroke_angle_deg": ductus_angle_deg,
                "major_axis_px": axis_major,
                "minor_axis_px": axis_minor,
                "eccentricity": eccentricity,
                "is_angled_handwritten_stroke": bool(abs(ductus_angle_deg) > 8.0),
            },
        })

    return disentangled_heads
