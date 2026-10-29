"""Image ingestion and geometry-first score layout proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StaffSystem:
    index: int
    lines: tuple[int, int, int, int, int]
    spacing: float

    @property
    def top(self) -> int:
        return self.lines[0]

    @property
    def bottom(self) -> int:
        return self.lines[-1]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["top"] = self.top
        result["bottom"] = self.bottom
        return result


def load_grayscale(path: str | Path, page: int = 0, dpi: int = 200) -> np.ndarray:
    """Load the selected raster image or PDF page as an 8-bit grayscale array."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF support requires PyMuPDF; install the project dependencies.") from exc
        with fitz.open(file_path) as document:
            if page < 0 or page >= len(document):
                raise ValueError(f"PDF page index {page} is outside 0..{len(document) - 1}.")
            pix = document[page].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            encoded = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError("Could not decode the rasterized PDF page.")
            return gray

    encoded = np.frombuffer(file_path.read_bytes(), dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unsupported or corrupt image file: {file_path}")
    return gray


def binarize(gray: np.ndarray) -> np.ndarray:
    """Return foreground ink as 255, using Otsu thresholding."""
    if gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def _merge_active_rows(active: np.ndarray, counts: np.ndarray, max_gap: int = 2) -> list[int]:
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    rows = []
    for group in groups:
        weights = counts[group].astype(np.float64)
        rows.append(int(round(np.average(group, weights=np.maximum(weights, 1)))))
    return rows


def detect_staff_systems(gray: np.ndarray) -> tuple[list[StaffSystem], np.ndarray, np.ndarray]:
    """Find five-line staff groups; return systems, binary ink mask and staff-line mask.

    This is a geometric proposal, not a full OMR decoder. It assumes approximately
    horizontal staff lines and currently does not rectify page curl or perspective.
    """
    binary = binarize(gray)
    height, width = gray.shape
    kernel_width = max(25, width // 12)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    staff_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    row_counts = (staff_mask > 0).sum(axis=1)
    active = row_counts >= max(18, int(width * 0.28))
    rows = _merge_active_rows(active, row_counts, max_gap=2)

    groups: list[StaffSystem] = []
    i = 0
    while i + 4 < len(rows):
        candidate = rows[i:i + 5]
        gaps = np.diff(candidate).astype(float)
        spacing = float(np.median(gaps))
        if spacing >= 3.0 and spacing <= max(60.0, height / 15):
            relative_error = float(np.max(np.abs(gaps - spacing)) / spacing)
            if relative_error <= 0.28:
                groups.append(StaffSystem(
                    index=len(groups),
                    lines=tuple(candidate),
                    spacing=spacing,
                ))
                i += 5
                continue
        i += 1

    return groups, binary, staff_mask


def detect_barlines(binary: np.ndarray, staff: StaffSystem) -> list[int]:
    """Propose barline x-coordinates by measuring continuous vertical ink in staff gaps."""
    _, width = binary.shape
    top, bottom = staff.top, staff.bottom
    kernel_height = max(7, int(round(staff.spacing * 2.8)))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    counts = (vertical[top:bottom + 1, :] > 0).sum(axis=0)
    candidates = np.flatnonzero(counts >= max(6, int(staff.spacing * 2.1)))
    if len(candidates) == 0:
        return []
    groups: list[list[int]] = [[int(candidates[0])]]
    for x in candidates[1:]:
        x = int(x)
        if x - groups[-1][-1] <= 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    positions = [int(round(np.mean(group))) for group in groups]
    return [x for x in positions if 2 < x < width - 3]


def extract_notehead_candidates(
    gray: np.ndarray,
    binary: np.ndarray,
    staff_mask: np.ndarray,
    staffs: list[StaffSystem],
) -> list[dict]:
    """Propose notehead centers from line-suppressed connected components.

    Staff lines can split a head into fragments; long components can also include
    attached stems. For tall components, a horizontal-support profile trims narrow
    stem rows and estimates the compact head region. The returned box is a geometry
    proposal; the untouched source pixels remain the identity anchor.
    """
    if not staffs:
        return []
    spacing = float(np.median([staff.spacing for staff in staffs]))
    clean = cv2.subtract(binary, staff_mask)
    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)

    components: list[dict] = []
    min_area = max(3, int(spacing * spacing * 0.035))
    max_area = int(spacing * spacing * 3.0)
    for component in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[component]]
        cx, cy = [float(value) for value in centroids[component]]
        if area < min_area or area > max_area:
            continue
        if not (0.52 * spacing <= w <= 2.0 * spacing):
            continue

        # A component whose stem does not cross another staff line can be tall.
        # In its row profile the head has broad support, while the stem is narrow.
        if h > 1.4 * spacing:
            region = labels[y:y + h, x:x + w] == component
            row_counts = region.sum(axis=1)
            active_rows = np.flatnonzero(row_counts >= max(4, int(0.20 * spacing)))
            if len(active_rows) < max(3, int(0.30 * spacing)):
                continue
            row_top, row_bottom = int(active_rows.min()), int(active_rows.max())
            head_band = region[row_top:row_bottom + 1]
            column_counts = head_band.sum(axis=0)
            active_columns = np.flatnonzero(column_counts >= max(2, int(0.12 * spacing)))
            if len(active_columns) < max(3, int(0.45 * spacing)):
                continue
            col_left, col_right = int(active_columns.min()), int(active_columns.max())
            x_new, y_new = x + col_left, y + row_top
            w_new, h_new = col_right - col_left + 1, row_bottom - row_top + 1
            area_new = int(region[row_top:row_bottom + 1, col_left:col_right + 1].sum())
            cx = x + (col_left + col_right) / 2.0
            cy = y + (row_top + row_bottom) / 2.0
            x, y, w, h, area = x_new, y_new, w_new, h_new, area_new

        if not (0.22 * spacing <= h <= 1.45 * spacing):
            continue
        if not (0.45 <= w / max(h, 1) <= 3.8):
            continue
        staff_index = min(
            range(len(staffs)),
            key=lambda idx: min(
                abs(cy - staffs[idx].top),
                abs(cy - staffs[idx].bottom),
                abs(cy - (staffs[idx].top + staffs[idx].bottom) / 2),
            ),
        )
        staff = staffs[staff_index]
        if cy < staff.top - 2.0 * spacing or cy > staff.bottom + 2.0 * spacing:
            continue
        components.append({
            "box": [x, y, w, h],
            "center": [cx, cy],
            "area": area,
            "staff_index": staff_index,
        })

    # A staffline can split a single head into two wide fragments. Merge only
    # vertically separated boxes; overlapping components may be separate chord tones.
    used: set[int] = set()
    merged: list[dict] = []
    for i, first in enumerate(components):
        if i in used:
            continue
        ax, ay, aw, ah = first["box"]
        acx, acy = first["center"]
        best_j: int | None = None
        best_score: float | None = None
        for j in range(i + 1, len(components)):
            if j in used:
                continue
            second = components[j]
            if first["staff_index"] != second["staff_index"]:
                continue
            bx, by, bw, bh = second["box"]
            bcx, bcy = second["center"]
            horizontal_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) / max(1, min(aw, bw))
            vertical_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
            dx, dy = abs(acx - bcx), abs(acy - bcy)
            if (
                horizontal_overlap >= 0.65
                and dx < 0.38 * spacing
                and 0.22 * spacing < dy < 1.15 * spacing
                and 0 < vertical_gap <= 0.42 * spacing
            ):
                score = dx / spacing + vertical_gap / spacing
                if best_score is None or score < best_score:
                    best_score, best_j = score, j

        if best_j is not None:
            second = components[best_j]
            total_area = first["area"] + second["area"]
            center_x = (
                first["center"][0] * first["area"] + second["center"][0] * second["area"]
            ) / total_area
            center_y = (
                first["center"][1] * first["area"] + second["center"][1] * second["area"]
            ) / total_area
            merged.append({
                "center": [center_x, center_y],
                "area": total_area,
                "staff_index": first["staff_index"],
            })
            used.update((i, best_j))

    for i, component in enumerate(components):
        if i in used:
            continue
        merged.append({
            "center": component["center"],
            "area": component["area"],
            "staff_index": component["staff_index"],
        })

    proposals: list[dict] = []
    head_width = max(5, int(round(1.16 * spacing)))
    head_height = max(3, int(round(0.76 * spacing)))
    crop_half_width = max(5, int(round(1.1 * spacing)))
    crop_half_height = max(5, int(round(0.9 * spacing)))
    for item in merged:
        cx, cy = item["center"]
        x = max(0, int(round(cx - head_width / 2)))
        y = max(0, int(round(cy - head_height / 2)))
        x1 = min(gray.shape[1], x + head_width)
        y1 = min(gray.shape[0], y + head_height)
        cx_int, cy_int = int(round(cx)), int(round(cy))
        crop_x0 = max(0, cx_int - crop_half_width)
        crop_y0 = max(0, cy_int - crop_half_height)
        crop_x1 = min(gray.shape[1], cx_int + crop_half_width + 1)
        crop_y1 = min(gray.shape[0], cy_int + crop_half_height + 1)
        proposals.append({
            "bbox": [x, y, x1 - x, y1 - y],
            "center": [round(cx, 3), round(cy, 3)],
            "area_px": int(item["area"]),
            "aspect_ratio": round(head_width / max(head_height, 1), 4),
            "staff_index": int(item["staff_index"]),
            "crop": gray[crop_y0:crop_y1, crop_x0:crop_x1],
        })

    proposals.sort(key=lambda item: (item["staff_index"], item["center"][0], item["center"][1]))
    kept: list[dict] = []
    for candidate in proposals:
        cx, cy = candidate["center"]
        if any(
            candidate["staff_index"] == old["staff_index"]
            and np.hypot(cx - old["center"][0], cy - old["center"][1]) < 0.4 * spacing
            for old in kept
        ):
            continue
        kept.append(candidate)
    for index, candidate in enumerate(kept):
        candidate["id"] = f"proposal-{index:04d}"
    return kept
def staff_step_and_pitch(cy: float, staff: StaffSystem) -> tuple[int, str, int]:
    """Map a notehead center to a diatonic treble-clef pitch proposal."""
    step = int(round((staff.bottom - cy) / (staff.spacing / 2.0)))
    diatonic = ["C", "D", "E", "F", "G", "A", "B"]
    absolute_index = 2 + step  # bottom line of treble staff is E4
    octave_delta, letter_index = divmod(absolute_index, 7)
    return step, diatonic[letter_index], 4 + octave_delta
