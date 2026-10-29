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
    """Find compact, approximately elliptical components after removing long staff lines."""
    if not staffs:
        return []
    spacings = [staff.spacing for staff in staffs]
    spacing = float(np.median(spacings))
    clean = cv2.subtract(binary, staff_mask)
    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    kernel_w = max(5, int(round(spacing * 0.78)))
    kernel_h = max(3, int(round(spacing * 0.42)))
    kernel_w += 1 - kernel_w % 2
    kernel_h += 1 - kernel_h % 2
    head_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
    opened = cv2.morphologyEx(clean, cv2.MORPH_OPEN, head_kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8)

    proposals: list[dict] = []
    min_area = max(3, int(spacing * spacing * 0.045))
    max_area = int(spacing * spacing * 2.2)
    for component in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[component]]
        cx, cy = [float(v) for v in centroids[component]]
        if area < min_area or area > max_area:
            continue
        if not (0.38 * spacing <= w <= 2.1 * spacing):
            continue
        if not (0.24 * spacing <= h <= 1.35 * spacing):
            continue
        aspect = w / max(h, 1)
        if not 0.62 <= aspect <= 3.2:
            continue
        staff_index = min(
            range(len(staffs)),
            key=lambda idx: min(abs(cy - staffs[idx].top), abs(cy - staffs[idx].bottom), abs(cy - (staffs[idx].top + staffs[idx].bottom) / 2)),
        )
        staff = staffs[staff_index]
        if cy < staff.top - 2.0 * spacing or cy > staff.bottom + 2.0 * spacing:
            continue
        pad = max(3, int(round(0.5 * spacing)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        crop = gray[y0:y1, x0:x1]
        proposals.append({
            "id": f"staff{staff_index}-component{component}",
            "bbox": [x, y, w, h],
            "center": [round(cx, 3), round(cy, 3)],
            "area_px": area,
            "aspect_ratio": round(aspect, 4),
            "staff_index": staff_index,
            "crop": crop,
        })

    proposals.sort(key=lambda item: (-item["area_px"], item["center"][0], item["center"][1]))
    kept: list[dict] = []
    for candidate in proposals:
        cx, cy = candidate["center"]
        if any(
            np.hypot(cx - old["center"][0], cy - old["center"][1]) < 0.42 * spacing
            for old in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (item["staff_index"], item["center"][0]))
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
