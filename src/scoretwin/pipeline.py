"""IDSR first vertical slice: analyze, preserve, visualize and export."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .imaging import detect_barlines, detect_staff_systems, extract_notehead_candidates, load_grayscale, staff_step_and_pitch
from .musicxml import build_musicxml
from .proposal import get_default_model, load_model, predict_patch


def _classify_candidates(candidates: list[dict], model: dict) -> list[dict]:
    classified = []
    for candidate in candidates:
        scores = predict_patch(model, candidate.pop("crop"))
        label = max(scores, key=scores.get)
        confidence = float(scores[label])
        if confidence < 0.48:
            label = "unknown"
        item = {**candidate, "class": label, "confidence": round(confidence, 4),
                "class_probabilities": {key: round(float(value), 4) for key, value in scores.items()}}
        classified.append(item)
    return classified


def analyze_array(gray: np.ndarray, model: dict | None = None) -> dict:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("analyze_array expects a uint8 grayscale image.")
    if min(gray.shape) < 32:
        raise ValueError("Input image is too small to contain a score page.")
    started = time.perf_counter()
    staffs, binary, staff_mask = detect_staff_systems(gray)
    candidates = extract_notehead_candidates(gray, binary, staff_mask, staffs)
    classified = _classify_candidates(candidates, model or get_default_model())

    staff_objects = []
    for staff in staffs:
        barlines = detect_barlines(binary, staff)
        staff_objects.append({**staff.to_dict(), "barlines": barlines})

    notes = []
    unknown_objects = []
    for item in classified:
        staff_index = item["staff_index"]
        staff = staffs[staff_index]
        cx, cy = item["center"]
        if item["class"] in ("filled_head", "hollow_head"):
            step, letter, octave = staff_step_and_pitch(cy, staff)
            notes.append({
                "id": item["id"],
                "staff_index": staff_index,
                "bbox": item["bbox"],
                "center": item["center"],
                "pitch": {"step": letter, "octave": octave},
                "diatonic_step_from_bottom_line": step,
                "pitch_confidence": item["confidence"],
                "duration": None,
                "uncertainty": ["key_signature", "accidental", "duration", "voice", "instrument_or_part", "measure_assignment"],
            })
        else:
            unknown_objects.append({
                "id": item["id"],
                "staff_index": staff_index,
                "bbox": item["bbox"],
                "center": item["center"],
                "class": item["class"],
                "confidence": item["confidence"],
                "reason": "Candidate was not confidently classified as a notehead.",
            })

    warnings = []
    if not staffs:
        warnings.append("No five-line staff group detected. Preserve the source and review the page manually.")
    if not notes:
        warnings.append("No noteheads passed the current proposal classifier; the MusicXML will contain no inferred notes.")
    warnings.extend([
        "Pitch proposals assume treble clef and a C-major/no-accidental interpretation; key signatures and accidentals are not yet inferred.",
        "Durations, voice assignments, instrument mapping and full score structure remain unresolved and are marked for review in MusicXML.",
        "The synthetic model's confidence is not calibrated confidence on real manuscripts or photographs.",
    ])
    return {
        "schema": "scoretwin-scene-0.1",
        "image": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "staffs": staff_objects,
        "notes": notes,
        "unknown_objects": unknown_objects,
        "proposal_count": len(classified),
        "note_count": len(notes),
        "warnings": warnings,
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def render_overlay(gray: np.ndarray, scene: dict) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for staff in scene.get("staffs", []):
        for y in staff["lines"]:
            cv2.line(overlay, (0, int(y)), (overlay.shape[1] - 1, int(y)), (50, 170, 50), 1, cv2.LINE_AA)
        for x in staff.get("barlines", []):
            cv2.line(overlay, (int(x), int(staff["top"])), (int(x), int(staff["bottom"])), (170, 120, 20), 1, cv2.LINE_AA)

    for note in scene.get("notes", []):
        x, y, width, height = note["bbox"]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (30, 60, 230), 1, cv2.LINE_AA)
        cv2.putText(overlay, f'{note["pitch"]["step"]}{note["pitch"]["octave"]}?',
                    (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 60, 230), 1, cv2.LINE_AA)
    for item in scene.get("unknown_objects", []):
        x, y, width, height = item["bbox"]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 150, 210), 1, cv2.LINE_AA)
    return overlay


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    page: int = 0,
    model_path: str | Path | None = None,
) -> dict:
    """Generate a self-contained analysis bundle and preserve input bytes unchanged."""
    source = Path(input_path)
    output = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    gray = load_grayscale(source, page=page)
    model = load_model(model_path) if model_path else get_default_model()
    scene = analyze_array(gray, model=model)
    scene["source"] = {
        "filename": source.name,
        "sha256": source_hash,
        "bytes": len(source_bytes),
        "page_index": page if source.suffix.lower() == ".pdf" else None,
        "preserved_losslessly": True,
    }

    suffix = source.suffix.lower() or ".bin"
    preserved_name = f"source-original{suffix}"
    (output / preserved_name).write_bytes(source_bytes)
    scene["source"]["preserved_file"] = preserved_name

    overlay = render_overlay(gray, scene)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode diagnostic overlay.")
    (output / "overlay.png").write_bytes(encoded.tobytes())

    (output / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    musicxml = build_musicxml(scene)
    (output / "provisional.musicxml").write_text(musicxml, encoding="utf-8")

    # Verify the immutable source copy before reporting completion.
    copied_hash = hashlib.sha256((output / preserved_name).read_bytes()).hexdigest()
    if copied_hash != source_hash:
        raise IOError("Preserved source hash does not match the uploaded input.")
    return {
        "output_dir": str(output),
        "scene": scene,
        "musicxml_path": str(output / "provisional.musicxml"),
        "overlay_path": str(output / "overlay.png"),
        "source_hash_verified": True,
        "source_sha256": source_hash,
    }
