"""Export an IDSR scene graph as self-contained glTF 2.0.

Coordinates preserve the scene's page-pixel frame. Z offsets are semantic display
layers, not measurements of physical ink depth. Binary geometry is embedded as a
base64 data URI so the output is a single portable .gltf document.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from typing import Any

import numpy as np


_MATERIALS = [
    {
        "name": "printed-geometry",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.16, 0.17, 0.19, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.92,
        },
        "doubleSided": True,
    },
    {
        "name": "notehead-ink",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.035, 0.04, 0.05, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.88,
        },
        "doubleSided": True,
    },
    {
        "name": "unresolved-region",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.9, 0.48, 0.08, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.9,
        },
        "doubleSided": True,
    },
    {
        "name": "lyrics-ink",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.15, 0.48, 0.85, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
        },
        "doubleSided": True,
    },
    {
        "name": "dynamics-ink",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.65, 0.18, 0.78, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
        },
        "doubleSided": True,
    },
    {
        "name": "performer-annotation-z2",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.92, 0.62, 0.12, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
        },
        "doubleSided": True,
    },
]


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _page_position(x: float, y: float, z: float, page_height: float) -> list[float]:
    # glTF is Y-up; input page coordinates are top-left-origin, Y-down.
    return [_finite(x, "x"), page_height - _finite(y, "y"), _finite(z, "z")]


def _quad(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z: float,
    page_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = [
        _page_position(x0, y0, z, page_height),
        _page_position(x1, y0, z, page_height),
        _page_position(x1, y1, z, page_height),
        _page_position(x0, y1, z, page_height),
    ]
    return np.asarray(vertices, dtype=np.float32), np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def _segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    thickness: float,
    z: float,
    page_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    half = max(0.2, float(thickness)) / 2.0
    if length <= 1e-8:
        return _quad(x0 - half, y0 - half, x0 + half, y0 + half, z, page_height)
    px, py = -dy / length * half, dx / length * half
    points = [
        (x0 + px, y0 + py),
        (x1 + px, y1 + py),
        (x1 - px, y1 - py),
        (x0 - px, y0 - py),
    ]
    vertices = np.asarray(
        [_page_position(x, y, z, page_height) for x, y in points],
        dtype=np.float32,
    )
    return vertices, np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def _ellipse(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    z: float,
    page_height: float,
    hollow: bool,
    rotation: float = 0.0,
    segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    if rx <= 0 or ry <= 0:
        raise ValueError("Notehead radii must be positive.")
    cosine, sine = math.cos(rotation), math.sin(rotation)

    def point(radius_x: float, radius_y: float, angle: float) -> list[float]:
        local_x = radius_x * math.cos(angle)
        local_y = radius_y * math.sin(angle)
        x = cx + local_x * cosine - local_y * sine
        y = cy + local_x * sine + local_y * cosine
        return _page_position(x, y, z, page_height)

    if hollow:
        inner_rx, inner_ry = rx * 0.62, ry * 0.48
        vertices: list[list[float]] = []
        indices: list[int] = []
        for index in range(segments):
            angle0 = 2.0 * math.pi * index / segments
            angle1 = 2.0 * math.pi * (index + 1) / segments
            base = len(vertices)
            vertices.extend([
                point(rx, ry, angle0),
                point(rx, ry, angle1),
                point(inner_rx, inner_ry, angle1),
                point(inner_rx, inner_ry, angle0),
            ])
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)

    vertices = [_page_position(cx, cy, z, page_height)]
    for index in range(segments):
        vertices.append(point(rx, ry, 2.0 * math.pi * index / segments))
    indices = []
    for index in range(segments):
        a = 1 + index
        b = 1 + (index + 1) % segments
        indices.extend([0, a, b])
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)



def _rectangle_outline(
    x: float,
    y: float,
    width: float,
    height: float,
    z: float,
    page_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a thin, non-occluding rectangular outline for unresolved regions."""
    if width <= 0 or height <= 0:
        raise ValueError("Unknown-region width and height must be positive.")
    thickness = min(1.4, max(0.45, min(width, height) * 0.12))
    horizontal = min(thickness, height / 2.0)
    vertical = min(thickness, width / 2.0)
    rectangles = [
        (x, y, x + width, y + horizontal),
        (x, y + height - horizontal, x + width, y + height),
        (x, y + horizontal, x + vertical, y + height - horizontal),
        (x + width - vertical, y + horizontal, x + width, y + height - horizontal),
    ]
    vertices_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    for x0, y0, x1, y1 in rectangles:
        vertices, indices = _quad(x0, y0, x1, y1, z, page_height)
        offset = sum(len(item) for item in vertices_all)
        vertices_all.append(vertices)
        indices_all.append(indices + offset)
    return np.concatenate(vertices_all), np.concatenate(indices_all)

def build_gltf(
    scene: dict,
    layer_separation: float = 6.0,
    background_png: bytes | None = None,
) -> str:
    """Serialize supported scene primitives to a self-contained glTF 2.0 file.

    Supported geometry: staff lines, barlines, noteheads, linked stems and unknown
    regions. Symbolic labels, uncertainty, capture transforms and provenance are kept
    in node extras and at scene level. Unsupported elements are counted in extras.
    """
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise ValueError("Unrecognized ScoreTwin scene schema.")
    image = scene.get("image", {})
    try:
        page_width = _finite(image["width"], "image width")
        page_height = _finite(image["height"], "image height")
    except (KeyError, TypeError) as exc:
        raise ValueError("Scene image dimensions are required.") from exc
    if page_width < 1 or page_height < 1 or page_width * page_height > 80_000_000:
        raise ValueError("Image dimensions are invalid or exceed the 80 megapixel limit.")
    layer_separation = _finite(layer_separation, "layer separation")
    if layer_separation <= 0 or layer_separation > 1000:
        raise ValueError("Layer separation must be in (0, 1000].")

    notes_by_id = {str(note.get("id")): note for note in scene.get("notes", [])}
    stem_targets: dict[str, list[str]] = {}
    outgoing_relations: dict[str, list[dict]] = {}
    incoming_relations: dict[str, list[dict]] = {}
    for relation in scene.get("relations", []):
        source_id = str(relation.get("source", ""))
        target_id = str(relation.get("target", ""))
        outgoing_relations.setdefault(source_id, []).append(dict(relation))
        incoming_relations.setdefault(target_id, []).append(dict(relation))
        if relation.get("type") == "stem_of":
            stem_targets.setdefault(source_id, []).append(target_id)

    binary = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    skipped: list[dict] = []
    materials = list(_MATERIALS)
    embedded_image: dict | None = None
    embedded_texture: dict | None = None
    background_resolution: list[int] | None = None

    if background_png is not None:
        if not isinstance(background_png, bytes) or not background_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("background_png must contain PNG-encoded analysis-raster bytes.")
        if len(background_png) > 80 * 1024 * 1024 or len(background_png) < 24:
            raise ValueError("Background texture size is invalid or exceeds 80 MiB.")
        texture_width, texture_height = struct.unpack(">II", background_png[16:24])
        if texture_width < 1 or texture_height < 1 or texture_width * texture_height > 80_000_000:
            raise ValueError("Background texture dimensions are invalid or exceed 80 megapixels.")
        background_resolution = [int(texture_width), int(texture_height)]
        embedded_image = {
            "name": "analysis-raster",
            "mimeType": "image/png",
            "uri": "data:image/png;base64," + base64.b64encode(background_png).decode("ascii"),
        }
        embedded_texture = {"sampler": 0, "source": 0}
        materials.append({
            "name": "analysis-raster-unlit",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
            "extensions": {"KHR_materials_unlit": {}},
        })

    material_by_kind = {
        "staffline": 0,
        "barline": 0,
        "stem": 0,
        "beam": 1,
        "accidental": 1,
        "augmentation_dot": 1,
        "notehead": 1,
        "unknown_region": 2,
        "lyric_syllable": 3,
        "lyric_hyphen": 3,
        "dynamic_marking": 4,
        "dynamic_glyph_mf": 4,
        "dynamic_glyph_f_or_p": 4,
        "hairpin_crescendo": 4,
        "hairpin_diminuendo": 4,
        "volta_bracket": 5,
        "investigated_trace": 5,
        "performer_directive": 5,
    }

    def _buffer_view(data: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
            "target": target,
        })
        return len(buffer_views) - 1

    def _position_accessor(vertices: np.ndarray) -> int:
        data = np.ascontiguousarray(vertices, dtype="<f4")
        view_index = _buffer_view(data.tobytes(), 34962)
        accessors.append({
            "bufferView": view_index,
            "byteOffset": 0,
            "componentType": 5126,
            "count": int(len(data)),
            "type": "VEC3",
            "min": [float(value) for value in data.min(axis=0)],
            "max": [float(value) for value in data.max(axis=0)],
        })
        return len(accessors) - 1

    def _index_accessor(indices: np.ndarray) -> int:
        data = np.ascontiguousarray(indices, dtype="<u4")
        view_index = _buffer_view(data.tobytes(), 34963)
        accessors.append({
            "bufferView": view_index,
            "byteOffset": 0,
            "componentType": 5125,
            "count": int(len(data)),
            "type": "SCALAR",
            "min": [int(data.min())],
            "max": [int(data.max())],
        })
        return len(accessors) - 1

    def _add_node(
        primitive: dict,
        vertices: np.ndarray,
        indices: np.ndarray,
        material: int,
        layer: float,
    ) -> None:
        if vertices.size == 0 or indices.size == 0:
            skipped.append({"id": primitive.get("id"), "reason": "empty_geometry"})
            return
        if not np.isfinite(vertices).all():
            raise ValueError(f"Primitive {primitive.get('id')} has non-finite coordinates.")
        if int(indices.max()) >= len(vertices) or int(indices.min()) < 0:
            raise ValueError(f"Primitive {primitive.get('id')} has invalid mesh indices.")
        pos_accessor = _position_accessor(vertices)
        index_accessor = _index_accessor(indices)
        mesh_index = len(meshes)
        meshes.append({
            "name": str(primitive.get("id", f"primitive-{mesh_index}")),
            "primitives": [{
                "attributes": {"POSITION": pos_accessor},
                "indices": index_accessor,
                "material": material,
                "mode": 4,
            }],
            "extras": {
                "idsr_kind": primitive.get("kind"),
                "coordinate_units": "source_pixels",
            },
        })
        node_index = len(nodes)
        note = notes_by_id.get(str(primitive.get("id")), {})
        nodes.append({
            "name": str(primitive.get("id", f"primitive-{node_index}")),
            "mesh": mesh_index,
            "translation": [0.0, 0.0, 0.0],
            "extras": {
                "idsr": {
                    "id": primitive.get("id"),
                    "kind": primitive.get("kind"),
                    "semantic_layer_z": layer,
                    "geometry": primitive.get("geometry", {}),
                    "appearance": primitive.get("appearance", {}),
                    "symbolic": primitive.get("symbolic", {}),
                    "provenance": primitive.get("provenance", {}),
                    "uncertainty": primitive.get("uncertainty", []),
                    "note_metadata": {
                        key: note.get(key)
                        for key in (
                            "staff_index",
                            "shape_class",
                            "pitch",
                            "voice",
                            "duration",
                            "duration_hypothesis",
                            "stem_ids",
                        )
                        if note.get(key) is not None
                    },
                    "relations": {
                        "stem_of": stem_targets.get(str(primitive.get("id")), []),
                        "outgoing": outgoing_relations.get(str(primitive.get("id")), []),
                        "incoming": incoming_relations.get(str(primitive.get("id")), []),
                    },
                }
            },
        })

    for primitive in scene.get("primitives", []):
        kind = str(primitive.get("kind", ""))
        geometry = primitive.get("geometry", {})
        primitive_id = str(primitive.get("id", ""))
        staff_index = int(primitive.get("staff_index", 0) or 0)
        note = notes_by_id.get(primitive_id, {})
        voice_value = note.get("voice", 1)
        try:
            voice_layer = max(0, int(voice_value) - 1)
        except (TypeError, ValueError):
            voice_layer = 0

        if kind == "staff_system":
            continue
        if kind == "staffline":
            x0 = float(geometry.get("x0_px", 0))
            x1 = float(geometry.get("x1_px", page_width))
            y0 = float(geometry.get("y0_px", 0))
            y1 = float(geometry.get("y1_px", y0))
            thickness = float(geometry.get("thickness_px_estimate", 1.0))
            vertices, indices = _quad(x0, y0 - thickness / 2, x1, y1 + thickness / 2, 0.0, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], 0.0)
        elif kind == "barline":
            x0 = float(geometry.get("x0_px", 0))
            y0 = float(geometry.get("y0_px", 0))
            x1 = float(geometry.get("x1_px", x0))
            y1 = float(geometry.get("y1_px", page_height))
            thickness = float(geometry.get("thickness_px_estimate", 1.0))
            vertices, indices = _segment(x0, y0, x1, y1, thickness, 0.0, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], 0.0)
        elif kind == "stem":
            targets = stem_targets.get(primitive_id, [])
            target_note = notes_by_id.get(targets[0], {}) if targets else {}
            target_voice = target_note.get("voice", 1)
            try:
                target_voice_layer = max(0, int(target_voice) - 1)
            except (TypeError, ValueError):
                target_voice_layer = 0
            z = layer_separation * (
                1.0 + max(0, int(target_note.get("staff_index", staff_index) or 0)) * 0.15
                + target_voice_layer * 0.15
            )
            x0 = float(geometry.get("x0_px", 0))
            y0 = float(geometry.get("y0_px", 0))
            x1 = float(geometry.get("x1_px", x0))
            y1 = float(geometry.get("y1_px", page_height))
            thickness = float(geometry.get("thickness_px_estimate", 1.0))
            vertices, indices = _segment(x0, y0, x1, y1, thickness, z, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind == "notehead":
            cx = float(geometry.get("center_x_px", 0))
            cy = float(geometry.get("center_y_px", 0))
            rx = float(geometry.get("radius_x_px", 0))
            ry = float(geometry.get("radius_y_px", 0))
            staff_index = int(note.get("staff_index", staff_index) or 0)
            z = layer_separation * (1.0 + max(0, staff_index) * 0.15 + voice_layer * 0.15)
            fit = note.get("geometry_refinement") or {}
            rotation_rad = float(geometry.get("rotation_rad", 0.0))
            if fit.get("accepted") and fit.get("rotation_deg") is not None:
                rotation_rad = math.radians(float(fit["rotation_deg"]))
            vertices, indices = _ellipse(
                cx, cy, rx, ry, z, page_height,
                hollow=note.get("shape_class") == "hollow_head",
                rotation=rotation_rad,
            )
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind == "beam":
            sub_beams = geometry.get("sub_beams")
            z = layer_separation * (1.25 + max(0, staff_index) * 0.15)
            if sub_beams and len(sub_beams) > 1:
                all_v = []
                all_i = []
                for sb in sub_beams:
                    sx0, sy0, sw, sh = [float(v) for v in sb]
                    v, idx = _rectangle_outline(sx0, sy0, max(2.0, sw), max(2.0, sh), z, page_height)
                    offset = len(all_v)
                    all_v.extend(v)
                    all_i.extend(idx + offset)
                _add_node(primitive, np.array(all_v, dtype=np.float32), np.array(all_i, dtype=np.uint32), material_by_kind[kind], z)
            elif geometry.get("beam_count", 1) == 2:
                x0 = float(geometry.get("x0_px", 0))
                y0 = float(geometry.get("y0_px", 0))
                x1 = float(geometry.get("x1_px", x0 + 8.0))
                y1 = float(geometry.get("y1_px", y0))
                thickness = max(1.5, float(geometry.get("thickness_px_estimate", 3.5)) * 0.45)
                sep_beam = thickness * 1.6
                v1, i1 = _segment(x0, y0 - sep_beam / 2.0, x1, y1 - sep_beam / 2.0, thickness, z, page_height)
                v2, i2 = _segment(x0, y0 + sep_beam / 2.0, x1, y1 + sep_beam / 2.0, thickness, z, page_height)
                v = np.vstack([v1, v2])
                i = np.concatenate([i1, i2 + len(v1)])
                _add_node(primitive, v, i, material_by_kind[kind], z)
            else:
                x0 = float(geometry.get("x0_px", 0))
                y0 = float(geometry.get("y0_px", 0))
                x1 = float(geometry.get("x1_px", x0 + 8.0))
                y1 = float(geometry.get("y1_px", y0))
                thickness = max(1.5, float(geometry.get("thickness_px_estimate", 3.5)))
                vertices, indices = _segment(x0, y0, x1, y1, thickness, z, page_height)
                _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind == "accidental":
            bbox = geometry.get("bbox_px", [0, 0, 6, 10])
            x, y, width, height = [float(value) for value in bbox]
            z = layer_separation * (1.15 + max(0, staff_index) * 0.15)
            vertices, indices = _rectangle_outline(
                x, y, max(2.0, width), max(2.0, height), z, page_height
            )
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind == "augmentation_dot":
            cx = float(geometry.get("center_x_px", 0))
            cy = float(geometry.get("center_y_px", 0))
            r = max(1.2, float(geometry.get("radius_px", 2.0)))
            z = layer_separation * (1.1 + max(0, staff_index) * 0.15)
            vertices, indices = _ellipse(
                cx, cy, r, r, z, page_height, hollow=False, rotation=0.0, segments=16
            )
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind == "unknown_region":
            bbox = geometry.get("bbox_px", [0, 0, 0, 0])
            x, y, width, height = [float(value) for value in bbox]
            z = layer_separation * 0.5
            vertices, indices = _rectangle_outline(x, y, width, height, z, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind in ("lyric_syllable", "lyric_hyphen"):
            bbox = geometry.get("bbox_px", [0, 0, 8, 4])
            x, y, width, height = [float(value) for value in bbox]
            z = layer_separation * 0.70
            vertices, indices = _rectangle_outline(x, y, max(2.0, width), max(2.0, height), z, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind in ("dynamic_marking", "dynamic_glyph_mf", "dynamic_glyph_f_or_p", "hairpin_crescendo", "hairpin_diminuendo"):
            bbox = geometry.get("bbox_px", [0, 0, 10, 6])
            x, y, width, height = [float(value) for value in bbox]
            z = layer_separation * 0.90
            vertices, indices = _rectangle_outline(x, y, max(2.0, width), max(2.0, height), z, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind == "volta_bracket":
            x0 = float(geometry.get("exact_x_start_cm", 0.0)) * (300.0 / 2.54)
            x1 = float(geometry.get("exact_x_end_cm", 1.0)) * (300.0 / 2.54)
            y0 = float(geometry.get("y_top_px", 100.0))
            z = layer_separation * 1.50
            vertices, indices = _segment(x0, y0, x1, y0, 2.0, z, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        elif kind in ("investigated_trace", "performer_directive"):
            bbox = geometry.get("bbox_px", [0, 0, 8, 8])
            x, y, width, height = [float(value) for value in bbox]
            z = layer_separation * 2.0  # Floating on Layer z=2
            vertices, indices = _rectangle_outline(x, y, max(2.0, width), max(2.0, height), z, page_height)
            _add_node(primitive, vertices, indices, material_by_kind[kind], z)
        else:
            skipped.append({"id": primitive_id, "reason": f"unsupported_kind:{kind}"})

    if background_png is not None:
        # Place the raster texture just behind the estimated symbolic geometry.
        # UVs remain page-aligned even if the embedded preview was downscaled.
        plane_vertices, plane_indices = _quad(0.0, 0.0, page_width, page_height, -0.1, page_height)
        position_ref = _position_accessor(plane_vertices)
        texcoords = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="<f4")
        texcoord_view = _buffer_view(texcoords.tobytes(), 34962)
        accessors.append({
            "bufferView": texcoord_view,
            "byteOffset": 0,
            "componentType": 5126,
            "count": 4,
            "type": "VEC2",
            "min": [0.0, 0.0],
            "max": [1.0, 1.0],
        })
        texcoord_ref = len(accessors) - 1
        index_ref = _index_accessor(plane_indices)
        background_mesh_index = len(meshes)
        meshes.append({
            "name": "analysis-raster-plane",
            "primitives": [{
                "attributes": {"POSITION": position_ref, "TEXCOORD_0": texcoord_ref},
                "indices": index_ref,
                "material": len(materials) - 1,
                "mode": 4,
            }],
            "extras": {
                "idsr_kind": "analysis_raster_texture",
                "coordinate_units": "source_pixels",
            },
        })
        nodes.append({
            "name": "analysis-raster-plane",
            "mesh": background_mesh_index,
            "translation": [0.0, 0.0, 0.0],
            "extras": {
                "idsr": {
                    "kind": "analysis_raster_texture",
                    "semantic_layer_z": -0.1,
                    "texture_resolution_px": background_resolution,
                    "identity_note": "This is a PNG encoding of the decoded analysis raster. The original input file is preserved separately byte-for-byte.",
                }
            },
        })

    while len(binary) % 4:
        binary.append(0)
    # glTF buffers require a positive byteLength even when the scene has no mesh data.
    if not binary:
        binary.extend(b"\x00\x00\x00\x00")
    gltf = {
        "asset": {"version": "2.0", "generator": "ScoreTwin IDSR glTF exporter"},
        "scene": 0,
        "scenes": [{"name": "ScoreTwin IDSR scene", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [{
            "byteLength": len(binary),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode("ascii"),
        }],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extras": {
            "schema": "scoretwin-gltf-scene-0.1",
            "background_texture_embedded": background_png is not None,
            "background_texture_resolution_px": background_resolution,
            "background_texture_resampled_to_analysis_frame": (
                background_resolution != [int(page_width), int(page_height)]
                if background_resolution is not None else False
            ),
            "source_schema": scene.get("schema"),
            "image": {"width_px": page_width, "height_px": page_height},
            "coordinate_frame": scene.get("coordinate_frame", "source_raster_pixel_centers"),
            "coordinate_units": "one glTF unit equals one page pixel in X/Y",
            "axis_mapping": "X right, Y up (page-height minus source y), Z semantic layer",
            "layer_separation": layer_separation,
            "depth_semantics": "Z offsets are visualization layers, not measured physical ink depth",
            "source": scene.get("source", {}),
            "capture_diagnostics": scene.get("capture_diagnostics", {}),
            "identity_reconstruction": scene.get("identity_reconstruction", {}),
            "unsupported_primitive_count": len(skipped),
            "unsupported_primitives": skipped[:100],
            "primitive_count": len(nodes) - (1 if background_png is not None else 0),
            "node_count_including_background_plane": len(nodes),
        },
    }
    if background_png is not None:
        gltf["images"] = [embedded_image]
        gltf["textures"] = [embedded_texture]
        gltf["samplers"] = [{
            "magFilter": 9729,
            "minFilter": 9729,
            "wrapS": 33071,
            "wrapT": 33071,
        }]
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    return json.dumps(gltf, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def validate_embedded_gltf(gltf_text: str) -> dict:
    """Perform dependency-free structural checks on the embedded-buffer glTF."""
    gltf = json.loads(gltf_text)
    if gltf.get("asset", {}).get("version") != "2.0":
        raise ValueError("glTF asset version must be 2.0.")
    buffers = gltf.get("buffers", [])
    if len(buffers) != 1:
        raise ValueError("Expected one self-contained glTF buffer.")
    uri = buffers[0].get("uri", "")
    prefix = "data:application/octet-stream;base64,"
    if not uri.startswith(prefix):
        raise ValueError("glTF buffer must be an embedded base64 data URI.")
    raw = base64.b64decode(uri[len(prefix):], validate=True)
    if len(raw) != int(buffers[0].get("byteLength", -1)):
        raise ValueError("Embedded buffer byteLength does not match the decoded data.")
    images = gltf.get("images", [])
    for image_index, image in enumerate(images):
        image_uri = image.get("uri", "")
        image_prefix = "data:image/png;base64,"
        if not image_uri.startswith(image_prefix):
            raise ValueError(f"Image {image_index} is not embedded as a PNG data URI.")
        image_bytes = base64.b64decode(image_uri[len(image_prefix):], validate=True)
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Image {image_index} has an invalid PNG signature.")
    textures = gltf.get("textures", [])
    for texture_index, texture in enumerate(textures):
        if not 0 <= int(texture.get("source", -1)) < len(images):
            raise ValueError(f"Texture {texture_index} refers to a missing image.")
        sampler = texture.get("sampler")
        if sampler is not None and not 0 <= int(sampler) < len(gltf.get("samplers", [])):
            raise ValueError(f"Texture {texture_index} refers to a missing sampler.")
    for material_index, material in enumerate(gltf.get("materials", [])):
        texture_info = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if texture_info is not None and not 0 <= int(texture_info.get("index", -1)) < len(textures):
            raise ValueError(f"Material {material_index} refers to a missing base-color texture.")
    views = gltf.get("bufferViews", [])
    for index, view in enumerate(views):
        offset = int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
        if offset < 0 or length < 0 or offset + length > len(raw) or offset % 4:
            raise ValueError(f"bufferView {index} points outside the embedded buffer or is misaligned.")
    accessors = gltf.get("accessors", [])
    for index, accessor in enumerate(accessors):
        view_index = int(accessor["bufferView"])
        if view_index < 0 or view_index >= len(views):
            raise ValueError(f"Accessor {index} refers to a missing bufferView.")
        view = views[view_index]
        component_type = int(accessor["componentType"])
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(accessor["type"])
        component_size = {5121: 1, 5123: 2, 5125: 4, 5126: 4}.get(component_type)
        if components is None or component_size is None:
            raise ValueError(f"Accessor {index} uses an unsupported component type or shape.")
        byte_offset = int(accessor.get("byteOffset", 0))
        byte_length = int(accessor["count"]) * components * component_size
        if byte_offset < 0 or byte_offset + byte_length > int(view["byteLength"]):
            raise ValueError(f"Accessor {index} reads outside its bufferView.")
    nodes = gltf.get("nodes", [])
    for node_index, node in enumerate(nodes):
        mesh_ref = node.get("mesh")
        if mesh_ref is not None and not 0 <= int(mesh_ref) < len(gltf.get("meshes", [])):
            raise ValueError(f"Node {node_index} refers to a missing mesh.")
    for scene_index, entry in enumerate(gltf.get("scenes", [])):
        for node_ref in entry.get("nodes", []):
            if not 0 <= int(node_ref) < len(nodes):
                raise ValueError(f"Scene {scene_index} refers to a missing node.")
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        for primitive in mesh.get("primitives", []):
            position_ref = int(primitive["attributes"]["POSITION"])
            index_ref = int(primitive["indices"])
            material_ref = int(primitive["material"])
            if not 0 <= position_ref < len(accessors):
                raise ValueError(f"Mesh {mesh_index} has an invalid POSITION accessor.")
            if not 0 <= index_ref < len(accessors):
                raise ValueError(f"Mesh {mesh_index} has an invalid index accessor.")
            if not 0 <= material_ref < len(gltf.get("materials", [])):
                raise ValueError(f"Mesh {mesh_index} has an invalid material.")
            attribute_accessors = {}
            for attribute_name, accessor_ref in primitive.get("attributes", {}).items():
                if not 0 <= int(accessor_ref) < len(accessors):
                    raise ValueError(f"Mesh {mesh_index} has an invalid {attribute_name} accessor.")
                attribute_accessors[attribute_name] = accessors[int(accessor_ref)]
            position_accessor = attribute_accessors["POSITION"]
            index_accessor = accessors[int(primitive["indices"])]
            if position_accessor.get("type") != "VEC3" or index_accessor.get("type") != "SCALAR":
                raise ValueError(f"Mesh {mesh_index} has incorrect accessor shapes.")
            if "TEXCOORD_0" in attribute_accessors:
                uv_accessor = attribute_accessors["TEXCOORD_0"]
                if uv_accessor.get("type") != "VEC2" or int(uv_accessor["count"]) != int(position_accessor["count"]):
                    raise ValueError(f"Mesh {mesh_index} has incorrect TEXCOORD_0 data.")
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError(f"Mesh {mesh_index} is not a triangle mesh.")
            index_count = int(index_accessor["count"])
            if index_count % 3:
                raise ValueError(f"Mesh {mesh_index} has a non-triangular index count.")
            view = views[int(index_accessor["bufferView"])]
            global_offset = int(view.get("byteOffset", 0)) + int(index_accessor.get("byteOffset", 0))
            indices = struct.unpack(
                "<" + "I" * index_count,
                raw[global_offset:global_offset + index_count * 4],
            )
            vertex_count = int(position_accessor["count"])
            if indices and max(indices) >= vertex_count:
                raise ValueError(f"Mesh {mesh_index} contains an index outside its vertex range.")
    return {
        "valid": True,
        "nodes": len(gltf.get("nodes", [])),
        "meshes": len(gltf.get("meshes", [])),
        "buffer_bytes": len(raw),
        "has_background_texture": bool(images),
        "background_texture_resolution_px": gltf.get("extras", {}).get("background_texture_resolution_px"),
        "coordinate_units": gltf.get("extras", {}).get("coordinate_units"),
        "depth_semantics": gltf.get("extras", {}).get("depth_semantics"),
    }
