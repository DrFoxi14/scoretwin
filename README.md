# ScoreTwin / IDSR

ScoreTwin is the active implementation workspace for Identity-preserving Digital Score Reconstruction (IDSR).

## Current vertical slice

This repository contains a runnable research prototype:

- image and PDF ingestion;
- five-line staff detection with a global-skew warning (rotation/perspective rectification is not yet applied);
- notehead proposals that attempt to reconstruct fragments cut by staff lines, followed by a bounded local template-fit refinement;
- a small synthetic patch classifier, trainable locally;
- a provenance-aware geometry graph (staff systems, staff lines, barlines, notehead proposals, unknown regions and typed relations) with pixel coordinates and uncertainty flags;
- a source-aligned 2D editor for dragging notehead proposals, a separate structural 2D render of known primitives, and an interactive semantic-layer view in 3D;
- editable pitch proposals, downloadable scene JSON and provisional MusicXML export, with uncertain durations and score structure marked for review;
- MusicXML serialization for explicitly annotated chord groups, voices, accidentals and repeated systems; those fields remain unrecognized from pixels;
- a local browser UI, CLI, synthetic-page generator and synthetic regression benchmark;
- tests and a GitHub Actions workflow.

This is not yet a validated general-purpose OMR engine. The current proposal model is trained only on synthetic crops. Stem detection is a morphology-plus-proximity heuristic; duration hypotheses from filled/hollow heads and stem presence remain explicitly uncalibrated and are not used to silently change exported MusicXML durations. Beams, flags and rhythm are not yet decoded. Pitch proposals assume treble clef and a C-major/no-accidental interpretation. Duration, voice, accidentals, staff-to-part mapping and ambiguous overlaps are not reliably inferred. The MusicXML exporter marks these assumptions rather than claiming a correct transcription. Stacked noteheads with nearly aligned x-coordinates can be emitted as provisional chord hypotheses, and those relations are marked for review.

## Run locally

Python 3.11 or newer is required.

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[test]"
    pytest

Create and analyze a synthetic example (a useful first smoke test):

    scoretwin demo --out ./scoretwin-demo

Evaluate notehead proposals on generated pages, with exact synthetic notehead-center ground truth:

    scoretwin evaluate-synthetic --pages 100 --seed 20261029 --noise-sigma 2 --blur-sigma 0.35 --contract-staff-spaces 0.3 --out ./results/synthetic-capture-pilot-2026-10-29.json

Compare a clean synthetic source page, its projective camera simulation, and correction using exact page corners:

    scoretwin evaluate-perspective --pages 100 --seed 20261029 --noise-sigma 2 --blur-sigma 0.35 --contract-staff-spaces 0.3 --out ./results/perspective-capture-pilot-2026-10-29.json

Evaluate a separately annotated manifest (the command does not manufacture ground truth):

    scoretwin evaluate-manifest --manifest ./data/manifest.json --out ./results/real-evaluation.json

Train the small proposal model on generated synthetic crops:

    scoretwin train-model --out ./models/proposal.npz --samples-per-class 600 --epochs 50

Analyze one image:

    scoretwin analyze ./score.png --out ./runs/score

Analyze a PDF page (zero-indexed page number):

    scoretwin analyze ./score.pdf --page 0 --out ./runs/score

Create an optional global-rotation preview with a reversible transform back to source coordinates:

    scoretwin rectify-preview ./score.jpg --out ./runs/rectification

The bundle stores a byte-identical source copy, a resampled preview, and both coordinate transforms. This is a capture diagnostic; the default OMR analysis still runs on the source raster. It handles one global rotation only, not perspective, page curl, or local deformation.

For a perspective photo, manually select the page corners in top-left, top-right, bottom-right, bottom-left order, then pass those source-raster pixel coordinates as JSON:

    scoretwin rectify-preview ./score.jpg --corners-json '[[42,31],[1420,77],[1375,1980],[18,1902]]' --out ./runs/page-rectification
    scoretwin analyze ./score.jpg --corners-json '[[42,31],[1420,77],[1375,1980],[18,1902]]' --out ./runs/score-rectified

The transform maps those four points to a rectangle and records the forward and inverse homographies. It assumes the page is planar and the corners are correct; it does not automatically detect the page or flatten curvature, folds, or local warping. The source file remains byte-identical in the analysis bundle.

To opt in to analyzing the rectified raster while retaining source coordinates in the scene metadata:

    scoretwin analyze ./score.jpg --rectify-skew --out ./runs/score-rectified

The overlay and structural render use the analysis raster; `source-original.*` remains byte-identical, and `scene.json` plus `capture-transform.json` include mappings back to rasterized source coordinates.

Run the browser app:

    scoretwin serve --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000. Uploaded files are processed by the local server process. No remote inference API is used by this prototype. On an image, **Select four page corners** enables a clockwise tap sequence from top-left through bottom-left; analysis applies the corresponding planar homography and records its inverse. This is optional and does not flatten a curled page. After edits, use **Download edited scene JSON** to keep the adjusted geometry and pitch proposals; **Regenerate MusicXML from edits** exports the current symbolic projection.

## Output bundle

- source file copied byte-for-byte to the bundle and verified by SHA-256;
- scene.json: staff geometry, a primitive graph, typed relations, continuous pitch-position offsets, source provenance, software versions, model-training metadata and warnings;
- overlay.png: source image with staff/notehead proposals;
- geometry-render.png: structural render of the currently modeled primitives (not a complete/pixel-identical page render);
- provisional.musicxml: XML output carrying explicit review markers.

Unresolved regions include a heuristic reason such as low local contrast, low local sharpness/blur, or an unfamiliar/uncertain mark. These thresholds are not calibrated; the UI explicitly treats them as diagnostic hints, not probabilities.

The preserved source is the lossless identity anchor. The overlay is a diagnostic layer, not a claim that all original ink has been parsed into primitives. The 3D view uses a semantic layer offset; it is not a measurement of the physical relief of ink.

For hand-annotated scene JSON, MusicXML export accepts note fields `chord_id`, `voice`, `onset`, `duration`, `type`, `accidental`, and `pitch.alter`. Staffs can share `part_id` and specify `part_name`, `system_index`, `staff_number`, `key_fifths`, `time_signature`, and `clef`; notes can override `system_index` or set `measure_index`. `onset` and `duration` use MusicXML divisions (the current default is one division per quarter note). The exporter serializes supplied annotations, adds review-marked geometric chord hypotheses, and keeps unresolved fields flagged; it does not infer voices or accidentals.

## Epistemic status

- [DEMONSTRATED BY CONSTRUCTION] The original uploaded file is copied byte-for-byte and hash-checked in CLI bundles.
- [MEASURED BY TESTS] Staff detection, staff-line-crossing notehead proposal, synthetic classifier execution, local template-fit refinement, scene-graph creation, MusicXML XML parsing and bundle generation are checked by tests.
- [HYPOTHESIS] The synthetic patch classifier and geometric proposals will transfer to real score photographs. This must be tested on a held-out, real-page benchmark.
- [NOT YET IMPLEMENTED] End-to-end recovery of complete symbolic-geometric scenes from arbitrary photographs, calibrated symbolic ambiguity, physically measured ink depth and validated DID/MDL metrics.

## Synthetic capture pilot

[MĂSURAT — synthetic only] The checked-in run evaluated 100 procedurally generated pages (1,800 ground-truth noteheads), seeds 20261029–20261128, two staffs per page, 9 notes per staff, grayscale noise σ=2 and Gaussian blur σ=0.35 px. Geometry remained registered; there was no perspective, rotation, color ISP, page curvature or real manuscript. The patch classifier used 360 synthetic training crops and 90 synthetic holdout crops.

[MĂSURAT — synthetic only] After local template-fit refinement, matching at 0.3 staff spaces produced 1,790 true positives, 2 false positives and 10 false negatives: micro precision 0.9989, recall 0.9944 and F1 0.9967. Macro page F1 was 0.9966 (page-bootstrap 95% CI 0.9939–0.9988). Mean page p95 localization error was 0.0778 staff spaces (95% CI 0.0735–0.0820); all 100 pages met the ≤0.3-space localization threshold (Wilson 95% CI 96.3%–100%). The combined gate F1 ≥0.95 and p95 ≤0.3 spaces passed on 96/100 pages (Wilson 95% CI 90.2%–98.4%). The p95 is conditional on matched noteheads; precision/recall separately account for misses and extras. The patch classifier scored 0.9556 on 90 synthetic holdout crops; that is crop classification accuracy, not page-level note detection. See [the full report](results/synthetic-capture-pilot-2026-10-29.json) for per-page data, model hash, versions and hardware.

[MĂSURAT] This is a synthetic-only result, not a real-photo benchmark or the IDSR DID metric. The image pipeline does not infer voices, accidentals, durations or staff-to-part mapping.

[IPOTEZĂ] The synthetic proposal and refinement may transfer to registered score photographs. A falsifying test is a held-out set of real photographs with independently prepared notehead-center annotations and per-page precision, recall, F1 and staff-space localization confidence intervals.

[MĂSURAT] The run used Python 3.9.6, NumPy 1.22.0 and OpenCV 5.0.0. The project requires Python ≥3.11, NumPy ≥1.26 and OpenCV ≥4.9,<5; the JSON records that this environment does not meet those constraints. Treat the result and latency as exploratory until rerun on supported dependencies and confirmed by CI.

## Registered perspective-capture pilot

[MĂSURAT — paired synthetic only] The checked-in run used 100 synthetic source pages with two staffs and 18 noteheads each, seeds 20261029–20261128, grayscale noise σ=2 and blur σ=0.35 px. Each page was warped by a known planar homography. The corrected view was given the exact four generated page corners; this excludes automatic corner-detection error. The JSON stores per-page raster hashes, corners, and both homographies. The run is not a real-camera or real-manuscript test, and its score is not IDSR DID.

[MĂSURAT — paired synthetic only] The source-reference view detected 2 staffs on all 100 pages and scored micro F1 0.9967 (macro page F1 0.9966, page-bootstrap 95% CI 0.9939–0.9986). The uncorrected projective view detected 0 staffs and proposed 0 noteheads on all 100 pages (0/100; Wilson 95% upper bound 3.7%). With exact-corner correction, 2 staffs were detected on all pages; notehead micro F1 was 0.8039, with recall 0.7378 and precision 0.8830. Macro page F1 was 0.7968 (95% CI 0.7710–0.8216). Mean page p95 center error was 0.0993 staff spaces (95% CI 0.0953–0.1034), conditional on matched noteheads; every page met that conditional ≤0.3-space threshold. False negatives and false positives remain separate: 472 missed and 176 extra proposals across 100 pages.

[IPOTEZĂ] The uncorrected failure is caused by the staff detector's near-horizontal morphology assumption: projective line slope prevents its horizontal opening kernel from retaining staff lines. Falsify this with a sweep over known page tilts and keystone levels, 100 seeds per level, and report staff-detection rate before and after correction. The checked-in per-page report records the 0/100 staff detections but does not isolate the causal image-processing operation.

[MĂSURAT] This pilot used Python 3.9.6, NumPy 1.22.0 and OpenCV 5.0.0, outside declared project constraints. The result shows what exact manual corners can recover in this simulator; it gives no evidence about incorrect user taps, a curled page, phone ISP, actual photographs, or symbolic transcription quality.

## Next gates

1. Keep the synthetic regression benchmark deterministic and report precision/recall rather than only file creation.
2. Measure notehead precision/recall on registered real photographs.
3. Add stem, beam, accidental and duration hypotheses with explicit user correction.
4. Train or retain a proposal model only if it beats a simpler baseline on real pages.
5. Add DID and MDL only after testing metrics against adversarial manipulations and human judgments.


## External annotated-page manifest

To evaluate real photographs, create a JSON file with schema `scoretwin-annotated-pages-v1`. Every image entry needs an image path relative to the manifest and a `notes` array whose items include `center: [x, y]` in the rasterized image coordinate frame and `staff_index` numbered from top to bottom. A `pitch` object can optionally contain `step` and `octave`; `staff_spacing_px` is strongly recommended. An optional SHA-256 is checked against the original file.

Example:

```json
{
  "schema": "scoretwin-annotated-pages-v1",
  "pages": [{
    "id": "page-001",
    "image": "images/page-001.jpg",
    "staff_spacing_px": 17.0,
    "notes": [
      {"center": [245.2, 319.7], "staff_index": 0, "pitch": {"step": "G", "octave": 4}}
    ]
  }]
}
```

The report includes per-page counts, precision/recall/F1, center error in pixels and staff spaces, pitch accuracy when labelled, and page-bootstrap 95% intervals for detection and normalized geometry error. These intervals describe variation across the supplied pages; they do not repair bias from a small or unrepresentative sample. Do not call this a real-photo benchmark unless the manifest actually references real photos with independently prepared annotations.
