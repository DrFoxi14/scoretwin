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
- a local browser UI, CLI, synthetic-page generator and synthetic regression benchmark;
- tests and a GitHub Actions workflow.

This is not yet a validated general-purpose OMR engine. The current proposal model is trained only on synthetic crops. Stem detection is a morphology-plus-proximity heuristic; it does not yet decode beams, flags or rhythm. Pitch proposals assume treble clef and a C-major/no-accidental interpretation. Duration, voice, accidentals, staff-to-part mapping and ambiguous overlaps are not reliably inferred. The MusicXML exporter marks these assumptions rather than claiming a correct transcription. Stacked noteheads with nearly aligned x-coordinates can be emitted as provisional chord hypotheses, and those relations are marked for review.

## Run locally

Python 3.11 or newer is required.

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[test]"
    pytest

Create and analyze a synthetic example (a useful first smoke test):

    scoretwin demo --out ./scoretwin-demo

Evaluate notehead proposals on generated pages, with exact synthetic notehead-center ground truth:

    scoretwin evaluate-synthetic --pages 16 --seed 20261029 --out ./results/synthetic-evaluation.json

Evaluate a separately annotated manifest (the command does not manufacture ground truth):

    scoretwin evaluate-manifest --manifest ./data/manifest.json --out ./results/real-evaluation.json

Train the small proposal model on generated synthetic crops:

    scoretwin train-model --out ./models/proposal.npz --samples-per-class 600 --epochs 50

Analyze one image:

    scoretwin analyze ./score.png --out ./runs/score

Analyze a PDF page (zero-indexed page number):

    scoretwin analyze ./score.pdf --page 0 --out ./runs/score

Run the browser app:

    scoretwin serve --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000. Uploaded files are processed by the local server process. No remote inference API is used by this prototype. After edits, use **Download edited scene JSON** to keep the adjusted geometry and pitch proposals; **Regenerate MusicXML from edits** exports the current symbolic projection.

## Output bundle

- source file copied byte-for-byte to the bundle and verified by SHA-256;
- scene.json: staff geometry, a primitive graph, typed relations, continuous pitch-position offsets, source provenance, software versions, model-training metadata and warnings;
- overlay.png: source image with staff/notehead proposals;
- geometry-render.png: structural render of the currently modeled primitives (not a complete/pixel-identical page render);
- provisional.musicxml: XML output carrying explicit review markers.

Unresolved regions include a heuristic reason such as low local contrast, low local sharpness/blur, or an unfamiliar/uncertain mark. These thresholds are not calibrated; the UI explicitly treats them as diagnostic hints, not probabilities.

The preserved source is the lossless identity anchor. The overlay is a diagnostic layer, not a claim that all original ink has been parsed into primitives. The 3D view uses a semantic layer offset; it is not a measurement of the physical relief of ink.

## Epistemic status

- [DEMONSTRATED BY CONSTRUCTION] The original uploaded file is copied byte-for-byte and hash-checked in CLI bundles.
- [MEASURED BY TESTS] Staff detection, staff-line-crossing notehead proposal, synthetic classifier execution, local template-fit refinement, scene-graph creation, MusicXML XML parsing and bundle generation are checked by tests.
- [HYPOTHESIS] The synthetic patch classifier and geometric proposals will transfer to real score photographs. This must be tested on a held-out, real-page benchmark.
- [NOT YET IMPLEMENTED] End-to-end recovery of complete symbolic-geometric scenes from arbitrary photographs, calibrated symbolic ambiguity, physically measured ink depth and validated DID/MDL metrics.

## Next gates

1. Keep the synthetic regression benchmark deterministic and report precision/recall rather than only file creation.
2. Measure notehead precision/recall on registered real photographs.
3. Add stem, beam, accidental and duration hypotheses with explicit user correction.
4. Train or retain a proposal model only if it beats a simpler baseline on real pages.
5. Add DID and MDL only after testing metrics against adversarial manipulations and human judgments.


## External annotated-page manifest

To evaluate real photographs, create a JSON file with schema \`scoretwin-annotated-pages-v1\`. Every image entry needs an image path relative to the manifest and a \`notes\` array whose items include \`center: [x, y]\` in the rasterized image coordinate frame and \`staff_index\` numbered from top to bottom. A \`pitch\` object can optionally contain \`step\` and \`octave\`; \`staff_spacing_px\` is strongly recommended. An optional SHA-256 is checked against the original file.

Example:

\`\`\`json
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
\`\`\`

The report includes per-page counts, precision/recall/F1, center error in pixels and staff spaces, pitch accuracy when labelled, and page-bootstrap 95% intervals for detection and normalized geometry error. These intervals describe variation across the supplied pages; they do not repair bias from a small or unrepresentative sample. Do not call this a real-photo benchmark unless the manifest actually references real photos with independently prepared annotations.
