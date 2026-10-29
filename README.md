# ScoreTwin / IDSR

ScoreTwin is the active implementation workspace for Identity-preserving Digital Score Reconstruction (IDSR).

## Current vertical slice

This repository now contains a runnable research prototype rather than a research plan alone:

- image and PDF ingestion;
- staff-line detection and preliminary notehead proposals;
- a small synthetic patch classifier that can be trained locally;
- a geometry-first scene JSON with source hash and uncertainty flags;
- a visual overlay for inspecting proposals;
- provisional MusicXML export, with uncertain durations and notation explicitly marked for review;
- a local browser UI and command line;
- tests and CI.

The critical limitation is intentional: this is not yet a validated general-purpose OMR engine. A notehead position can be proposed from the staff geometry; duration, voice, accidentals, instrument assignment and ambiguous overlaps are not reliably inferred in this first vertical slice. The exporter marks its assumptions rather than claiming a correct transcription.

## Run locally

Python 3.11 or newer is required.

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[test]"
    pytest

Analyze one image:

    scoretwin analyze ./score.png --out ./runs/score

Analyze a PDF page (zero-indexed page number):

    scoretwin analyze ./score.pdf --page 0 --out ./runs/score

Train the small proposal model on generated synthetic crops:

    scoretwin train-model --out ./models/proposal.npz --samples-per-class 600 --epochs 50

Run the browser app:

    scoretwin serve --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000. Uploaded files are processed by the local server process. No remote inference API is used by this prototype.

## Output bundle

- source file copied byte-for-byte to the bundle;
- scene.json: staff systems, proposals, pitch-position estimates, provenance and warnings;
- overlay.png: diagnostic visualization;
- provisional.musicxml: schema-shaped MusicXML output carrying explicit review markers.

The output preserves the original file as the lossless identity anchor. The overlay is a diagnostic layer, not a claim that all original ink has been parsed into primitives.

## Epistemic status

- [DEMONSTRATED BY CONSTRUCTION] The original uploaded file can be copied and SHA-256 verified byte-for-byte.
- [MEASURED BY TESTS] Staff detection, synthetic classifier execution, MusicXML XML parsing and bundle generation are checked by the test suite and CI.
- [HYPOTHESIS] The synthetic patch classifier will improve proposal ranking on real print; this must be checked on a held-out, real-page benchmark.
- [NOT YET IMPLEMENTED] A trained end-to-end detector that discovers complete symbolic-geometric scenes from arbitrary score photographs, ambiguity-aware duration/voice inference, true physical ink depth and a calibrated DID benchmark.

## Next gates

1. Make the current pipeline reproducible in CI.
2. Measure notehead precision/recall on registered real photographs.
3. Add barline, stem, beam, accidental and duration hypotheses with human correction hooks.
4. Train a proposal model only if its real-page improvement beats the heuristic baseline.
5. Add full DID and MDL only after the metrics are tested against adversarial manipulations and human judgments.
