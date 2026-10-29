# IDSR human calibration protocol (draft 0.1)

Status: **[IPOTEZĂ]** This is a proposed experiment, not a human study already run by the repository.

## 1. Target question

Does a component metric predict a person's judgment that the model's *interpreted symbols and geometry* match the source? Do not let the lossless residual masquerade as recognition: recomposing the structural render with residual reproduces the decoded raster even when the structural model misses symbols.

Run separate manifests for separate questions:

- **Visual structure:** show the original source next to geometry-render.png. Do not show reconstructed-analysis.png or residual; they would copy missing content back into the image.
- **Symbolic transcription:** compare candidate MusicXML against a source-verified transcription or an expert-reviewed reference. The source image alone is not a verified transcription when notation is ambiguous.
- **Ambiguity handling:** judge whether unclear/unfamiliar marks are flagged instead of confidently invented.

The current evaluator accepts one ordinal rating per pair. Run separate study manifests for each rating dimension.

## 2. Proposed sample and power

**[IPOTEZĂ — planning heuristic]** Start with at least **5 independent music readers/engravers** and **60 unique comparison pairs** drawn from distinct source pieces. The included planner estimates about 47 independent pairs for detecting a rank association near |rho| = 0.4, two-sided alpha = 0.05, power 80%, using a Fisher-z approximation. Sixty pairs allow an exclusion margin. This is not a guarantee for Spearman correlation or clustered data; simulate power after a pilot estimates ties, missing ratings, and within-piece dependence.

Balance easy and difficult inputs: clear printed music, small missing/added marks, chord and voice ambiguity, dense multi-staff pages, low contrast, perspective, unfamiliar notation, and out-of-distribution handwriting. Source pieces—not augmented views of one page—are the independent groups.

Each rater sees about 60 unique pairs plus 10 hidden repeats (about 70 judgments per dimension). Do not select only cases the system already handles.

## 3. Blind presentation

Randomize pair order per rater. Remove filenames, system names, confidence values, and ground truth from the UI. Randomize left/right placement when comparing systems. Insert hidden repeats under different IDs while retaining the sealed repeat_group label. A rater should not see a repeated item twice in a short sequence.

For visual structure, ask: **"How closely does the candidate preserve visible notation and geometry?"** Anchor 1 = clearly different/incomplete, 4 = partly matching with obvious omissions, 7 = visually indistinguishable for the visible marks represented. Permit "cannot judge"; apply a pre-registered exclusion rule rather than converting it to a low score.

For symbolic correctness, score pitch/rhythm/voice separately from whether uncertainty was handled honestly. Where the source supports several readings, experts should adjudicate the accepted alternatives before scoring.

## 4. Manifest and command

The JSON schema is scoretwin-human-calibration-v1. Each pair has a unique ID, an independence_group identifying its source piece, a ratings mapping from anonymized rater IDs to integer 1–7, and metrics copied from a saved comparison report. repeat_group identifies blinded repeats. metric_directions declares whether a higher or lower metric value means greater identity.

Example values below are placeholders only; they are **not collected ratings**.

\`\`\`json
{
  "schema": "scoretwin-human-calibration-v1",
  "rating_scale": {
    "min": 1,
    "max": 7,
    "anchors": {"1": "clearly different", "7": "visually identical"}
  },
  "metric_directions": {
    "mask_iou": "higher",
    "geometry_f1": "higher",
    "center_error_sp": "lower"
  },
  "study_design": {
    "blinded": true,
    "randomized_order": true,
    "rating_dimension": "visual_structure",
    "preregistered": false
  },
  "pairs": [
    {
      "id": "pair-001",
      "independence_group": "source-piece-001",
      "ratings": {"rater-01": 3, "rater-02": 4, "rater-03": 3},
      "metrics": {"mask_iou": 0.72, "geometry_f1": 0.68, "center_error_sp": 0.31}
    },
    {
      "id": "repeat-001",
      "independence_group": "source-piece-001",
      "repeat_group": "pair-001",
      "ratings": {"rater-01": 4, "rater-02": 4, "rater-03": 3},
      "metrics": {"mask_iou": 0.72, "geometry_f1": 0.68, "center_error_sp": 0.31}
    }
  ]
}
\`\`\`

Run:

    scoretwin calibrate-human --study ./study/human-ratings.json --out ./results/human-calibration.json --bootstrap-samples 2000

The report includes Spearman association with 95% cluster-bootstrap intervals, pairwise quadratic-weighted kappa, within-rater repeat agreement, sample counts, and the sample-size heuristic. Repeat items are collapsed before metric correlation, so they do not double-count independent comparisons; bootstrap resampling uses independence_group.

## 5. Proposed go/no-go gates

These are **[IPOTEZĂ]** thresholds to preregister, not empirical facts:

- **Agreement:** mean pairwise quadratic-weighted kappa at least 0.60 and hidden-repeat exact agreement at least 0.70. If not, fix anchors or rater training before calibration.
- **Metric validity:** on held-out source pieces, at least one preregistered component should have |rho| >= 0.50 and a 95% cluster-bootstrap interval excluding zero. If not, revise the metric or investigate power/heterogeneity before adding more data.
- **Scalar DID:** never fit weights on the evaluation set. Fit any weighted score on training source groups and assess it on held-out groups. Publish weights, transforms, exclusions, and uncertainty. If weights change materially across genres/capture types, report conditional scores instead of a universal DID.
- **Abstention:** measure unknown-region coverage separately. A system must not inflate its score by omitting difficult glyphs without a coverage penalty.

## 6. Threats to validity

Main risks: source-piece leakage, raters guessing system identity from its style, too many simple printed pages, ambiguous anchors, visible repeats, multiple correlated pages from one piece, and retrospective tuning after seeing held-out results. Record exclusions and analysis decisions before inspecting correlations. A high correlation does not prove universal validity or out-of-distribution performance.
