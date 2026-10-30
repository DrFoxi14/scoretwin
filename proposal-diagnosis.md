# `proposal-diagnosis.md` — Diagnosticul GATE 1 pe 5 pagini reale la rezoluție completă ($\ge 15\text{ px/sp}$)

Acest raport este generat ca urmare a declanșării **GATE 1** (`notehead_f1 < 0.30` pe `20/20 = 100%` din pagini, `F1` mediu = `0.0299` în [`results/m2-data-2026-10-30.json`](file:///Users/paul_s/Gemini/scoretwin/results/m2-data-2026-10-30.json)). Am instrumentat fără modificări de cod pâlnia de propunere și aliniere pe 5 pagini reprezentative din setul M2-data (`pair_027_p2`, `pair_080_p0`, `pair_097_p0`, `pair_106_p0`, `pair_111_p2`, rezoluții `15.25–19.25 px/sp`). Toate afirmațiile poartă eticheta epistemică obligatorie.

---

## 1. Pâlnia cantitativă măsurată pe cele 5 pagini reale (`[MĂSURAT]`)

| Metrică de etapă în pipeline | `pair_027_p2` (`16.13 px/sp`) | `pair_080_p0` (`19.25 px/sp`) | `pair_097_p0` (`18.75 px/sp`) | `pair_106_p0` (`16.00 px/sp`) | `pair_111_p2` (`15.25 px/sp`) |
|---|---:|---:|---:|---:|---:|
| Portative detectate (`scan` / `ref`) | `8` / `8` | `8` / `8` | `9` / `11` | `12` / `12` | `10` / `10` |
| `ransac_alignment_inliers` ([`registered_eval.py:86`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L86)) | `40` (aliniat Y) | `0` (respins) | `0` (respins) | `0` (respins) | `0` (respins) |
| Parametrii afini verticali `(a_y, b_y_px)` | `(0.9913, +20.40)` | `(1.0093, -189.68)` | `N/A (9 != 11)` | `(1.0899, -51.09)` | `(1.1021, -60.54)` |
| Deplasare 2D reală `(dx, dy)` în `sp` (`phaseCorrelate`) | `(+5.635, +0.238)` | `(-0.210, -8.993)` | `(-13.917, -40.936)` | `(-13.906, +41.963)` | `(-0.292, +9.782)` |
| Total componente conexe în `clean` (`cc_total`) | `3290` | `2271` | `9379` | `1667` | `2935` |
| Respinse: arie prea mică (`area < 0.035 * sp^2`) | `2320` | `1210` | `5976` | `602` | `1551` |
| Respinse: arie prea mare (`area > 3.0 * sp^2`) | `27` | `20` | `20` | `41` | `53` |
| Respinse: prea înguste (`w < 0.52 * sp`, stem/barline) | `239` | `260` | `1250` | `268` | `460` |
| **Respinse: prea late (`w > 2.0 * sp`, grupuri cu beam / cuvinte)** | **`141`** | **`135`** | **`269`** | **`168`** | **`197`** |
| Trecute de filtrul geometric CC (`passed_cc_filter`) | `563` | `646` | `1864` | `588` | `674` |
| Candidați extrași de [`extract_notehead_candidates`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L587) | `212` | `181` | `370` | `153` | `239` |
| — din care în zona de versuri (`|cy - staff| > 0.55 sp`) | `49` (`23.1%`) | `17` (`9.4%`) | `69` (`18.6%`) | `9` (`5.9%`) | `83` (`34.7%`) |
| **Respinse de clasificatorul liniar 16×16 (`artifact`/`unknown`)** | **`86` (`40.6%`)** | **`116` (`64.1%`)** | **`143` (`38.6%`)** | **`109` (`71.2%`)** | **`54` (`22.6%`)** |
| Note păstrate după clasificator (`filled` + `hollow`) | `126` | `65` | `227` | `44` | `185` |
| — din care fals-pozitive în zona de versuri | `37` (`29.4%`) | `14` (`21.5%`) | `55` (`24.2%`) | `6` (`13.6%`) | `76` (`41.1%`) |
| Note în referința digitală (`ref_notes_n`) | `271` | `122` | `237` | `278` | `336` |
| `notehead_f1` brut (fără corecția translației 2D) | **`0.0957`** | **`0.0745`** | **`0.0348`** | **`0.0123`** | **`0.1113`** |
| `notehead_f1` după simpla translație rigidă 2D `(dx, dy)` | **`0.1310`** | **`0.3850`** | **`0.0261`** | **`0.0186`** | **`0.2035`** |

---

## 2. Top 3 cauze-rădăcină ale eșecului (ordonate după impactul măsurat)

### Cauza 1: Eșecul înregistrării 2D în [`split_and_register_pair`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L47-L112) (dezalinieri de `5–41 sp` pe `17/20` pagini)
- **`[DEMONSTRAT]` + `[MĂSURAT]` Mecanismul exact în cod**:
  1. La linia [`registered_eval.py:86`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L86), condiția de acceptare a transformării afine verticale este `if 0.90 <= a_y <= 1.10 and abs(b_y) <= 45.0:`. Pragul `45.0 px` a fost scris pentru miniaturi de `603×896 px` (`4.75 px/sp`, unde `45 px = 9.5 sp`). La rezoluție reală (`3000–3647 px` înălțime, `16–21.5 px/sp`), o diferență normală de margine superioară de $2\text{–}5\%$ între scan și PDF-ul vectorial produce `b_y = -189.68 px` pe `pair_080_p0`, `b_y = -51.09 px` pe `pair_106_p0` și `b_y = -60.54 px` (`a_y = 1.1021`) pe `pair_111_p2`.Drept urmare, **17 din cele 20 de pagini au `ransac_alignment_inliers = 0`**, iar imaginile rămân complet nealiniate vertical (`8.99 sp` decalaj pe `pair_080_p0`, `9.78 sp` pe `pair_111_p2`), ceea ce face imposibilă potrivirea în raza `tolerance_sp = 0.65 sp`!
  2. La liniile [`registered_eval.py:87-90`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L87-L90), matricea afină este `[[1.0, 0.0, 0.0], [0.0, float(a_y), float(b_y)]]`, adică **axa orizontală $x$ nu este niciodată scalată sau translatată (`a_x = 1.0, b_x = 0.0`)**. Pe `pair_027_p2`, deși alinierea pe $y$ reușește (`residual_staff_dy_px = 2.45 px = 0.15 sp`), există un decalaj orizontal necompensat de `dx = +90.87 px` (`5.635 sp`) între marginea stângă a scanului și cea a vectorului!
  3. Când aplicăm o simplă translație rigidă 2D `(dx, dy)` estimată prin `cv2.phaseCorrelate` pe `pair_080_p0`, `notehead_f1` sare instantaneu de la **`0.0745` la `0.3850` (`+416%`)**, iar pe `pair_111_p2` sare de la **`0.1075` la `0.2035` (`+89%`)** (`[MĂSURAT]`).

### Cauza 2: Distrugerea grupurilor legate prin beam (`w > 2.0 * sp`) și contaminarea cu versuri în [`extract_notehead_candidates`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L587-L800)
- **`[DEMONSTRAT]` + `[MĂSURAT]` Mecanismul exact în cod**:
  1. **Aruncarea grupurilor de optimi/șaisprezecimi legate prin beam**: La linia [`imaging.py:619`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L619), orice componentă conexă din `clean = cv2.subtract(binary, staff_mask)` care are lățimea `w > 2.0 * spacing` (sau `area > 3.0 * spacing^2` la linia `613`) este respinsă **înainte** de analiza profilului pe rânduri (`sub_boxes`). Într-o partitură reală la `15–21 px/sp`, 2–4 optimi legate printr-un beam orizontal sau oblic (ale căror stem-uri nu sunt tăiate de liniile de portativ) formează o singură componentă conexă cu lățimea `w ∈ [3.5 sp, 12.0 sp]` și sunt șterse complet! Pe cele 5 pagini sunt aruncate astfel între **`135` și `269` componente largi per pagină** (`141` pe `pair_027_p2`, `168` pe `pair_106_p0`, `197` pe `pair_111_p2`), ceea ce explică de ce `extract_notehead_candidates` extrage doar `153–239` candidați pe pagini cu `271–336` note reale (`[MĂSURAT]`).
  2. **Acceptarea literelor din versurile corale drept capete de note**: La linia [`imaging.py:695`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L695), banda verticală permisă în jurul fiecărui portativ este `[staff.top - 2.0 * spacing, staff.bottom + 2.0 * spacing]`. În partiturile corale SATB, silabele versurilor sunt tipărite exact între `staff.bottom + 0.6 * spacing` și `staff.bottom + 1.9 * spacing`. Literele minuscule cu buclă închisă (`a, e, o, d, b, p, g`) la `16 px/sp` au înălțimea `bh ∈ [0.45 sp, 0.85 sp]` și lățimea `bw ∈ [0.52 sp, 0.90 sp]`, trecând toate filtrele geometrice și de clasificator. Pe `pair_111_p2`, **`76 din cele 185 de note finale (`41.1%`)`** sunt în realitate litere din versuri! Pe `pair_027_p2`, **`37 din 126 (`29.4%`)`** sunt litere din versuri (`[MĂSURAT]`).

### Cauza 3: Domain-shift sever în clasificatorul liniar 16×16 ([`proposal.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/proposal.py)) pe capete de note secționate de portativ
- **`[DEMONSTRAT]` + `[MĂSURAT]` Mecanismul exact în cod**:
  1. Clasificatorul liniar softmax cu 6 clase din [`proposal.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/proposal.py) este antrenat exclusiv pe patch-uri sintetice `16×16` generate procedural (elipse pline/goale perfecte peste linii sintetice uniforme).
  2. Pe scanările reale (`pair_080_p0`, `pair_106_p0`), liniile de portativ au variații de grosime și halou de toner; deși patch-ul `crop` este extras din `gray` ([`imaging.py:784`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L784)), centrul `(cx, cy)` al unei note aflate pe linie este deplasat cu `0.15–0.30 sp` atunci când o jumătate din elipsă este erodată de `staff_mask`, sau când atașamentul stem-ului/beam-ului intră în fereastra `crop_half_width = 1.1 * spacing`.
  3. Drept urmare, clasificatorul etichetează drept `"artifact"` sau `"unknown"` **`109 din 153 candidați (`71.2%`)`** pe `pair_106_p0` și **`116 din 181 candidați (`64.1%`)`** pe `pair_080_p0` (`[MĂSURAT]`), prăbușind recall-ul la `< 0.16` chiar înainte de aliniere.

---

## 3. Planul chirurgical de remediere (fără FPN, fără „mai multe date”, cost 0\$, timp 1 săptămână)

1. **Repararea înregistrării 2D în [`split_and_register_pair`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L47-L112)** (`[IPOTEZĂ]` — falsificabilă prin rularea pe cele 20 de pagini M2-data; țintă: `ransac_inliers > 0` pe $\ge 18/20$ pagini și eroare de aliniere a sistemelor $< 0.15\,\text{sp}$ pe ambele axe):
   - Înlocuirea pragului fix `abs(b_y) <= 45.0` cu un prag relativ la înălțimea paginii (`abs(b_y) <= 0.12 * target_h`);
   - Potrivirea portativelor prin programare dinamică / RANSAC 1D pe distanțele dintre portative (astfel încât $9$ vs. $11$ portative detectate pe `pair_097_p0` să nu blocheze potrivirea celor $9$ portative comune);
   - Estimarea simultană a scalei și translației orizontale $(a_x, b_x)$ din capetele stânga/dreapta ale liniilor de portativ (`x0_px, x1_px`) și rafinarea fină 2D prin `cv2.findTransformECC(..., cv2.MOTION_AFFINE)` pe măștile de portativ + barlines.
2. **Descompunerea grupurilor cu beam înainte de filtrul de lățime și suprimarea benzii de versuri în [`extract_notehead_candidates`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L587-L800)** (`[IPOTEZĂ]` — falsificabilă pe cele 5 pagini de diagnostic; țintă: reducerea fals-pozitivelor din versuri sub `5%` și dublarea recall-ului pe optimi legate):
   - Tăierea componentelor largi (`w > 1.8 * sp`) de-a lungul stem-urilor verticale detectate morfologic (`detect_stem_candidates`) sau prin deschidere morfologică eliptică (`0.65 * sp × 0.45 * sp`) înainte de respingerea după `w` și `area`;
   - Restrângerea benzii verticale implicite de la `±2.0 * sp` la `[staff.top - 1.15 * sp, staff.bottom + 1.15 * sp]`, permițând candidați la distanțe între `1.15 sp` și `2.0 sp` **doar** dacă există o linie suplimentară (ledger line) orizontală locală sau un stem atașat care pătrunde în portativ.
3. **Înlocuirea respingerii dure din clasificatorul liniar 16×16 cu scorul fizic de elipsă Beer-Lambert + stem-attachment** (`[IPOTEZĂ]` — falsificabilă pe cele 20 de pagini M2-data; țintă: `notehead_f1 >= 0.55` pe scanări aliniate):
   - Un candidat geometric aflat pe portativ (sau atașat la un stem validat) care obține corelație de șablon eliptic `fit_score >= 0.28` în [`refine_notehead_geometry`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L935) și adâncime optică medie $\bar{\tau} \ge 0.35$ nu trebuie șters de clasificatorul liniar `16×16` când acesta prezice `"artifact"`.
