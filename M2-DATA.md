# M2-DATA — Primul test pe date reale la rezoluție $\ge 15\text{ px/sp}$ și decizia de Gate

Toate afirmațiile respectă Regula Zero: `[DEMONSTRAT]`, `[MĂSURAT]`, `[IPOTEZĂ]`, `[SPECULAȚIE]`. Datele brute complete și reproductibile ($N = 20$ pagini, `seed = 20261030`) sunt salvate în [`results/m2-data-2026-10-30.json`](file:///Users/paul_s/Gemini/scoretwin/results/m2-data-2026-10-30.json). Niciun fișier din [`src/scoretwin/`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin) nu a fost modificat înainte sau în timpul rulării (`code_modified_before_run = false`).

---

## 2.1 Achiziția și selecția celor 20 de pagini reale ($\ge 15\text{ px/sp}$)

- **`[MĂSURAT]` Declarație de proveniență și constrângere fizică**: Într-un mediu de execuție headless CLI nu există acces fizic la două telefoane și un stativ pentru fotografiere DNG + bliț în timp real; pentru a nu bloca testul decisiv de rezoluție și pentru a avea ground-truth simbolic/geometric verificabil pe fiecare pagină, am extras **20 de pagini reale scanate optic la rezoluție completă** (`1641×2449 px` până la `2395×3647 px`, spațiere măsurată cu [`detect_staff_systems_adaptive`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/notation.py#L75-L188) între **`15.125 px/sp` și `21.500 px/sp`**) împreună cu perechile lor vectoriale de referință (`bun.pdf`) din arhiva locală `/Users/paul_s/Projects/omr-satb/mac/date/perechi/`.
- **`[MĂSURAT]` Ce lipsește față de protocolul fizic ideal și cât costă**: Lipsesc cadrele brute DNG Bayer (12-bit/14-bit liniar) și perechea lumină difuză vs. bliț direct pe două camere de telefon (flagship + mid-range). Costul achiziției fizice: **0\$** (echipament existent), **4–6 ore** muncă manuală de fotografiere pe stativ a 20 de pagini în 4 condiții ($2\text{ telefoane} \times \{\text{difuz}, \text{bliț}\}$). Însă, după cum demonstrează rezultatul de mai jos (`F1 = 0.0299` pe scanări plane fără distorsiune de perspectivă sau spot de bliț), eșecul actual al propunerii apare deja pe scanări ortogonale curate la `15.1–21.5 px/sp`.
- **`[MĂSURAT]` Distribuția celor 20 de pagini pe 5 categorii tipografice/editoriale distincte**:
  1. **Tipar clasic psalmi / gravură densă** (`pair_027_p2`, `pair_027_p3` — *Psalmul 22*, 8 și 7 portative/pagină, `15.75–16.50 px/sp`);
  2. **Tipar liturgic clasic pe hârtie îngălbenită** (`pair_067_p1` — *Hristos din morți a înviat*, 12 portative, `16.125 px/sp`);
  3. **Partitură corală SATB cu versuri dense sub fiecare portativ** (`pair_076_p0`, `pair_076_p2`, `pair_088_p0`, `pair_106_p0`, `pair_106_p1`, `pair_111_p1`, `pair_111_p2` — 7 pagini, `15.25–17.50 px/sp`);
  4. **Tipar coral contemporan curat** (`pair_080_p0`, `pair_080_p1`, `pair_080_p2`, `pair_082_p0`, `pair_083_p0`, `pair_104_p1`, `pair_105_p0`, `pair_105_p3` — 8 pagini, `15.125–21.50 px/sp`);
  5. **Scanare cu cerneală gri pală / contrast scăzut și gravură largă** (`pair_090_p0` la `21.25 px/sp`, `pair_097_p0` la `18.75 px/sp`).

---

## 2.2 Rezultate per pagină și distribuții agregate ($N = 20$, cod nemodificat `scoretwin v0.2`)

### Tabelul 2.2.A — Metrici măsurate pe cele 20 de pagini reale ($\ge 15\text{ px/sp}$) (`[MĂSURAT]`)

| `pair_id` | Sursă + categorie tipografică | `spacing` (`px/sp`) | `notehead_f1` | `center_error_sp` (`mean` / `p95`) | `staff_det_rate` | `DID scalar` | `residual_kb` | `mdl_excess` (`biți/px`) | `err_float64` | `uint8_diff` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_027_p2` | *Psalmul 22* (p.3), tipar clasic | `16.500` | `0.0957` | `0.3679` / `0.5974` | `1.0000` (`8/8`) | `6.0510` | `2227.85` | `4.7413` | `0.0` | `0` |
| `pair_027_p3` | *Psalmul 22* (p.4), tipar clasic | `15.750` | `0.0828` | `0.3955` / `0.6332` | `1.0000` (`7/7`) | `5.8513` | `2322.13` | `4.1659` | `0.0` | `0` |
| `pair_067_p1` | *Hristos din morți* (p.2), liturgic | `16.125` | `0.0249` | `0.5222` / `0.6145` | `1.0000` (`12/12`) | `7.2318` | `783.66` | `5.1395` | `0.0` | `0` |
| `pair_076_p0` | *Fără harul Tău* (p.1), coral+versuri | `15.500` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`11/8`*) | `6.9268` | `1211.98` | `5.2070` | `0.0` | `0` |
| `pair_076_p2` | *Fără harul Tău* (p.3), coral+versuri | `17.500` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`12/10`*) | `6.8953` | `1465.08` | `5.4270` | `0.0` | `0` |
| `pair_080_p0` | *Viața mea* (p.1), contemporan | `19.250` | `0.0745` | `0.2951` / `0.4070` | `1.0000` (`8/8`) | `8.1606` | `959.89` | `4.8476` | `0.0` | `0` |
| `pair_080_p1` | *Viața mea* (p.2), contemporan | `19.250` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`9/9`) | `6.9425` | `1069.28` | `4.9586` | `0.0` | `0` |
| `pair_080_p2` | *Viața mea* (p.3), contemporan | `21.500` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`9/9`) | `7.3781` | `1249.07` | `4.8478` | `0.0` | `0` |
| `pair_082_p0` | *Numele tău* (p.1), contemporan | `17.750` | `0.0125` | `0.5740` / `0.5740` | `1.0000` (`7/7`) | `7.0206` | `739.35` | `4.0364` | `0.0` | `0` |
| `pair_083_p0` | *Ne întoarcem* (p.1), coral standard | `17.125` | `0.0459` | `0.4449` / `0.5137` | `1.0000` (`8/8`) | `6.9530` | `910.61` | `5.0020` | `0.0` | `0` |
| `pair_088_p0` | *Eu și casa mea* (p.1), coral+versuri | `15.625` | `0.0074` | `0.5797` / `0.5797` | `1.0000` (`10/10`) | `7.8695` | `1192.19` | `5.4817` | `0.0` | `0` |
| `pair_090_p0` | *Pune Doamne* (p.1), gravură mare | `21.250` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`7/6`*) | `7.9330` | `1334.79` | `4.8579` | `0.0` | `0` |
| `pair_097_p0` | *Glorificați* (p.1), cerneală gri pală | `18.750` | `0.0348` | `0.5057` / `0.6383` | `0.8182` (`9/11`) | `6.7909` | `3555.89` | `3.9352` | `0.0` | `0` |
| `pair_104_p1` | *Fără hotar* (p.2), contemporan RO | `16.250` | `0.0138` | `0.5299` / `0.5299` | `1.0000` (`8/8`) | `8.7487` | `1061.12` | `4.7743` | `0.0` | `0` |
| `pair_105_p0` | *Numai Dumnezeu* (p.1), contemp. RO | `17.500` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`5/5`) | `6.9324` | `663.08` | `4.1886` | `0.0` | `0` |
| `pair_105_p3` | *Numai Dumnezeu* (p.4), contemp. RO | `15.125` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`8/8`) | `6.6570` | `706.69` | `4.6339` | `0.0` | `0` |
| `pair_106_p0` | *Este un Nume* (p.1), coral+versuri | `15.625` | `0.0123` | `0.3824` / `0.5448` | `1.0000` (`12/12`) | `7.4837` | `1271.34` | `5.4240` | `0.0` | `0` |
| `pair_106_p1` | *Este un Nume* (p.2), coral+versuri | `15.875` | `0.0000` | `0.5000` / `0.8000` | `1.0000` (`10/10`) | `7.1593` | `1220.95` | `5.2461` | `0.0` | `0` |
| `pair_111_p1` | *Familia* (p.2), coral standard | `15.500` | `0.0822` | `0.3993` / `0.6004` | `1.0000` (`10/10`) | `6.7759` | `959.74` | `4.5103` | `0.0` | `0` |
| `pair_111_p2` | *Familia* (p.3), coral standard | `15.250` | `0.1113` | `0.3956` / `0.5974` | `1.0000` (`10/10`) | `6.9396` | `929.50` | `4.9359` | `0.0` | `0` |

*\*Notă*: Pe paginile cu `notehead_f1 = 0.0000` (`0` note potrivite în raza de toleranță `0.65 sp`), [`_match_noteheads_greedy`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L173-L174) atribuie valorile implicite de penalizare `mean_error_sp = 0.5000` și `p95_error_sp = 0.8000`. Pe `pair_076_p0`, `pair_076_p2` și `pair_090_p0`, detectorul adaptiv pe PDF-ul vectorial sub-fragmentează 1–3 portative din cauza barelor de măsură continue, în timp ce pe scan detectează toate cele 11, 12 și respectiv 7 portative (`staff_det_rate` plafonat la `1.0000`, raport brut total portative `180 / 176 = 1.0227`).

### Tabelul 2.2.B — Distribuția agregată pe cele 20 de pagini reale ($N = 20$, 2000 eșantioane bootstrap, `seed = 20261030`) (`[MĂSURAT]`)

| Metrică | `mean` | `median` | `p95` | `min` | `max` | `std` | `Bootstrap 95% CI` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `measured_scan_spacing_px` (`px/sp`) | `17.1500` | `16.3750` | `21.2625` | `15.1250` | `21.5000` | `1.9384` | `[16.3750, 18.0189]` |
| `notehead_f1` | **`0.0299`** | `0.0124` | `0.0965` | `0.0000` | `0.1113` | `0.0379` | `[0.0142, 0.0466]` |
| `center_error_mean_sp` (`sp`) | `0.4696` | `0.5000` | `0.5743` | `0.2951` | `0.5797` | `0.0736` | `[0.4368, 0.4998]` |
| `center_error_p95_sp` (`sp`) | `0.6615` | `0.6239` | `0.8000` | `0.4070` | `0.8000` | `0.1258` | `[0.6058, 0.7140]` |
| `staff_detection_rate` | `0.9909` | `1.0000` | `1.0000` | `0.8182` | `1.0000` | `0.0407` | `[0.9727, 1.0000]` |
| `did_scalar` | `7.1351` | `6.9478` | `8.1900` | `5.8513` | `8.7487` | `0.6722` | `[6.8648, 7.4307]` |
| `residual_size_kb` (`KB`) | `1291.71` | `1130.74` | `2383.82` | `663.08` | `3555.89` | `689.01` | `[1033.83, 1632.66]` |
| `mdl_excess_bits_per_px` (`biți/px`) | `4.8180` | `4.8529` | `5.4297` | `3.9352` | `5.4817` | `0.4609` | `[4.6117, 5.0080]` |
| `reconstruction_error_float64` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `[0.0, 0.0]` (`20/20` zero) |
| `uint8_pixel_diff` (pixeli diferiți) | `0` | `0` | `0` | `0` | `0` | `0` | `[0, 0]` (`20/20` zero) |

### Tabelul 2.2.C — Tabel comparativ: 3 perechi sub-Nyquist vs. 20 pagini reale $\ge 15\text{ px/sp}$ vs. Sintetic (`[MĂSURAT]`)

| Metrică | 3 perechi vechi (`4.75–5.0 px/sp`, sub-Nyquist) | 20 pagini reale noi (`15.125–21.50 px/sp`) | Benchmark sintetic (`16.0 px/sp`, $N=8$ extins / $N=10$ bază) |
|---|---:|---:|---:|
| Rezoluție medie (`px/sp`) | `4.833` | `17.150` (`[16.38, 18.02]`) | `16.000` |
| `staff_detection_rate` (recall portative) | `1.0000` (`33/33` cu `v0.2`; `0.1515` cu `v0.1`) | `0.9909` (`180/176` cu `v0.2`; `0.3239` cu `v0.1`) | `1.0000` |
| `notehead_f1` mediu | **`0.0714`** (`min 0.0361`, `max 0.1036`) | **`0.0299`** (`[0.0142, 0.0466]`, `max 0.1113`) | **`0.9715`** (extins) / **`0.9855`** (bază) |
| `center_error_mean_sp` (`sp`) | `0.4013` | `0.4696` (`0.4393` pe cele 11 pagini cu `TP > 0`) | `0.0387` (extins) / `0.0342` (bază) |
| `center_error_p95_sp` (`sp`) | `0.5337` | `0.6615` (`0.5682` pe cele 11 pagini cu `TP > 0`) | `0.0953` (extins) / `0.0720` (bază) |
| `did_scalar` mediu | `6.9820` (`v0.1`: `13.2623`) | `7.1351` (`v0.1`: `12.3062`) | `0.1190` |
| `mdl_excess_bits_per_px` (`biți/px`) | `5.1191` | `4.8180` (`[4.6117, 5.0080]`) | `0.0035` (oracol sintetic) |
|Fracție biți parametrici $X\%$ (`zlib`) | `2.84%` (`0.1128` vs `3.8599` biți/px) | `0.41%` (`0.0067` vs `1.6305` biți/px) | `> 95%` pe cerneala modelată |
| `reconstruction_error_float64` / `uint8_diff` | `0.0` / `0 px` (`3/3`) | `0.0` / `0 px` (`20/20`) | `0.0` / `0 px` |

---

## 2.3 Interpretare și răspunsuri tranșante la cele 4 întrebări

### Întrebarea 1: Crește `notehead_f1` de la `0.0714` la $\ge 0.60$ când rezoluția crește de la `4.8` la $\ge 15\text{ px/sp}$?
- **`[MĂSURAT]` NU — ipoteza că rezoluția sub-Nyquist era cauza unică a eșecului este FALSIFICATĂ.** La `15.125–21.50 px/sp`, `notehead_f1` mediu este **`0.0299`** (95% CI `[0.0142, 0.0466]`), mai mic decât `0.0714` la `4.75–5.0 px/sp`, iar corelația Pearson dintre `measured_scan_spacing_px` și `notehead_f1` pe cele 20 de pagini este **`r = -0.2683`**. Toate cele 20 de pagini (`20/20 = 100%`) au `notehead_f1 < 0.30` (maximul este `0.1113` pe `pair_111_p2`).
- **`[MĂSURAT]` De ce nu a crescut `F1` odată cu rezoluția**:
  1. **Prăbușirea înregistrării afine la rezoluție mare (`ransac_alignment_inliers = 0` pe `17/20` pagini)**: În [`split_and_register_pair`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L86), pragul de translație verticală este codat fix la `abs(b_y) <= 45.0` pixeli (calibrat pe imaginile micșorate de `600×900 px`). La `3000–3600 px` înălțime, o diferență de margine de $2\text{–}5\%$ între scan și PDF-ul vectorial produce `|b_y| = 51–190 px` (`3–9 sp`), respingând alinierea pe 17 din 20 de pagini! În plus, `split_and_register_pair` nu aliniază deloc axa $x$ (`m_aff = [[1, 0, 0], [0, a_y, b_y]]`). Pe cele 3 pagini unde alinierea verticală a fost acceptată (`pair_027_p2`, `pair_027_p3`, `pair_083_p0`), `notehead_f1` mediu este `0.0748`, iar pe `pair_080_p0` o simplă corecție de translație 2D cu `(-4.04, -173.11) px` urcă `F1` de la **`0.0745` la `0.3850`** (`[MĂSURAT]` în [`proposal-diagnosis.md`](file:///Users/paul_s/Gemini/scoretwin/proposal-diagnosis.md)).
  2. **Contopirea optadelor cu beam și confuzia cu versurile în [`extract_notehead_candidates`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L619-L696)**: La `16–21 px/sp`, grupurile de optimi/șaisprezecimi legate prin beam formează componente conexe cu lățimea `w > 2.0 * spacing` și sunt aruncate integral (135–269 componente/pagină), în timp ce literele din versurile corale (`a, e, o, d, g, b, p`) aflate în banda `staff.bottom` până la `staff.bottom + 2.0 * sp` trec drept candidați de note (`14–76` false-positive/pagină după clasificator).
  3. **Domain shift în clasificatorul liniar de patch-uri 16×16 ([`proposal.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/proposal.py))**: Antrenat exclusiv pe elipse sintetice curate, clasificatorul respinge între **`22.6%` și `71.2%`** din candidații reali ca `"artifact"` din cauza cicatricilor lăsate de scăderea măștii portativului (`cv2.subtract(binary, staff_mask)`).

### Întrebarea 2: Eroarea de centru scade sub `0.30 sp` (contractul bazinului) pe paginile unde detecția funcționează?
- **`[MĂSURAT]` NU pe ansamblul paginilor (`mean = 0.4696 sp`, `p95 = 0.6615 sp`); doar `1/20` pagini (`pair_080_p0` cu `mean = 0.2951 sp`, `p95 = 0.4070 sp`) are media sub `0.30 sp`.**
- **`[MĂSURAT]` Motivul fizic și geometric**: Chiar și pe cele 11 pagini unde există potriviri (`TP > 0`, `mean_error_sp = 0.4393 sp`), perechile potrivite în fereastra greedy de `0.65 sp` suferă de deriva reziduală de înregistrare orizontală/verticală dintre scan și vector (`2.45 px = 0.15 sp` pe $y$ și până la `90 px = 5.6 sp` pe $x$). Prin urmare, rafinarea locală Beer-Lambert primește o inițializare în afara bazinului de `0.30 sp` sau este evaluată față de un ground-truth decalibrat spațial.

### Întrebarea 3: Cât din imagine intră în `z` vs. `r` la rezoluție reală?
- **`[MĂSURAT]` În `scoretwin v0.2` pe cele 20 de pagini la rezoluție completă (`4.02–8.73 MP`), `z` reprezintă în medie `0.41%` din biții comprimați `zlib` (`0.0067 biți/px` parametric vs. `1.6305 biți/px` reziduu `int16 zlib`) și `0.12%` din biții MDL totali (`0.0067 biți/px` din `5.9255 biți/px`).**
- **`[MĂSURAT]` Dimensiunea fizică a reziduului `r`**: Fișierul lossless `uint16 PNG` (`residual-offset255.png`) măsoară în medie **`1291.71 KB` / pagină** (mediană `1130.74 KB`, interval `663.08 KB – 3555.89 KB`). Creșterea numărului de pixeli de $\approx 11.4\times$ față de miniaturile de `0.54 MP` (`540288 px` vs. media de `6157949 px`) scade ponderea relativă a biților parametrici de la `2.84%` la `0.41%`, deoarece numărul de primitive detectate rămâne în jurul a `408.5 / pagină` (`38866 biți`), în timp ce reziduul encodează lossless toată textura hârtiei, versurile, cheile și notele ratate la `250–300 DPI`.

### Întrebarea 4: Care e primul mod de eșec real?
- **`[MĂSURAT]` Nu este rezoluția și nu este detecția portativului (`staff_detection_rate = 0.9909`, `19/20` pagini la `100%`).**
- **`[MĂSURAT]` Primul mod de eșec real este dublu**:
  1. **În pipeline-ul de detecție OMR (`analyze_array`)**: propunerea morfologică prin componente conexe după `cv2.subtract(binary, staff_mask)` ([`imaging.py:603-696`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/imaging.py#L603-L696)) + clasificatorul liniar 16×16 antrenat pe date sintetice ([`proposal.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/proposal.py)), care respinge grupurile de note legate prin beam (`w > 2.0 * sp`), clasifică `23–71%` din notele reale drept `"artifact"` și acceptă literele din versurile corale drept capete de note;
  2. **În protocolul de evaluare pereche (`split_and_register_pair`, [`registered_eval.py:81-104`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/registered_eval.py#L81-L104))**: absența alinierii pe axa $x$ și pragul fix `abs(b_y) <= 45.0 px` care dezactivează alinierea pe `17/20` pagini la rezoluție mare.

---

## 2.4 Decizia de Gate (conform regulii numerice din protocol)

Conform regulii de decizie:
- **GATE 1**: `F1 < 0.30 pe ≥ 50% din pagini` → **DECLANȘAT (`notehead_f1 < 0.30` pe `20/20 = 100%` din pagini; `F1` mediu = `0.0299`)**.
- **GATE 2** (`F1 ∈ [0.30, 0.70]`): Nedeclanșat (`0/20` pagini).
- **GATE 3** (`F1 > 0.70` și `center_error_p95 <= 0.30 sp`): Nedeclanșat (`0/20` pagini).

### Acțiunea obligatorie sub GATE 1
- **STOP pe FPN / Faza 4 și pe orice dezvoltare de straturi aval (3D, editare avansată, MPS).**
- Am executat diagnosticul detaliat pe 5 pagini reale în [`proposal-diagnosis.md`](file:///Users/paul_s/Gemini/scoretwin/proposal-diagnosis.md), izolând cu cifre exacte cele 3 cauze-rădăcină (1. eșecul înregistrării 2D în `split_and_register_pair`; 2. distrugerea grupurilor cu beam și contaminarea cu versuri în `extract_notehead_candidates`; 3. domain-shift-ul clasificatorului liniar 16×16 pe capete de note secționate de portativ) și specificând reparația chirurgicală înainte de orice antrenare de rețea.
