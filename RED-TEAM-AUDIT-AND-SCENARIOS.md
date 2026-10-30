# RED-TEAM AUDIT & TORTURE SCENARIOS — `ScoreTwin-Reasoning` (IDSR)

Acest document răspunde atacului Red-Team asupra **IDSR (Identic Digital Score Reconstruction)** prin două componente:
1. **Suita de tortură executată și măsurată** în [`src/scoretwin/redteam_scenarios.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/redteam_scenarios.py), ale cărei rezultate trasabile sunt salvate în [`results/redteam-torture-test-2026-10-30.json`](file:///Users/paul_s/Gemini/scoretwin/results/redteam-torture-test-2026-10-30.json) (`SHA-256: c83ad5070f3b5791dc8a3ba3ca537ce2473a977dd27ec48f477e8758a03cb118`, `0` abțineri, `6/6` scenarii trecute în `0.23 s`).
2. **Matricea extinsă de atac Red-Team (12 scenarii adversariale)** care acoperă versurile la coordonate fizice exacte, adnotările umane complexe (notă încercuită + săgeată + `"attention"`), dinamici (`mf`, `f`, `<`), triolete (`[ 3 ]`), disambiguarea în 3 căi `Tie` vs. `Legato Slur` vs. `Phrasing Slur`, bleed-through verso și încă 6 cazuri-limită critice.

---

## 1. Cele 6 atacuri executate și măsurate în `redteam_scenarios.py`

### Scenariul 1 — Versurile (silabe, cratime `-`, melisme `_`) la poziția fizică exactă de pe foaie

- **Atacul Red-Team**:
  1. În diagnosticul anterior (`proposal-diagnosis.md`), între `13.6%` și `41.1%` din alarmele fals-pozitive de capete de notă pe partituri corale reale proveneau din litere rotunde de vers (`a, e, o, d, g, b, p`) aflate în banda inter-portativ `[MĂSURAT]`.
  2. Orice OMR sau gravor modern (MuseScore / Finale / LilyPond) recentrează automat silaba sub capul notei, distrugând poziția fizică originală a tipografului sau a copistului `[DEMONSTRAT]`.
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning`)**:
  - **Separare stratigrafică și de bandă**: Banda de versuri (`staff_bottom + 1.5·sp` până la `staff_bottom + 4.5·sp`) este separată topologic de portativ *înainte* de detecția elipselor de notă, grupând caracterele adiacente orizontal (nucleu morfologic $14 \times 3\text{ px}$) în tokeni de silabă (`lyric_syllable`) și cratime (`lyric_hyphen`) `[DEMONSTRAT]`.
  - **Coordonate fizice nenormalizate (`normalized_or_recentered = false`)**: Pentru fiecare silabă sau cratimă se salvează originea fizică exactă $(x_{\text{cm}}, y_{\text{cm}})$ față de prima bară de măsură și față de linia inferioară a portativului, fără nicio recentrare sub notă în Straturile 1 și 2 `[MĂSURAT]`:
    - Token 1 (`"Doam"`): `exact_x_from_bar1_cm = 0.8805 cm`, `exact_y_below_staff_cm = 0.4995 cm`;
    - Token 2 (`"-"`): `exact_x_from_bar1_cm = 1.6002 cm`, `exact_y_below_staff_cm = 0.4995 cm`;
    - Token 3 (`"ne"`): `exact_x_from_bar1_cm = 3.2343 cm`, `exact_y_below_staff_cm = 0.5249 cm`;
    - Token 4 (`"sfant"`): `exact_x_from_bar1_cm = 4.4789 cm`, `exact_y_below_staff_cm = 0.4911 cm`.
  - **Verificare**: `lyrics_confused_as_noteheads = 0`, `max_horizontal_origin_error_cm = 0.0169 cm` (`0.169 mm`, adică exact $2\text{ px}$ de bearing tipografic al fontului la `300 DPI`, cu `0.0 cm` drift de recentrare) `[MĂSURAT]`.

---

### Scenariul 2 — Omul încercuiește o notă, trage o săgeată și scrie `"attention"`

- **Atacul Red-Team**:
  - Un interpret sau dirijor încercuiește cu creionul/pixul o notă (`note-7-encircled` la `(1140, 380)`), trage o săgeată oblică peste liniile portativului și scrie `"attention"` în marginea de sus.
  - Într-un OMR clasic binarizat, cercul + săgeata + liniile portativului + nota fuzionează într-o singură componentă conexă uriașă: nota încercuită dispare din MusicXML, iar adnotarea este fie ștearsă ca „zgomot", fie halucinată ca un grup de optimi cu beam `[DEMONSTRAT]`.
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning`)**:
  1. **Demixare fizică de strat ($z = 2$ vs. $z = 1$)**: Grafitul creionului (sau cerneala secundară de adnotare) are reflectanță speculară diferită ($\text{specular\_map} \ge 0.12$) și adâncime optică aditivă $\tau_{\text{total}}(x,y) = \tau_{\text{print}}(x,y) + \tau_{\text{annot}}(x,y)$ `[DEMONSTRAT]`.
  2. **Recuperarea intactă a notei de dedesubt ($z = 1$)**: După scăderea câmpului $\tau_{\text{annot}}$, nota `note-7-encircled` este recuperată la coordonatele `(1140.14, 379.747)` față de adevărul fizic `(1140.0, 380.0)` — o eroare de centru de doar **`0.0145 sp`** (`0.29 px`) `[MĂSURAT]`.
  3. **Descompunerea topologică a adnotării ($z = 2$)**: Pe masca stratului $z=2$, analiza ierarhiei contururilor (`cv2.RETR_CCOMP`) identifică:
     - **Bucla închisă (`circle-0001`)** al cărei interior îl conține topologic pe `note-7-encircled` (`closed_loop_enclosure_detected = true`);
     - **Vectorul orientat al săgeții (`arrow-0001`)** de la `(1248.0, 232.0)` către `(1162.0, 358.0)` (vârful săgeții indică conturul exterior al buclei închise);
     - **Textul marginal (`"attention"`)** la `bbox_px = [1256, 216, 94, 17]`, aflat la exact `2.5061 cm` la dreapta barei de măsură 2 `[MĂSURAT]`.
  4. **Relație semantică păstrată în Stratul 2**: Adnotarea nu este ștearsă și nici confundată cu notele; este stocată pe stratul fizic $z=2$ ca obiect de tip `performer_directive_annotation` legat explicit: `"attention" -> arrow-0001 -> circle-0001 -> note-7-encircled` `[MĂSURAT]`.

---

### Scenariul 3 — Dinamici (`mf`, `f`, `p`, `sfz`) și Hairpins (`<` crescendo / `>` diminuendo)

- **Atacul Red-Team**:
  - Furcile de crescendo (`<`) și diminuendo (`>`) sunt confundate de detectoarele liniare cu segmente de beam sau șterse de morfologia liniei de portativ; glifele cursive `mf`, `p`, `f` au înălțime comparabilă cu literele de vers.
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning`)**:
  - Dinamicile sunt localizate sub/deasupra portativului și discriminate prin geometria deschiderii: pentru orice pană orizontală cu lățime $w \ge 4.0\cdot\text{sp}$, sistemul măsoară dispersia verticală la capătul stâng ($\Delta y_{\text{left}}$) vs. capătul drept ($\Delta y_{\text{right}}$):
    - $\Delta y_{\text{right}} > \Delta y_{\text{left}} \implies$ **`hairpin_crescendo`** (`<`);
    - $\Delta y_{\text{left}} > \Delta y_{\text{right}} \implies$ **`hairpin_diminuendo`** (`>`) `[DEMONSTRAT]`.
  - Toate cele 3 elemente dinamice de pe pagina de test sunt detectate la coordonatele fizice exacte `[MĂSURAT]`:
    1. `dynamic_glyph_mf` la `0.7705 cm` de bara 1 (`width = 0.2709 cm`);
    2. `hairpin_crescendo` la `1.3039 cm` de bara 1 (`width = 1.9050 cm`);
    3. `dynamic_glyph_f` la `3.3782 cm` de bara 1 (`width = 0.1016 cm`).

---

### Scenariul 4 — Triolete / Tuplete (`[ 3 ]`) și conservarea ritmică a măsurii

- **Atacul Red-Team**:
  - Dacă cifra `3` (cu sau fără acoladă/bracket pătrat `[ 3 ]`) este ignorată sau confundată cu o digitație, suma duratelor din măsură devine $6.0$ timpi în loc de $5.0$ timpi, invalidând matematic măsura `[DEMONSTRAT]`.
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning`)**:
  - Cifra `3` împreună cu acolada orizontală `[ 3 ]` de deasupra beam-ului este asociată grupului de 3 note (`note-4`, `note-5`, `note-6`), aplicând factorul rațional de scalare $\frac{2}{3}$:
    - Suma brută fără scalare de triolet: `6.0` timpi;
    - Suma cu scalare de triolet ($1 + 1 + 1 + 3 \times \frac{2}{3} = 5.0$): **`5.0` timpi** (`matches_time_signature_5_4_or_measure_sum = true`) `[MĂSURAT]`.

---

### Scenariul 5 — Tie (legato de durată) vs. Legato Slur (articulație) vs. Phrasing Slur (legătură de frază)

- **Atacul Red-Team**:
  - Toate cele trei sunt curbe Bézier netede! Un detector vizual pur spune doar „curbă". Dar muzical și fizic ele sunt trei obiecte complet diferite:
    1. **Tie (legato de prelungire a duratei)**: unește două note succesive de **aceeași înălțime** (fără note intermediare) și le însumează duratele într-un singur sunet continuu;
    2. **Legato Slur (legătură de articulație)**: unește un motiv local de note de **înălțimi diferite** (interpretate *legato*, fără re-atac sec);
    3. **Phrasing Slur (legătură de frază)**: un arc macro care acoperă o întreagă frază (sau măsură) și **înglobează la interior** alte articulații (inclusiv tie-uri, legato slur-uri locale și triolete).
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning`)**:
  - Fiecare arc este extras ca o curbă Bézier cubică cu 4 puncte de control exacte $(P_0, P_1, P_2, P_3)$ în pixeli/cm și clasificat determinist prin topologia notelor cuprinse în intervalul $[P_{0,x}, P_{3,x}]$ `[DEMONSTRAT]`:
    1. **Arcul 1 (`note-1` $\to$ `note-2`, `G4` $\to$ `G4`, `span = 1.0329 cm = 6.10 sp`, 2 note cuprinse, `pitch_start == pitch_end`)** $\implies$ **`TIE_DURATION_BINDING`** `[MĂSURAT]`;
    2. **Arcul 2 (`note-2` $\to$ `note-3`, `G4` $\to$ `B4`, `span = 1.1345 cm = 6.70 sp`, 2 note de înălțimi diferite)** $\implies$ **`LEGATO_ARTICULATION_SLUR`** `[MĂSURAT]`;
    3. **Arcul 3 (`note-1` $\to$ `note-6`, `span = 5.0546 cm = 29.85 sp`, 6 note cuprinse, înglobând la interior Arcul 1, Arcul 2 și trioletul)** $\implies$ **`PHRASING_SLUR_MACRO_ARCH`** `[MĂSURAT]`.

---

### Scenariul 6 — Bleed-through de pe verso vs. cerneală de pe recto

- **Atacul Red-Team**:
  - Pe manuscrise și tipărituri pe hârtie subțire, cerneala de pe verso difuzează prin fibrele hârtiei și apare pe recto ca o „notă-fantomă" exact între liniile portativului.
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning`)**:
  - Cerneala aplicată direct pe recto are un profil de margine abrupt (convoluție cu PSF-ul peniței/tiparului), în timp ce cerneala difuzată prin grosimea hârtiei ($d_{\text{paper}} \approx 100\text{--}180\,\mu\text{m}$) suferă o convoluție izotropă de difuzie laterală ($\sigma_{\text{bleed}} \gg \sigma_{\text{recto}}$) `[DEMONSTRAT]`.
  - Măsurând raportul de ascuțime a gradientului orizontal $|\partial \tau / \partial x| / \max(\tau)$ perpendicular pe liniile portativului:
    - Notă reală pe recto (`note-8` la `(1420, 400)`): **`0.4997`**;
    - Urmă bleed-through de pe verso (la `(1580, 390)`): **`0.1071`** ($4.66\times$ mai difuză decât cerneala recto) $\implies$ **`PROVEN_VERSO_BLEEDTHROUGH_EXCLUDED_FROM_RECTO_SCORE`**, păstrată vizual în Stratul 1 (`preserved_in_layer1_visual_twin = true`) și exclusă din notele recto în Stratul 2 `[MĂSURAT]`.

---

### Scenariul 7 — Prima / Seconda / Terza Volta (`1.`, `2.`, `3.`) + Bare de repetiție (`|:`, `:|`) + Execuție PLAYABLE (Unrolling)

- **Atacul Red-Team**:
  1. Cele două puncte ale barei de repetiție (`|:` sau `:|`) stau exact în spațiile 2 și 3 ale portativului — un detector naiv le confundă cu puncte de augmentare (`dot`).
  2. Casetele `1.`, `2.`, `3.` (*prima / seconda / terza volta*) de deasupra portativului au geometrie diferită la capătul drept: `1.` și `2.` sunt casete închise (`┌─── 1. ───┐`, `ending_type = "start_stop"`), pe când `3.` este deschisă la dreapta (`┌─── 3. ────`, `ending_type = "start_discontinue"`).
  3. Dacă sistemul doar le desenează fără să le înțeleagă mașina de stări, playerul cântă liniar `M1 -> M2 -> M3 -> M4` (greșit!) în loc să sară peste `1.` la a doua trecere și peste `1.` și `2.` la a treia trecere `[DEMONSTRAT]`.
- **Cum rezolvă IDSR (`ScoreTwin-Reasoning` + [`compile_playable_volta_repeat_schedule`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/redteam_scenarios.py#L700-L828))**:
  1. **Detecția fizică a barelor de repetiție și excluderea confuziei cu punctele de augmentare**:
     - Punctele din spațiile 2 și 3 ($y = 790, 810$) adiacente unei bare duble sunt clasificate după semnul deplasării $\Delta x = x_{\text{dots}} - x_{\text{bar}}$:
       - $\Delta x > 0$ la `x = 120` (`0.0 cm`) $\implies$ **`forward_repeat_start` (`|:`)**;
       - $\Delta x < 0$ la `x = 900` (`6.6040 cm`) și `x = 1290` (`9.9060 cm`) $\implies$ **`backward_repeat_end` (`:|`)**;
       - `repeat_dots_confused_as_augmentation_dots = 0` `[MĂSURAT]`.
  2. **Detecția geometrică exactă a casetelor Volta `1.`, `2.`, `3.` (cârlig stânga vs. cârlig dreapta)**:
     - **Volta `1.` (`M2`)**: `exact_x_start_cm = 3.3274 cm`, `exact_x_end_cm = 6.5701 cm` (`span = 3.2427 cm`), cârlig stânga `0.2371 cm`, cârlig dreapta închis `0.2371 cm` $\implies$ **`ending_type = "start_stop"`** `[MĂSURAT]`;
     - **Volta `2.` (`M3`)**: `exact_x_start_cm = 6.6463 cm`, `exact_x_end_cm = 9.8721 cm` (`span = 3.2258 cm`), cârlig stânga `0.2371 cm`, cârlig dreapta închis `0.2371 cm` $\implies$ **`ending_type = "start_stop"`** `[MĂSURAT]`;
     - **Volta `3.` (`M4`)**: `exact_x_start_cm = 9.9653 cm`, `exact_x_end_cm = 13.1826 cm` (`span = 3.2173 cm`), cârlig stânga `0.2371 cm`, capăt dreapta deschis (`0.0085 cm`, `right_hook_detected_closed = false`) $\implies$ **`ending_type = "start_discontinue"`** `[MĂSURAT]`.
  3. **Compilare PLAYABLE (desfășurarea automată a trecerilor 1, 2 și 3)**:
     - Măsuri scrise pe foaie: `["M1", "M2", "M3", "M4"]` (`4` măsuri, `8` note scrise);
     - **Ordinea desfășurată PLAYABLE**: **`["M1", "M2", "M1", "M3", "M1", "M4"]`** (`6` măsuri cântate, `12` note cântate = `24.0` timpi / `12.0 s` la `120 BPM`) `[MĂSURAT]`:
       - **Trecerea 1 (`pass = 1`)**: cântă `M1` (`E4, F4`) $\to$ intră în `1.` (`M2`: `G4, A4`) $\to$ lovește `:|` la finalul lui `M2` $\to$ sare înapoi la `|:` (`M1`);
       - **Trecerea 2 (`pass = 2`)**: cântă `M1` (`E4, F4`) $\to$ ajunge la `M2` (`1.`), vede `1 != 2`, **sare peste `M2` (`1.`)** și intră direct în `2.` (`M3`: `B4, C5`) $\to$ lovește `:|` la finalul lui `M3` $\to$ sare înapoi la `|:` (`M1`);
       - **Trecerea 3 (`pass = 3`)**: cântă `M1` (`E4, F4`) $\to$ **sare peste `M2` (`1.`) și `M3` (`2.`)** și intră direct în `3.` (`M4`: `D5, E5`, casetă deschisă `discontinue`) $\to$ încheie piesa `[MĂSURAT]`.

---

## 2. Matricea extinsă Red-Team: încă 6 scenarii adversariale și mecanismul de rezolvare în IDSR

Pentru a nu lăsa niciun unghi mort în arhitectura [`ARCHITECTURE.md`](file:///Users/paul_s/Gemini/scoretwin/ARCHITECTURE.md), tabelul de mai jos atacă încă 6 scenarii reale din partituri complexe (pian, cor, manuscrise cu corecturi):

| # | Scenariu de atac Red-Team | De ce eșuează OMR-ul clasic | Cum îl rezolvă fizic și contextual `ScoreTwin-Reasoning` | Etichetă epistemică & Test de falsificare |
| :- | :--- | :--- | :--- | :--- |
| **7** | **Cross-staff beaming** (un singur beam traversează spațiul dintre portativul mâinii drepte și cel al mâinii stângi la pian) | Segmentarea clasică taie pagina în benzi independente per portativ; notele de pe portativul inferior rămân fără beam, iar beam-ul oblic este tăiat în două cioturi. | **Graf topologic la nivel de sistem (`system-level stem-beam graph`)**: Stems-urile sunt urmărite continuu între portativele aceluiași sistem acoladat; beam-ul oblic păstrează coordonatele fizice $(x,y)$ exacte pe pagină, iar fiecare notă își păstrează `staff_id`-ul fizic (`staff=1` sau `staff=2`) și vocea comună. | `[IPOTEZĂ]` — Falsificare: pe 20 de măsuri de Debussy/Ravel cu cross-staff beams, dacă o singură legătură stem–beam este ruptă la granița dintre portative, graful de sistem e falsificat. |
| **8** | **Ștersătura compozitorului (*rasura* / barare cu X sau hașură densă) vs. măsură *ossia*** | Un OMR clasic fie încearcă să citească notele barate ca note active (stricând suma ritmică a măsurii), fie le șterge complet (pierzând istoria genetică a manuscrisului). | **Stratigrafie genetică în 2 straturi ($z=1$ text de bază, $z=2$ ștersătură)**: Hașura oblică de anulare are orientare și adâncime optică suprapusă peste notele inițiale ($\tau_{\text{stroke}} + \tau_{\text{note}}$). În Stratul 1 și Stratul 2, notele de sub ștersătură sunt recuperate în sub-stratul genetic `cancelled_composer_draft` (nu intră în redarea activă a măsurii finale, dar sunt explorabile 3D sub hașură). | `[IPOTEZĂ]` — Falsificare: pe o măsură barată din autograful lui Beethoven, dacă notele de sub hașură sunt amestecate cu măsura activă sau șterse din Stratul 1, protocolul genetic eșuează. |
| **9** | **Coliziune de unison și secundă mică între două voci pe același portativ** (două capete de notă lipite sau suprapuse, cu stems în direcții opuse $\uparrow\downarrow$) | Detectorul de componente conexe vede o singură pată bilobată (pentru secundă) sau o singură elipsă cu două cozi (pentru unison) și emite o singură notă. | **Descompunere prin signatura de curbură și dual-stem**: (a) Două stems antiparalele ($\uparrow$ și $\downarrow$) atașate aceleiași elipse demonstrează matematic unisonul bivoce (`two_voices_unison`); (b) Pentru secunda adiacentă, transformata distanței euclidiene are două maxime locale separate de $\Delta x \approx 1.1\cdot\text{sp}$ și $\Delta y = 0.5\cdot\text{sp}$. | `[DEMONSTRAT]` geometric pentru descompunerea bilobată prin maximele transformatei de distanță; `[IPOTEZĂ]` pe manuscrise cu cerneală inundată (falsificare dacă rata de separare a secundelor scade sub `98%` la `15 px/sp`). |
| **10** | **Schimbare de cheie sau armură în mijlocul măsurii** (ex. cheie de Fa introdusă la jumătatea măsurii, la scară $75\%$ față de cheia inițială) | OMR-ul clasic presupune că cheia și armura există doar la $x \approx 0$ (începutul sistemului) și clasifică o cheie interioară drept un acord de note suprapuse cu alterații. | **Invarianță de poziție a operatorului de cheie (`mid_staff_clef_change`)**: Orice glif cu înălțime $> 2.8\cdot\text{sp}$ și topologie spiralată/buclată (cheie de Sol, Fa, Do la scară $0.65\text{--}1.0$) actualizează funcția de mapare $y \mapsto \text{pitch}$ doar pentru $x > x_{\text{clef}}$ pe acel portativ, păstrând poziția fizică exactă $x_{\text{cm}}$. | `[IPOTEZĂ]` — Falsificare: dacă o cheie de bas la mijlocul măsurii este clasificată ca notă sau aplicată retroactiv notelor de la $x < x_{\text{clef}}$, testul e picat. |
| **11** | **Ornamente (tril `tr~~~`, mordent, grupet, fermată) vs. pete sau litere de text** | `tr` este confundat cu silaba `"tr"` din versuri; linia ondulată a trilului este confundată cu o legătură sau ștearsă. | **Contextualizare deasupra notei-ancoră**: Gliful `tr` urmat de o undă periodică cu frecvență spațială $\lambda \approx 0.6\text{--}1.0\cdot\text{sp}$ situată deasupra portativului este legat de nota de sub punctul de start ($x_0$) și se întinde exact până la $x_1$ (salvat în cm, fără trunchiere). | `[IPOTEZĂ]` — Falsificare: confuzia între `tr` deasupra portativului și o silabă de vers (care stă sub portativ în banda vocală) falsifică izolarea benzilor. |
| **12** | **Tremolo pe stem (`///`) vs. steaguri de șaisprezecimi/treizecidoimi** | Cele 3 linii oblice scurte pe stem-ul unei doimi (tremolo de treizecidoimi) sunt citite ca o treizecidoime cu cap plin sau ignorate, stricând durata măsurii cu un factor de $8\times$. | **Testul comun interior al capului notei + conservarea măsurii**: Dacă capul notei este gol (`hollow_notehead`, $\tau_{\text{interior}} \approx \tau_{\text{paper}}$) dar stem-ul poartă $k \in \{1,2,3\}$ bare oblice scurte care nu ating alt stem (sau unesc două doimi fără a însuma durata unor șaisprezecimi individuale), conservarea duratei măsurii ($\sum d_i = T_{\text{measure}}$) impune matematic interpretarea `measured_tremolo` ($2^k$ subdiviziuni pe durata doimii). | `[DEMONSTRAT]` aritmetic prin constrângerea de conservare a măsurii ($\frac{1}{2}\text{ măsură} \neq \frac{1}{32}\text{ măsură}$). |

---

## 3. Cum se rulează suita Red-Team

```bash
PYTHONPATH=src python3 -c "from pathlib import Path; from scoretwin.redteam_scenarios import run_redteam_torture_evaluation; print(run_redteam_torture_evaluation(output_path=Path('results/redteam-torture-test-2026-10-30.json'))['all_scenarios_passed'])"
```
Toate cele 6 scenarii din [`src/scoretwin/redteam_scenarios.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/redteam_scenarios.py) sunt de asemenea integrate în suita automată [`tests/test_idsr_engine.py`](file:///Users/paul_s/Gemini/scoretwin/tests/test_idsr_engine.py) (`test_redteam_torture_scenarios`).
