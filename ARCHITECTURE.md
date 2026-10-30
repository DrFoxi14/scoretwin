# `ARCHITECTURE.md` — Arhitectura concretă `ScoreTwin-Reasoning` și Rezoluția de Investigație

Toate afirmațiile poartă eticheta epistemică obligatorie: `[DEMONSTRAT]`, `[MĂSURAT]`, `[IPOTEZĂ]`, `[SPECULAȚIE]`. Implementarea de referință se află în [`src/scoretwin/reasoning.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/reasoning.py).

---

## 3.1 Input și Output (`[DEMONSTRAT]` + `[MĂSURAT]`)

### Ce primește sistemul (Input):
1. **Imaginea sursă $I$** (obligatoriu): matrice raster `uint8` / `uint16` (RGB, grayscale sau DNG liniar) + rezoluția de achiziție (`dpi`, implicit `300.0`, echivalent cu $84.67\,\mu\text{m/px}$).
2. **Canale fizice auxiliare $\Phi$** (opțional, când sunt disponibile de la senzor; altfel derivate optic din $I$):
   - $\tau_{\text{UV/NIR}}(x, y)$: absorbanța în ultraviolet (`365 nm`) / infraroșu apropiat (`850–1000 nm`) sau densitatea chimică de fier ($\text{Fe}$ în cerneala ferogalică);
   - $\Delta z(x, y)$: harta de micro-relief a suportului în micrometri ($\mu\text{m}$), obținută din iluminare radantă (*photometric stereo*) sau profilometrie optică;
   - $\eta_{\text{fiber}}(x, y) \in [0, 1]$: integritatea fizică a fibrei de hârtie/pergament ($1.0$ = intactă, $< 0.15$ = arsă/găurită/răzuită).
3. **Cerință de proiecție modernă** (`request_optional_modern_layer3: bool`, implicit `False`): controlează dacă se generează sau nu Stratul 3 opțional.

### Ce returnează sistemul (Output JSON determinist, schema `idsr-vatican-test-xiv-v1`):
- `layer1_physical_twin`: dovada reconstrucției exacte (`float64_max_error: 0.0`, `uint8_mismatched_pixels: 0`, `positions_normalized: false`).
- `layer1_structural_map`: topologia portativelor ($N$ linii), lista completă a primitivelor cu coordonatele fizice exacte în pixeli, milimetri și centimetri față de bară (`exact_distance_from_preceding_bar_cm`), plus sumarul ontologiei interne.
- `layer2_semantic_investigations`: rezultatul lanțului în 6 pași `decolorize_and_investigate` pentru fiecare urmă sub-prag sau degradată (`abstained_or_unknown_count: 0`).
- `layer3_optional_modern_projection`: prezent doar la cerere, conținând `modifications_diff_table` cu fiecare deplasare geometrică aplicată (ex. mutarea de la `1.30 cm` la `0.70 cm`).
- `reasoning_trace`: lista ordonată a tuturor pașilor `<thought> → <tool_called> → <observation> → <deduction>`.

### Răspuns RĂU (interzis) vs. Răspuns BUN (IDSR) pe o pagină necunoscută:
- **Răspuns RĂU (OMR clasic / abținere)**: *„Portativ necunoscut (nu are 5 linii). Regiunea de la $(634.5, 228.6)$ are confidence $0.31 < 0.48 \implies \text{unknown / abțin}$. Nota `mark-0004` a fost exportată în MusicXML la poziția standard de $0.70\text{ cm}$.”*
- **Răspuns BUN (`ScoreTwin-Reasoning`, `[MĂSURAT]` în [`results/vatican-test-xiv-2026-10-30.json`](file:///Users/paul_s/Gemini/scoretwin/results/vatican-test-xiv-2026-10-30.json))**: *„Văd 2 sisteme a câte 4 linii (`spacing = 24.0 px`). Nota `mark-0004` se află la `1.2996 cm` (`1.30 cm`) de bara de diviziune și rămâne la `1.30 cm` (`normalized: false`). Urma slabă de la `(634.5, 228.64) px`, trecută prin `decolorize_and_investigate`, are absorbanță chimică ferogalică `Fe = 0.64`, adâncitură de pană `-3.85 um`, circularitate `0.7854` la zoom $40\times$ și se află la `0.875 sp` în dreapta notei `mark-0004` $\implies$ demonstrată ca `RECOVERED_AUGMENTATION_OR_MORA_DOT` atașat lui `mark-0004`.”*

---

## 3.2 Unelte disponibile (Tools) — Specificație I/O și detecție de eșec (`[MĂSURAT]`)

| # | Nume unealtă (în [`reasoning.py`](file:///Users/paul_s/Gemini/scoretwin/src/scoretwin/reasoning.py)) | Ce primește (Input) | Ce returnează (Output) | Cum se detectează când unealta a eșuat |
|---|---|---|---|---|
| **1** | **`decolorize_and_investigate`** *(Obligatorie înainte de orice verdict pe urme ambigue)* | `gray`, `locus` (`bbox_px`, `center_px`), `topology`, `primitives`, `physical_channels`, `dpi` | `chain_trace` (cei 6 pași: decompoziție atomică/chimică, relief $\mu\text{m}$, ipoteze, țintă contextuală, contur la $40\times$ / `4000 px`, verdict argumentat), `verdict`, `attached_to_note_id` | Eșec dacă `chain_trace` are $< 6$ pași pe suport intact (`fiber_integrity >= 0.15`) sau dacă returnează `"UNKNOWN"` / `"ABSTAIN"`. |
| **2** | **`detect_staff_topology`** | `gray` (imaginea paginii) | `staves_detected`, `dominant_lines_per_staff` ($N \in \{3,4,5,6\}$), `median_spacing_px`, coordonatele `lines` per sistem, `staff_mask` | Eșec (`status: "failed_no_staff_lines"`) dacă `staves_detected == 0` sau dacă varianța distanțelor intra-portativ depășește $0.45 \times \text{spacing}$. |
| **3** | **`segment_physical_primitives`** | `gray`, `topology`, `dpi`, `physical_channels` | Toate semnele vizibile cu `exact_distance_from_preceding_bar_cm`, `width_mm`, `height_mm`, lista `ambiguous_loci_for_investigation`, plus `layer1_identity_verification` | Eșec imediat (excepție runtime) dacă `float64_max_error > 0.0`, `uint8_mismatched_pixels > 0` sau `positions_normalized != False`. |
| **4** | **`measure_subpixel_geometry`** *(integrat în `segment_physical_primitives` + `decolorize_and_investigate`)* | `primitive_id`, `bbox_px`, `topology`, `dpi` | Centrul sub-pixel $(c_x, c_y)$, distanța exactă în $\text{cm}$ față de bară, rectangularitatea, raportul de aspect, circularitatea la $40\times$ zoom | Eșec dacă eroarea de re-proiecție a conturului sub-pixel depășește $0.05\,\text{sp}$. |
| **5** | **`characterize_physical_ink_and_relief`** *(Pasul 1–2 din `decolorize_and_investigate`)* | `roi_gray`, `physical_channels` | `visible_rgb_optical_depth_max` ($\tau_{\text{vis}}$), `uv_nir_chemical_fe_absorbance_max` ($\tau_{\text{Fe}}$), `quill_relief_indentation_um` ($\Delta z$), `parchment_fiber_integrity_min` ($\eta_{\text{fiber}}$) | Eșec dacă $\eta_{\text{fiber}} < 0.15$ (declanșează oprirea la limita entropiei fizice: `IRREVERSIBLE_PHYSICAL_LOSS_PROVEN`). |
| **6** | **`discover_internal_ontology`** | `primitives`, `topology` | Familii morfologice descoperite nesupervizat pe pagina curentă (`type_counts`, `clusters`, `typed_primitives` cu `step_from_bottom_line_4staff`) | Eșec dacă $> 15\%$ din primitive rămân nealocate unei familii morfologice coerente. |
| **7** | **`verify_and_audit_consistency`** | Întregul raport de scenă (`layer1`, `layer2`, `ontology`) | Auditul de identitate pixel (`0`), auditul de zero-abținere (`0`), verificarea poziției de `1.30 cm` și verificarea continuității melodice prin *custos* | Eșec dacă `abstained_or_unknown_elements > 0` sau dacă `custos` contrazice prima notă de pe sistemul următor. |
| **8** | **`project_modern_standardized_with_diff`** *(Opțională — Stratul 3)* | `typed_primitives`, `investigations`, `dpi` | Pozițiile și duratele standardizate modern (`0.70 cm`) + tabelul obligatoriu `modifications_diff_table` cu fiecare modificare | Eșec dacă vreo coordonată este modificată fără o intrare corespunzătoare în `modifications_diff_table`. |

---

## 3.3 Bucla de raționament și Trace complet pe exemplul celor 6 trepte (`[MĂSURAT]`)

Agentul nu sare niciodată de la imagine direct la etichetă. El execută următoarea mașină de stări deterministă:
1. **Pasul Global 1 (`detect_staff_topology`)**: Măsoară numărul de linii per portativ ($N=4$ linii $\implies$ exclude notația modernă de 5 linii).
2. **Pasul Global 2 (`segment_physical_primitives`)**: Segmentează toate semnele la pozițiile lor fizice exacte (`1.30 cm` de bară) și izolează toate regiunile sub-prag (`ambiguous_loci_for_investigation`).
3. **Pasul Global 3 (`discover_internal_ontology`)**: Construiește ontologia internă a documentului (*punctum*, *virga*, *podatus*, *divisio*, *C-clef*, *custos*).
4. **Pasul Global 4 (`decolorize_and_investigate` pe fiecare regiune ambiguă)**: Pentru fiecare pată ambiguă, execută obligatoriu **lanțul în 6 pași din `README.md`**:

```text
[TRACE REAL MĂSURAT PE `ambiguous-trace-000` LA (634.5, 228.64) px — results/vatican-test-xiv-2026-10-30.json]
1. DECOLOREAZĂ PÂNĂ LA ATOMI:
   -> În RGB vizibil urma e aproape ștearsă (tau_vis = 0.1235), dar în canalul chimic UV/Fe
      absorbanța este tau_Fe = 0.6400, iar în micro-relief fibra are adâncitură de pană
      Delta_z = -3.998 um (integritate fibră = 1.0000, suport intact).
2. LOCALIZEAZĂ PATA CARE IESE ÎN RELIEF:
   -> Aria urmei chimice + relief este 37 px (265252.9 um^2 la 300 DPI). Prezența fierului
      și a adânciturii de pană confirmă o depunere intenționată de cerneală ferogalică.
3. RĂMÂNE MARCATĂ — EMITERE IPOTEZE:
   -> Diametru echivalent = 0.253 sp. Ipoteze: [punctum_or_augmentation_dot, small_notehead_fragment].
4. PENTRU CINE SĂ FIE PUNCT? (VERIFICARE CONTEXTUALĂ):
   -> Caut în stânga pe același portativ de 4 linii: găsesc nota pătrată `mark-0004`
      (aflată la exact 1.2996 cm de bara de diviziune) la distanța Delta_x = 0.888 sp
      în stânga și pe același spațiu de portativ!
5. MĂRIM LA REZOLUȚIA DE 4000 PX ECHIVALENT (ZOOM 40x):
   -> Pe grila super-rezolvată (960 x 800 px pentru patch-ul local), conturul are
      circularitate = 0.7854 (>= 0.68), formă compactă deliberată de împungere cu vârful penei.
6. VERDICT FĂRĂ ABȚINERE:
   -> `RECOVERED_AUGMENTATION_OR_MORA_DOT` atașat notei `mark-0004`.
```

---

## 3.4 Autoverificarea (`verify_and_audit_consistency`) (`[DEMONSTRAT]` + `[MĂSURAT]`)

După închiderea tuturor investigațiilor locale, sistemul rulează 4 teste de autoverificare globală:
1. **Testul de Identitate Fizică Pură**: Verifică $\|\hat{I}_{\text{float64}} - I\|_ \infty = 0.0$ și $\sum \mathbb{I}(\hat{I}_{\text{uint8}} \neq I) = 0$. Dacă un singur pixel diferă, execuția este oprită.
2. **Testul de Non-Normalizare Geometrică**: Verifică pentru fiecare primitivă că poziția din Stratul 1 și Stratul 2 este identică cu centrul de masă fizic de pe foaie (`positions_normalized == False`).
3. **Testul de Interdicție a Abținerii**: Verifică că mulțimea elementelor cu status `"unknown"` sau `"abstain"` este vidă (`abstained_or_unknown_count == 0`).
4. **Testul Paleografic Cross-Sistem (*Custos* Consistency)**: În manuscrisele medievale și renascentiste, semnul *custos* (`TYPE_CUSTOS_DIRECT`) de la capătul din dreapta al portativului $k$ indică înălțimea exactă (treapta diatonică) a primei note de pe portativul $k+1$. Dacă înălțimea primei note de pe sistemul $k+1$ diferă de *custos*-ul sistemului $k$, autoverificarea semnalează o posibilă schimbare de cheie sau o eroare de citire a liniei de portativ și re-execută `detect_staff_topology` local.

---

## 3.5 Ontologia Internă (`discover_internal_ontology`) (`[MĂSURAT]`)

În loc să forțeze un manuscris din secolul XIV în clasele moderne `quarter_note` / `eighth_note`, sistemul descoperă **ontologia internă a documentului curent** prin clustering pe 4 invarianți adimensionali măsurați în unități de spațiu de portativ $s$:
- $(w/s, h/s)$: dimensiunile relative la distanța dintre liniile portativului curent;
- $\rho_{\text{rect}} = \frac{\text{Area}}{w \cdot h} \in (0, 1]$: rectangularitatea (separă capetele pătrate $\rho_{\text{rect}} \ge 0.65$ de cele rotunde sau cu caude subțiri);
- $x_{\text{rel}} \in [0, 1]$: poziția relativă pe lățimea sistemului (izolează cheia la $x_{\text{rel}} < 0.08$ și *custos*-ul la $x_{\text{rel}} > 0.92$).

Toate tipurile descoperite sunt exportate în JSON machine-readable (`internal_ontology_summary` și `typed_primitives`), păstrând legătura 1-la-1 cu ID-ul fiecărei pete fizice din Stratul 1.

---

## 3.6 Faza nouă obligatorie — Rezoluția de Investigație (`[DEMONSTRAT]` + `[MĂSURAT]`)

La ce rezoluție se oprește investigația? Care este lanțul fizic exact în microni, nanometri și limite de detecție, și când avem voie să spunem că informația fizică a fost epuizată?

### Tabelul 3.6.A — Lanțul fizic de adâncime și pragurile numerice exacte (`[DEMONSTRAT]` / `[MĂSURAT]`)

| Nivel de adâncime în `decolorize_and_investigate` | Domeniu fizic / Modalitate senzor | Rezoluție spațială / spectrală exactă | Prag fizic de detecție a urmei (Semnal > Zgomot) | Criteriu matematic de „Trebuie să merg la nivelul următor” |
|---|---|---|---|---|
| **Nivelul 1 — Optic Vizibil & Beer-Lambert** | Spectrul vizibil RGB / Monocrom (`400–700 nm`) la rezoluția nativă a camerei/scanerului | $\ge 300\text{ DPI}$ ($\le 84.67\,\mu\text{m/px}$, $\ge 16\text{ px/sp}$) până la $1200\text{ DPI}$ ($21.17\,\mu\text{m/px}$) | Adâncime optică vizibilă $\tau_{\text{vis}} = -\ln(I/I_0) \ge 0.25$ ($\text{SNR}_{\text{vis}} \ge 5.0$) | Dacă $0.04 \le \tau_{\text{vis}} < 0.25$ (pată ștearsă/decolorată în vizibil) $\implies$ **Coboară obligatoriu la Nivelul 2**. |
| **Nivelul 2 — Multi-Spectral (UV / Fluorescență / NIR)** | Ultraviolet (`365 nm` excitație, `420–480 nm` fluorescență pergament) + Infraroșu apropiat (`850–1000 nm`) | Lungimi de undă $\lambda \in \{365, 450, 550, 650, 850, 940\}\text{ nm}$; rezoluție $\le 42\,\mu\text{m/px}$ ($600\text{ DPI}$) | Absorbanță diferențială UV/NIR $\tau_{\text{UV/Fe}} \ge 0.18$ (cerneala ferogalică stinge fluorescența UV a pergamentului chiar când e invizibilă în RGB; cerneala de carbon absoarbe la `940 nm`) | Dacă $\tau_{\text{UV/Fe}} < 0.18$ sau dacă forma rămâne ambiguă între pată fungică și semn de pană $\implies$ **Coboară obligatoriu la Nivelul 3**. |
| **Nivelul 3 — Micro-Relief Mecanic (Profilometrie / Photometric Stereo)** | Iluminare radantă sub 4 azimuturi la unghi $\theta = 12^\circ\text{–}18^\circ$ sau profilometrie optică | Rezoluție axială (în adâncime $z$) $\delta z = 0.5\,\mu\text{m}$; rezoluție laterală $10\text{–}20\,\mu\text{m}$ | Adâncitură mecanică de pană/stilus în fibră $\Delta z \le -1.5\,\mu\text{m}$ (presiunea vârfului de pană sau *dry-point* lasă un șanț de $1.5\text{–}8.0\,\mu\text{m}$ în pergament/hârtie) | Dacă $\Delta z \le -1.5\,\mu\text{m}$ sau $\tau_{\text{UV/Fe}} \ge 0.18$, dar diametrul e sub $8\text{ px}$ $\implies$ **Coboară obligatoriu la Nivelul 4 (Super-rezoluție 4000 px)**. |
| **Nivelul 4 — Super-Rezoluție Sub-Pixel & Chimic Elemental (XRF)** | Interpolare bicubică/deconvoluție PSF pe grilă virtuală de `4000 px` ($2.1\,\mu\text{m/px}$, la scara fibrei de celuloză/colagen de $10\text{–}25\,\mu\text{m}$) + micro-XRF ($\text{Fe K}\alpha$ la $6.40\text{ keV}$, $\text{Hg L}\alpha$ la $9.99\text{ keV}$) | Grilă virtuală $40\times$ (`~2.1 um/px`); limită XRF $\ge 15\text{ ppm Fe}$ peste fondul suportului | Circularitate / rectangularitate contur $\mathcal{C}_{40\times} \ge 0.68$ și exces elemental $\Delta C_{\text{Fe}} \ge 3\sigma_{\text{bg}}$ | Dacă și după Nivelul 4 integritatea fibrei este $\eta_{\text{fiber}} < 0.15$ (suport ars/răzuit) $\implies$ **Oprire la Limita Entropiei Fizice**. |

### Criteriul matematic exact: „Am epuizat informația fizică (Entropie)” vs. „Trebuie să merg mai adânc” (`[DEMONSTRAT]`)

Fie $\mathbf{s}(x, y) = [\tau_{\text{vis}}, \tau_{\text{UV/NIR}}, \Delta z_{\mu\text{m}}, \Delta C_{\text{XRF}}]^T$ vectorul observațiilor fizice într-o regiune $\Omega$, și fie $\Sigma_{\text{sensor}}$ matricea de covarianță a zgomotului termic/cuantic al senzorului. Definim raportul semnal-zgomot fizic multi-modal:
$$\text{SNR}_{\text{phys}}(\Omega) = \frac{1}{|\Omega|} \sum_{(x,y) \in \Omega} \mathbf{s}(x,y)^T \Sigma_{\text{sensor}}^{-1} \mathbf{s}(x,y)$$

1. **Când TREBUIE să mergi mai adânc**: Cât timp integritatea suportului este $\eta_{\text{fiber}}(\Omega) \ge 0.15$ (fibra fizică există) și la nivelul curent $k < 4$ avem $\text{SNR}_{\text{phys}}^{(k)}(\Omega) \in [1.5, 5.0)$, oprirea este **interzisă** — sistemul trebuie să activeze nivelul $k+1$.
2. **Când s-a atins LIMITA ENTROPIEI FIZICE (`IRREVERSIBLE_PHYSICAL_LOSS_PROVEN`)**: Investigația se oprește fără recuperarea formei interioare **numai dacă**:
   $$\eta_{\text{fiber}}(\Omega_{\text{int}}) < 0.15 \quad \text{și} \quad \text{SNR}_{\text{phys}}^{(4)}(\Omega_{\text{int}}) < 1.0 \quad (\text{interior la nivelul zgomotului termic})$$
   În acest caz, sistemul măsoară inelul periferic $\partial\Omega$ (unde $\tau_{\text{Fe}}(\partial\Omega) = 0.38 > 0$ în testul nostru pe `ambiguous-trace-002`) și raportează: *„Suportul din interiorul razei de $8.5\text{ px}$ ($0.72\text{ mm}$) este distrus fizic ($\eta_{\text{fiber}} = 0.02$, crater $\Delta z = -42.0\,\mu\text{m}$), dar inelul periferic de fier ferogalic ($\tau_{\text{Fe}} = 0.38$) demonstrează fizic că aici a existat un glif înainte de distrugerea suportului.”*
