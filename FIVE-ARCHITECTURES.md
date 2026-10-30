# `FIVE-ARCHITECTURES.md` — Cinci arhitecturi disjuncte pentru `ScoreTwin-Reasoning` cu Tool Use

În conformitate cu Faza 1, primele cinci secțiuni descriu 5 arhitecturi care nu împart aceleași presupuneri despre cum se cuplează raționamentul muzicologic cu fizica foii, **fără evaluare comparativă în interiorul descrierilor**. Identificarea celei mai slabe arhitecturi apare exclusiv în Secțiunea 6.

---

## Arhitectura 1 — `Small-Musicology-CoT` (SLM specializat 3B–7B antrenat supervizat pe trace-uri de investigație paleografică)

- **`[IPOTEZĂ]` Presupunerea de bază**: Raționamentul muzicologic și paleografic este o competență lingvistic-secvențială care poate fi distilată în ponderile unui model de limbaj-viziune compact (3B–7B parametri) prin fine-tuning supervizat (SFT) pe zeci de mii de lanțuri de gândire (*Chain-of-Thought*) cu apeluri JSON de unelte.
- **`[IPOTEZĂ]` Mecanismul de funcționare**:
  1. Un encoder vizual multi-rezoluție proiectează imaginea paginii și lista inițială de componente conexe în spațiul de tokeni al unui model de 3B–7B parametri.
  2. Modelul generează autoregresiv tokeni `<think>...</think>` și apeluri de funcții (`detect_staff_topology`, `decolorize_and_investigate`, `discover_internal_ontology`), primind ieșirile numerice în context.
  3. Toată cunoașterea despre notații (neume, notație pătrată, mensurală, barocă, modernă) și despre ordinea apelării uneltelor este stocată parametric în greutățile modelului antrenat pe trace-uri sintetice și istorice.
- **`[IPOTEZĂ]` Test de falsificare**: Se rulează modelul 3B–7B pe 10 pagini dintr-un sistem de notație exclus complet din setul de SFT (ex. notație dacăziană sau tabulatură de lăută cu 6 linii); dacă modelul halucinează reguli din notația pătrată pe 4 linii în $\ge 20\%$ din pașii `<think>`, presupunerea că SFT pe trace-uri generalizează în afara distribuției de antrenare este falsificată.

---

## Arhitectura 2 — `Zero-Train-VLM-Orchestrator` (Model generalist frontieră cu prompting structurat și bucle de unelte deterministe)

- **`[IPOTEZĂ]` Presupunerea de bază**: Un model multimodal generalist de frontieră (fără niciun fine-tuning specific pe muzică) posedă deja capacitatea de deducție logică, geometrică și istorică necesară, dacă este constrâns printr-o mașină de stări de prompting și un set de unelte fizice exacte care îi interzic să ghicească din pixeli.
- **`[IPOTEZĂ]` Mecanismul de funcționare**:
  1. Zero antrenare de greutăți: sistemul folosește un VLM existent apelat printr-o gramatică formală (schema JSON strictă care nu conține valoarea `"unknown"` sau `"abstain"`).
  2. La fiecare pas, orchestratorul obligă modelul să formuleze o ipoteză falsificabilă și să emită un apel de unealtă fizică (`decolorize_and_investigate`, `measure_subpixel_geometry`).
  3. Toate calculele sub-pixel, măsurătorile în centimetri (`1.30 cm`) și decompozițiile Beer-Lambert sunt executate de motorul determinist Python/C++, iar VLM-ul face doar sinteza contextuală și verificarea ipotezelor.
- **`[IPOTEZĂ]` Test de falsificare**: Se măsoară invarianța trace-ului pe 20 de rulări repetate pe aceeași pagină medievală și costul/latența per pagină cu 200 de glife; dacă varianța deciziilor între rulări depășește $0\%$ sau dacă timpul de inferență depășește $60\text{ s/pagină}$, arhitectura este falsificată pentru operare deterministă de arhivă.

---

## Arhitectura 3 — `Inverse-Physics-SelfPlay-RL` (Agent antrenat prin Reinforcement Learning și Self-Play contra motorului fizic de randare+degradare)

- **`[IPOTEZĂ]` Presupunerea de bază**: A înțelege o pagină de manuscris înseamnă a găsi secvența de acțiuni de investigație și reconstrucție care minimizează excesul MDL ($\Delta L_{\text{MDL}}$) și eroarea fizică de re-sinteză, iar această politică poate fi învățată fără nicio etichetă umană prin *self-play* între un Generator de Degradări Fizice (adversar) și un Agent Investigator.
- **`[IPOTEZĂ]` Mecanismul de funcționare**:
  1. Un generator fizic creează partituri cu topologii variate ($3\text{–}6$ linii, glife procedurale din gramatici formale) și le aplică degradări fizice controlate (decolorare ferogalică, pete de foxing, abraziuni de fibră).
  2. Agentul primește doar imaginea degradată și primește recompensă $R = -\Delta L_{\text{MDL}} - \lambda_{\text{pos}} \|\hat{x} - x_{\text{exact}}\| - \lambda_{\text{false\_loss}}$ doar atunci când recuperează prin `decolorize_and_investigate` exact starea fizică ascunsă.
  3. Politica este optimizată prin GRPO / PPO direct pe recompensa de verificare fizică, fără trace-uri scrise de oameni.
- **`[IPOTEZĂ]` Test de falsificare**: Se evaluează politica antrenată prin RL pe 10 scanări reale de pergament ale căror degradări chimice diferă de simulatorul procedural; dacă rata de recuperare scade cu $> 25\%$ față de simulator (*sim-to-real gap*), presupunerea că RL pe simulator fizic este suficientă este falsificată.

---

## Arhitectura 4 — `Training-Free-Symbolic-Ontology-Graph` (Motor pur algoritmic de inducție gramaticală, graf de constrângeri și memorie externă de arhetipuri, zero rețele neuronale)

- **`[IPOTEZĂ]` Presupunerea de bază**: Notația muzicală din orice epocă este un cod grafic bidimensional cu redundanță structurală ridicată, astfel încât descoperirea ontologiei interne a paginii și rezolvarea urmelor ambigue pot fi realizate integral prin clustering morfologic ne-supervizat, satisfacere de constrângeri (CSP) și potrivire topologică într-o memorie externă de grafuri, fără niciun model neuronal.
- **`[IPOTEZĂ]` Mecanismul de funcționare**:
  1. Uneltele fizice (`detect_staff_topology`, `segment_physical_primitives`, `decolorize_and_investigate`) extrag vectorii de invarianți fizici și geometrici pentru fiecare pată de pe pagină: $(\tau_{\text{vis}}, \tau_{\text{Fe}}, \Delta z_{\mu\text{m}}, \text{rectangularitate}, w/s, h/s, \text{circularitate}_{40\times})$.
  2. Un algoritm de clustering non-parametric (Dirichlet Process / HDBSCAN pe invarianți) descoperă automat familiile de glife ale documentului curent (ontologia internă) și construiește graful de adiacență spațială.
  3. Fiecare urmă ambiguă este rezolvată printr-un solver determinist de reguli fizice și contextuale (ex. prezență $\text{Fe} + \text{relief} + \text{notă în stânga la } \Delta x \in [0.35s, 1.45s] \implies \text{punct de augmentare/mora}$).
- **`[IPOTEZĂ]` Test de falsificare**: Se rulează solverul pe o pagină cu glife cursiv-conectate (ex. polifonie Ars Subtilior sau schiță autografă unde 80% din semne se ating în ligaturi neregulate); dacă HDBSCAN nu poate separa familiile de glife fără un prior semantic, arhitectura pur non-neuronală este falsificată.

---

## Arhitectura 5 — `Neuro-Physical-Investigator` (Arhitectură hibridă: Nucleu fizic determinist de straturi + Agent de raționament bazat pe reguli fizice verificabile și inducție de ontologie internă)

- **`[IPOTEZĂ]` Presupunerea de bază**: Niciun model neuronal nu trebuie lăsat să ghicească coordonate sau să decidă din pixeli RGB, iar niciun algoritm fix de OMR nu trebuie să presupună un vocabular închis; prin urmare, sistemul trebuie împărțit într-un **Motor Fizic Verificabil** (care garantează 100% Stratul 1 la poziții exacte și execută decompoziția optică/chimică/relief în 6 trepte) și un **Agent de Raționament paleografic** care formulează ipoteze, apelează uneltele fizice, construiește ontologia internă a documentului și verifică fiecare deducție contra legilor fizice ale suportului.
- **`[IPOTEZĂ]` Mecanismul de funcționare**:
  1. **Stratul 1 determinist** blochează din construcție identitatea pixel-cu-pixel ($\text{eroare } \text{float64} = 0.0$, $\text{eroare } \text{uint8} = 0\text{ px}$) și înregistrează fiecare urmă la distanța ei fizică exactă în centimetri față de bară (`1.30 cm` rămâne `1.30 cm`).
  2. **Agentul de Raționament (Stratul 2)** rulează bucla explicită *Observă Topologia → Segmentează Fără Vocabular → Descopera Ontologia Internă → Investighează Fizic (`decolorize_and_investigate` în 6 pași) fiecare urmă ambiguă → Autoverifică Consistența (inclusiv *custos*-ul și conservarea cernelii)*.
  3. Orice ipoteză emisă de agent este acceptată **doar** dacă trece testul numeric al uneltei fizice (absorbanță $\tau_{\text{Fe}}$, relief $\Delta z_{\mu\text{m}}$, circularitate la $40\times$ zoom, țintă contextuală la $\Delta x \in [0.35s, 1.45s]$).
- **`[IPOTEZĂ]` Test de falsificare**: Se rulează arhitectura pe o pagină din secolul XIV fără nicio etichetă; dacă sistemul produce fie și o singură etichetă `"unknown"` / `"abstain"`, sau mută poziția unei note fără cerere explicită în Stratul 3, sau eșuează să recupereze o urmă cu semnătură ferogalică/relief prezentă în canalele fizice, arhitectura este falsificată.

---

## 6. Care dintre cele 5 arhitecturi este cea mai slabă și de ce

- **`[DEMONSTRAT]` + `[MĂSURAT]` Cea mai slabă arhitectură este Arhitectura 1 (`Small-Musicology-CoT` — SLM 3B–7B antrenat supervizat pe trace-uri pre-existente).**
- **Trei motive decisive pentru care Arhitectura 1 este cea mai slabă**:
  1. **Încalcă frontal „Clauza Vatican” și principiul zero-shot pe notații necunoscute (`[DEMONSTRAT]`)**: Un model mic (3B–7B) ale cărui decizii depind de memorarea în greutăți (SFT) a unor trace-uri de antrenare își prăbușește acuratețea imediat ce întâlnește o notație în afara distribuției de antrenare (exact cum clasificatorul din `proposal.py` a scăzut de la `F1 = 0.9855` pe sintetic la `F1 = 0.0299` pe pagini reale în [`M2-DATA.md`](file:///Users/paul_s/Gemini/scoretwin/M2-DATA.md)).
  2. **Cere exact ceea ce proiectul interzice — un corpus masiv de trace-uri muzicologice etichetate înainte de a funcționa (`[DEMONSTRAT]`)**: Pentru a antrena prin SFT un model de 3B–7B fără ca acesta să halucineze, ai nevoie de mii de pagini deja adnotate de paleografi. Asta înseamnă să revii pe ușa din dos la paradigma „mai multe date / etichetați voi întâi”.
  3. **Confundă fluența verbală cu verificarea fizică (`[DEMONSTRAT]`)**: Un SLM antrenat pe textul trace-urilor învață să *imite stilul* unei explicații paleografice („pare a fi o virga din secolul XIV”), nu să verifice matematic dacă $\tau_{\text{Fe}} \ge 0.25$ și $\Delta z \le -1.5\,\mu\text{m}$. În IDSR, adevărul stă în măsurătoarea fizică a uneltei, nu în probabilitatea următorului token.
