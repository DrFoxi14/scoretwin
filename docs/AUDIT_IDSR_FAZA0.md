# IDSR — audit epistemic și plan de execuție

**Status:** raport inițial, creat în repository-ul de lucru DrFoxi14/scoretwin.
**Data auditului:** 2026-10-29.
**Sursa inspectată pentru faza 0:** DrFoxi14/IDSR, ramura master, tree SHA bf0363634d91929826d99204a6af4135a52a1937.
**Limită de reproducibilitate:** cifrele marcate mai jos drept rezultate existente au fost citite din cod și din artefactul versionat results/results.json; această sesiune nu a executat local pytest sau experiments/run_all.py. Prin urmare, „inspectat” nu înseamnă „rerulat independent”.

## Rezumat executiv (~500 de cuvinte)

[DEMONSTRAT] IDSR definește obiectivul mai precis decât un OMR clasic: nu doar să ghicească evenimente muzicale, ci să păstreze separat observația pixelilor, geometria notelor, interpretarea simbolică și ipotezele despre captură. Un graf de primitive, o funcție de randare și un reziduu fără pierderi sunt o descompunere validă a datelor. Această descompunere nu garantează însă identificabilitatea unică a fiecărei componente.

[MĂSURAT] Repository-ul fazei 0 conține cod pentru scene sintetice, randare, perturbare, rafinare numerică, proiecție simbolică, serializare MusicXML și metrici. Artefactele raportează DID 0,1236 și IoU 0,8762 pentru cazul sintetic numit „manuscris”; pentru o regravare simbolică canonică a aceleiași muzici, DID-ul raportat este 3,362 și IoU 0,2303. Aceste cifre descriu experimente construite procedural, nu un benchmark independent cu manuscrise fotografiate de utilizatori.

[MĂSURAT] Propunerea geometrică din experiment nu este produsă de un model antrenat. Scriptul perturbă scena de referință cunoscută și apoi rafinează parametrii. În consecință, rezultatul arată că optimizarea poate recupera o soluție pornind dintr-o vecinătate controlată a adevărului; nu arată că un sistem vizual poate găsi acea vecinătate dintr-o fotografie necunoscută. Nici intervalele de încredere, nici variația între documente independente nu sunt raportate.

[DEMONSTRAT] Compoziția definită în randator, ca produs al transmisiilor fiecărei primitive, este comutativă prin algebra înmulțirii. Testul cu adâncimi aleatoare arată că parametrul z nu afectează acea implementare 2D. Nu demonstrează că adâncimea fizică a cernelii poate fi măsurată dintr-o fotografie, nici că suprapunerea reală a cernelurilor se comportă conform acelui model. Adâncimea semantică folosită pentru explozia 3D este o convenție de vizualizare, nu o măsurătoare.

[MĂSURAT] Experimentul MDL păstrează pixelii exact prin codarea reziduului, dar cifra raportată drept exces de 0,00347 biți/pixel este diferența dintre entropiile empirice ale reziduurilor, nu lungimea totală MDL demonstrată a obiectului. Conform valorilor înregistrate, scena comprimată și reziduul PNG însumează 173.845 octeți, în timp ce imaginea PNG de bază are 158.336 octeți. Identitatea exactă este reală pentru această cale de codare; avantajul de compresie totală nu este demonstrat.

[IPOTEZĂ] Următorul experiment decisiv trebuie să fie o evaluare oarbă pe imagini reale, cu surse nedezvăluite autorului metodei, ground truth geometric obținut dintr-o sursă digitală înregistrată sau adnotare umană și comparație cu baseline-uri. Ca primă cohortă propun 100 de pagini tipărite fotografiate și 100 de pagini manuscrise, distribuite între autori, surse și condiții de captură; numărul este un proiect de experiment, nu o constatare.

[SPECULAȚIE] Estimez la 5% probabilitatea ca viziunea completă să îndeplinească deja criteriul „cel puțin 90% dintre manuscrisele reale” fără restrângerea domeniului. Este un prior subiectiv, justificat de absența testelor reale, de neidentificabilitatea unor variabile fizice și de faptul că propunerea neuronală nu există încă. Acesta nu este un interval statistic și trebuie actualizat după benchmark.

[DEMONSTRAT] Verdictul este **INCONCLUDENT**: nucleul de conservare și optimizare merită testat, dar nu există încă dovezi pentru performanța end-to-end, pragurile propuse, 3D fizic sau utilizarea ca produs.

## 1. Domeniul auditului și proveniență

[MĂSURAT] Lista fișierelor de pe commitul inspectat include idsr/render.py, idsr/refine.py, idsr/metrics.py, idsr/proposal.py, idsr/superposition.py, idsr/symbolic.py, idsr/musicxml.py, experiments/run_all.py, results/results.json și tests/test_idsr.py.

[MĂSURAT] README-ul descrie 11 teste și comanda de experimente A–F. Acestea sunt afirmații din documentație; nu le-am confirmat printr-o rulare locală în această sesiune.

[DEMONSTRAT] experiments/run_all.py construiește scena adevărată cu engrave(...), produce observația cu photograph(...), iar propunerea experimentală se face cu perturb_scene(...). Acest traseu începe de la ground truth disponibil programului, nu de la o fotografie arbitrară.

## 2. Re-derivarea experimentelor existente A–F

### A — „Gâtuirea simbolică”

[MĂSURAT] setup(hand, seed=3, capseed=1) creează o singură scenă pentru fiecare stil, o captură și o imagine sintetică. Pentru stilul „manuscris”, results.json raportează: DID simbolic 3,362; IoU 0,2303; eroare geometrică medie 0,3551 sp; eroare p95 0,913 sp; F1 relații 0,7742. Pentru IDSR rafinat raportează DID 0,1236; IoU 0,8762; eroare geometrică medie 0,0321 sp; p95 0,1252 sp; F1 relații 1,0000; timp de rafinare 84,2 s.

[DEMONSTRAT] Baseline-ul „OMR simbolic perfect” nu este un OMR evaluat. Scriptul calculează simbolurile din scena adevărată (project(gt)), le elimină coordonatele și regravează canonic prin engrave(strip(ref_sym)), apoi folosește captura adevărată. Acesta testează pierderea geometrică produsă de regravarea canonică, nu rata de recunoaștere a unui model.

[DEMONSTRAT] Pe stilul „gravură”, condiția este favorabilă baseline-ului: aceeași familie de gravor poate reproduce forma inițială, iar rezultatul salvat raportează IoU 1,0 și DID 0,0. Acesta este un contraexemplu la o versiune universală a tezei P1, dacă intrarea și regravarea aparțin unei clase deterministe comune.

[IPOTEZĂ] Repetarea A trebuie făcută pe cel puțin 100 de pagini-sursă independente per domeniu, nu doar pe seed-uri diferite ale aceluiași șablon. Falsificare: dacă avantajul geometriei ancorate dispare pe pagini reale înregistrate sau dacă baseline-ul cu geometrie/stil explicit atinge același scor, afirmația despre superioritatea reprezentării IDSR trebuie restrânsă.

### B — „Bazinul de atracție”

[MĂSURAT] exp_B() fixează scena sintetică și captura, apoi testează deviațiile sigma_sp 0; 0,05; 0,15; 0,30; 0,50; 0,80; 1,20. Valorile salvate pentru eroarea geometrică după rafinare sunt aproximativ 0,027; 0,027; 0,027; 0,029; 0,077; 0,241; 0,574 sp.

[DEMONSTRAT] exp_B() reinițializează generatorul aleator cu seed 11 în fiecare punct al grilei. Există o singură perturbație per valoare sigma, nu replici independente pentru estimarea unei distribuții. Grila sugerează o tranziție între 0,30 și 0,50 sp pentru această scenă și acești parametri, nu un prag universal exact.

[DEMONSTRAT] proposal.py descrie explicit perturb_scene drept substitut pentru modelul de propunere. Reușita rafinării nu validează un model de difuzie, deoarece modelul nu este antrenat și nu inferă primitivele din pixeli.

[IPOTEZĂ] Rularea necesară este un factorial cu cel puțin 30 de scene independente și 10 perturbații per scenă la fiecare nivel sigma, plus erori de captură și primitive lipsă. Falsificare: dacă la aceeași eroare inițială distribuția succesului depinde puternic de clasă, densitate, captură sau glifă, contractul trebuie definit condiționat de aceste variabile, nu cu un singur prag.

### C — Adâncimea

[MĂSURAT] exp_C() aplică cinci configurări aleatoare pentru z_sem și z_phys pe aceeași scenă sintetică și raportează o diferență maximă 2D de 3,3306690738754696e-16.

[DEMONSTRAT] În render.py, formula de compunere folosește prod(1 - ink * alpha); z_sem și z_phys nu intră în această funcție. Prin urmare, invarianta 2D la z este consecința construcției software și a aritmeticii în virgulă mobilă, nu dovadă experimentală că straturile fizice se pot despărți fără să modifice observarea reală.

[IPOTEZĂ] Pentru adâncime fizică sunt necesare mostre reale măsurate profilometric sau capturi controlate multi-view și cu lumină razantă. Falsificare: dacă estimarea adâncimii nu se corelează cu măsurătorile de referință pe un set ținut separat, modulul trebuie prezentat doar ca vizualizare semantică.

### D — MusicXML-IDSR

[MĂSURAT] exp_D() serializează o scenă sintetică, o importă înapoi și compară scena serializată, parametrii de captură și randarea. results.json raportează scene_bit_exact=true, capture_exact=true, diferență maximă de randare 0.0, 14 evenimente parsate de music21 și 12 pătrimi în total.

[DEMONSTRAT] Testul validează dus-întorsul formatului propriu pentru un exemplu și parseabilitatea de către un consumator specific. Nu demonstrează validitate semantică pentru toate combinațiile MusicXML, toate vocile, toate ornamentele sau toate aplicațiile.

[IPOTEZĂ] Setul de conformance trebuie să includă cel puțin 100 de fișiere din exemple independente și round-trip prin minimum trei implementări MusicXML. Falsificare: orice pierdere tăcută a notelor, vocilor, versurilor, accidentelor sau identificatorilor de ancoră blochează declararea formatului ca interoperabil.

### E — Superpoziție și ambiguitate

[MĂSURAT] Scena ambiguă este construită în ambiguous_scene() dintr-o partitură procedurală: un punct este plasat între două interpretări, un flag este estompat, iar un cap de notă este deplasat. Raportul rezultat dă informație mutuală aproximativ 0,9149 biți între două variabile și diferență 2D zero între ipotezele simbolice.

[DEMONSTRAT] superposition.py enumeră configurațiile posibile prin produsul cartezian al variabilelor din fiecare măsură și introduce o penalizare pentru încălcarea duratei. În experiment, măsurile sunt independente; product_state_across_measures=true nu testează cuplaje lungi sau un MPS cu bond dimension măsurat. Nu există în acest cod un benchmark care să arate că difuzia păstrează ambiguități reale.

[IPOTEZĂ] Testul de scalare trebuie să crească sistematic numărul variabilelor ambigue și al cuplajelor între măsuri și să măsoare memoria, timpul, eroarea marginalelor și pierderea de masă după trunchiere. Falsificare: dacă aproximarea pierde alternative relevante sau costul crește exponențial pe clase obișnuite, promisiunea MPS trebuie limitată la subprobleme locale.

### F — MDL și reziduu

[MĂSURAT] results.json raportează imagine PNG de 158.336 octeți, scenă comprimată de 20.014 octeți, reziduu PNG de 153.831 octeți, entropie reziduală IDSR 4,680656 biți/pixel și entropie reziduală oracol 4,677184 biți/pixel. Diferența calculată este 0,0034718 biți/pixel.

[DEMONSTRAT] În cod, entropia este calculată din histograma valorilor reziduale, nu din codul efectiv PNG. Costul efectiv al scenei plus al reziduului, folosind numerele salvate, este 20.014 + 153.831 = 173.845 octeți, adică 15.509 octeți peste imaginea PNG de 158.336 octeți (aproximativ 9,79% mai mult decât acel PNG). Așadar, reconstrucția fără pierderi este demonstrată pentru acel traseu, dar compresia totală nu este.

[DEMONSTRAT] „Oracolul” din cod cunoaște scena geometrică și captura adevărate din generatorul sintetic. El reprezintă un comparator privilegiat, nu o limită universală de compresie. Diferența de entropie a reziduului nu este o măsură completă MDL până când costul tuturor parametrilor, codurilor, side-channel-urilor, anteturilor și reziduului nu este numărat prin același codec efectiv.

[IPOTEZĂ] Reimplementarea trebuie să raporteze separat: bytes ai fișierului sursă, bytes ai scenei, bytes ai tuturor reziduurilor, bytes ai metadatelor și costul exact al tuturor alternativelor. Falsificare: dacă mărimea totală nu este mai mică decât un codec de bază sau dacă rezultatul se schimbă dramatic la un codec echivalent, nu se va afirma câștig MDL.

## 3. Auditul metricii DID

[DEMONSTRAT] idsr/metrics.py combină eroarea fotometrică, distanța Chamfer a contururilor, EMD de layout, erorile simbolice ancorate, F1-ul relațiilor și proporția primitivelor nepotrivite, cu ponderi egale. IoU este raportat separat, nu intră direct în suma DID din implementarea inspectată. Dimensiunile termenilor sunt diferite; ponderile egale sunt o convenție, nu o calibrare perceptivă demonstrată.

[DEMONSTRAT] În cazul noise_floor furnizat, termenul fotometric se trunchiază la zero când eroarea medie este sub pragul respectiv. În experimente, noise_floor este calculat comparând randarea simulatorului cu observația sintetică produsă din aceeași scenă. Acesta nu este un plafon măsurat pe senzor real.

[IPOTEZĂ] Invarianța metricii la gauge trebuie testată pe o familie de transformări care păstrează observația, inclusiv scară, omografie și îndoire, nu doar pe deplasarea de pagină compensată. Falsificare: dacă două stări cu aceeași randare au DID diferit, afirmația de invarianță se retrage.

[IPOTEZĂ] Calibrarea DID trebuie să includă o analiză de sensibilitate a pragului de potrivire, a rezoluției raster, a pragului măștii și a fiecărei ponderi. Falsificare: dacă ordinea metodelor se schimbă la mici modificări plauzibile ale ponderilor, scalarul nu poate susține afirmații comparative fără raportarea componentelor.

## 4. Contraexemple și versiuni corectate ale tezelor

### P1 — limita simbolică

[DEMONSTRAT] Contraexemplul este o clasă de partituri born-digital în care muzica, fontul, spațierea, paginarea, randatorul și captura sunt deterministe și cunoscute. Dacă imaginea este exact funcția conținutului simbolic în această clasă, atunci Cov(I | m)=0; o reprezentare simbolică suficientă, urmată de același randator, poate avea IoU 1. Rezultatul de gravură din experimentul A este o versiune procedurală a acestui caz.

[DEMONSTRAT] Forma defensabilă a P1 este condiționată: pentru o reprezentare care păstrează numai conținutul muzical m, fără stil, geometrie, captură sau un reziduu de imagine, eroarea minimă medie în pierdere pătratică este cel puțin varianța condiționată a observației dată de m, dacă decodorul primește numai m. Dacă ieșirea „simbolică” poate codifica și stilul, coordonatele sau date auxiliare, ipoteza de limitare nu mai este valabilă în aceeași formă.

[IPOTEZĂ] Experiment de falsificare: comparație pe set fix tipărit, capturat cu același proces, între MusicXML simplu, MusicXML plus coordonate/stil și scena IDSR. P1 este restrânsă dacă oricare reprezentare permisă obține aceeași eroare fără reziduu sau fără geometrie explicită.

### P4 — Beer–Lambert și 3D

[DEMONSTRAT] Produsul transmisiilor este comutativ prin definiția matematică a randatorului. Aceasta nu dovedește că fizica tuturor materialelor muzicale este comutativă.

[DEMONSTRAT] Guașa opacă, corectorul alb, creionul cu presiune variabilă, cernelurile care se amestecă, reflexiile lucioase și relieful peniței pot încălca modelul unui strat substractiv simplu. O singură imagine RGB nu separă în general unic albedo-ul, iluminarea, translucența, textura hârtiei și grosimea cernelii.

[IPOTEZĂ] Experiment: 30 de mostre de fiecare material relevant, capturate cu expunere/iluminare controlată, câte o vedere frontală și mai multe iluminări razante; comparație a reziduului cu modelul Beer–Lambert și cu un model alternativ de compunere. Falsificare: dacă abaterea sistematică este mai mare decât zgomotul măsurat și afectează reconstrucția, P4 se păstrează doar ca proprietate a randatorului sintetic.

### Pragul de rafinare 0,3–0,5 sp

[DEMONSTRAT] Datele existente arată valori favorabile până la 0,30 sp și degradare la 0,50 sp pentru o scenă și o captură, cu un singur eșantion per nivel. Nu demonstrează că pragul real este 0,30, 0,40 sau 0,20 sp.

[IPOTEZĂ] Se estimează curba de succes versus eroarea inițială pe mai multe scene, densități, capturi, primitive și stări de lipsă/duplicare. Falsificare: dacă pragul variațional are intervale largi sau depinde de tipul primitivului, nu se va păstra un prag global.

### Superpoziția

[DEMONSTRAT] Informația mutuală a experimentului E este determinată de distribuțiile locale programate și de factorul pentru durata măsurii. Este o demonstrație a mecanicii de inferență din acel exemplu, nu o măsurătoare a ambiguității muzicii în general.

[IPOTEZĂ] Testul de scalare trebuie să compare probabilitățile cuplajelor și să crească sistematic numărul variabilelor ambigue și al legăturilor între măsuri, apoi să măsoare memoria, timpul, eroarea marginalelor și pierderea de masă după trunchiere. Falsificare: dacă aproximarea pierde alternative relevante sau costul crește exponențial pe clase obișnuite, promisiunea MPS trebuie limitată la subprobleme locale.

## 5. Cele cinci riscuri de generalizare prioritare

[IPOTEZĂ] **Risc 1 — decalajul de generare:** fontul și gravorul procedural nu reproduc variația caligrafică reală. Experiment: 50 de partituri reale de la autori care nu apar în datele de dezvoltare, evaluate fără ajustare per document; falsificare dacă eroarea geometrică p95 depășește 0,15 sp sau symbol F1 scade sub pragul Tier B.

[IPOTEZĂ] **Risc 2 — decalajul de captură/ISP:** blur, sharpening, HDR, JPEG și lumină neuniformă pot arăta diferit de simulator. Experiment: 50 de pagini, trei familii de telefoane, condiții controlate plus lumină dificilă, originale arhivate; falsificare dacă modelul raportează încredere mare pentru pagini care depășesc pragurile DID.

[IPOTEZĂ] **Risc 3 — densitate și suprapunere:** acordurile, vocile, beam-urile cross-staff și liniile suplimentare cresc suprapunerea primitivelor. Experiment: 50 de sisteme cu densitate mare, cu adevăr digital înregistrat; falsificare dacă recall-ul primitivelor sau F1-ul relațiilor cade mai mult de 5 puncte procentuale față de paginile rare.

[IPOTEZĂ] **Risc 4 — notație deschisă și mâini necunoscute:** neume, ornamente și simboluri rare pot fi clasificate greșit în loc să fie marcate necunoscute. Experiment: 50 de pagini cu clase de simboluri ținute complet în afara setului de dezvoltare; falsificare dacă modelul le etichetează frecvent cu încredere mare ca simbol cunoscut.

[IPOTEZĂ] **Risc 5 — colapsarea metricii:** un reziduu mare sau o calibrare prea flexibilă poate obține o reconstrucție bună fără înțelegere, iar metricile de măști pot ascunde erori locale. Experiment: evaluare adversarială pe 100 de pagini, cu ablație de reziduu, blur controlat, contururi îngroșate și manipularea zgomotului declarat; falsificare dacă scorul se îmbunătățește fără îmbunătățire în evaluarea umană oarbă.

## 6. Probabilitate subiectivă și standard investițional

[SPECULAȚIE] Estimez probabilitatea ca întregul produs să atingă ≥90% succes pe manuscrise reale, în sensul cumulativ al identității imaginii, MusicXML corect și scenă 3D semnificativă, la 5%. Estimarea pornește de la lipsa oricărui test real în artefactele inspectate, absența propunerii neuronale, ambiguitățile neidentificabile dintr-o singură imagine și nevalidarea DID. Experiment de actualizare: benchmark oarbă pe minimum 100 de manuscrise reale independente, cu intervale de încredere și definiție de succes înregistrată înainte de evaluare.

[IPOTEZĂ] Înaintea unei investiții de 10 milioane USD, aș cere un benchmark deținut de un evaluator independent, minimum 200 de pagini reale din autori și capturi care nu au apărut în dezvoltare, comparații cu baseline-uri de OMR plus conservare raster și adnotare simbolică independentă. Pragul de go este ca limita inferioară a intervalului de încredere pentru fiecare metrică critică să treacă pragurile Tier B pe subsetul declarat, fără falsificări majore de metrică.

## 7. Red team: unde are dreptate adversarul

[IPOTEZĂ] **Competitor cu 100 milioane USD:** ar combina un detector OMR bun, un segmentator de măști, recuperare vectorială pentru fonturi cunoscute, rasterul original păstrat și un editor de corectare umană. Poate atinge suficientă valoare pentru tipar fără difuzie conjugată sau MPS. Falsificare: baseline-ul trebuie implementat și măsurat pe aceeași cohortă înainte de a revendica avantajul arhitecturii IDSR.

[IPOTEZĂ] **Reviewer NeurIPS/ICLR:** ar respinge o lucrare care folosește numai date sintetice, un ground truth cunoscut pentru inițializarea optimizării, o metrică propusă de autor necalibrată uman și o afirmație despre fizică bazată pe un randator artificial. Falsificare: studiu independent preînregistrat, baseline-uri, ablații și raportarea eșecurilor pe distribuții externe.

[SPECULAȚIE] **Cercetător peste zece ani:** ar putea considera naivă ideea că o singură stare latentă parsimoniosă trebuie să explice simultan toate scările, stilurile, procesele de imprimare și toate convențiile de notație. Poate că soluția practică va fi un model hibrid cu reprezentări diferite, legate de proveniență și incertitudine, nu un latent unic compact.

## 8. Harta imposibilității oneste

[DEMONSTRAT] **Zgomotul de senzor:** nu există o funcție universală IoU(SNR) fără ipoteze despre grosimea liniei, contrast, PSF, rezoluție și pragul de segmentare. Într-un model Gaussian simplificat, două ipoteze de imagine cu medii μ0 și μ1 și zgomot independent de varianță σ² au divergență KL ||μ1−μ0||²/(2σ²). SNR singur nu specifică această normă geometrică; două obiecte cu același SNR pot avea IoU-uri fundamental diferite.

[DEMONSTRAT] **Cerneală ascunsă:** conținutul complet acoperit de ștampilă, corectură opacă sau ruptură nu este identificabil din acea imagine dacă nu produce semnal observabil. Mai mult compute nu recuperează informație fizică absentă.

[DEMONSTRAT] **Gauge:** dacă o transformare a coordonatelor paginii este compensată de inversa sa în transformarea capturii, observația rămâne aceeași. Fără un reper extern sau o convenție de fixare, parametrii separați nu sunt unici. Liniile portativului sunt un reper util, dar nu există pe toate notațiile și pot fi ascunse sau deformate.

[DEMONSTRAT] **Adâncime semantică vs. fizică:** ordinea „portativ în spate, notă în față” poate fi aleasă pentru explorare; nu rezultă dintr-o singură proiecție 2D. Adâncimea fizică necesită semnale suplimentare și un model validat.

[DEMONSTRAT] **Notație necunoscută:** din pixelii observați se poate conserva forma unei glife necunoscute, însă numele și sensul ei nu sunt identificabile doar din formă dacă lipsesc convențiile și contextul. Politica sigură este mască + abținere, cu eventuala interpretare păstrată ca ipoteză.

[DEMONSTRAT] **Eroare de copist:** „identic cu ce este scris” și „muzical corect” sunt obiective distincte. Sistemul trebuie să păstreze varianta diplomatică și, separat, eventualele corecturi propuse, fără să suprascrie observația.

[IPOTEZĂ] **Pagini multiple:** deriva de gauge va fi măsurată prin ancore cross-page și parametri de pagină partajați. Falsificare: dacă incertitudinea de poziție crește sistematic cu numărul paginii sau relațiile de continuitate devin incorecte, sistemul trebuie să refacă înregistrarea globală ori să ceară o ancoră.

[IPOTEZĂ] **Partituri orchestrale:** evaluarea trebuie stratificată după numărul de portative, densitate și suprapunere. Falsificare: dacă erorile cresc accelerat cu numărul de portative sau latența depășește bugetul produsului, Tier C trebuie amânat ori procesat sistem cu sistem.

## 9. Captura activă: ce observabilitate se poate câștiga

[DEMONSTRAT] **RAW/DNG:** accesul la date mai apropiate de măsurarea senzorului elimină o parte din incertitudinea ISP, dacă datele sunt într-adevăr brute și procesarea este cunoscută. Nu recuperează frecvențe spațiale eliminate de optică sau de sampling.

[DEMONSTRAT] **Burst:** în ipoteza cadrelor aliniate cu zgomot independent de medie zero și aceeași scenă, media a n cadre reduce varianța zgomotului la σ²/n; deviația standard scade cu 1/√n. Aceasta nu se aplică direct la motion blur, mișcare a hârtiei, schimbare de expunere sau sharpening corelat.

[DEMONSTRAT] **Iluminare razantă din direcții diferite:** furnizează constrângeri suplimentare asupra normalelor de suprafață numai sub modelul de iluminare folosit; albedo-ul, umbrele și materialele speculare pot rămâne confundate.

[DEMONSTRAT] **Multi-view:** punctele corespunzătoare între vederi pot constrânge geometria prin triangulație dacă intrinsecii camerei și mișcarea sunt suficient de cunoscute. O singură vedere nu oferă această triangulație.

[IPOTEZĂ] Ghidajul de captură va fi păstrat numai dacă un test utilizator randomizat arată îmbunătățire semnificativă a geometriei sau reducerea abținerilor fără o rată de abandon UX inacceptabilă. Falsificare: dacă captura ghidată nu depășește fluxul simplu în testul preînregistrat, se elimină din v1.

## 10. Arhitectura propusă după audit

[IPOTEZĂ] **M0 — encoder vizual multi-scală:** furnizează propuneri și hărți de incertitudine; este comparat cu un detector simplu și un encoder preantrenat. Falsificare: dacă ablația nu îmbunătățește recall-ul pe primitive necunoscute sau timpul, se elimină componenta complexă.

[IPOTEZĂ] **M1 — captura explicită:** estimează homografia, deformarea paginii, iluminarea și PSF cu intervale de incertitudine. Se detectează eșecul prin reziduu structurat și matricea Hessian/Jacobian prost condiționată. Falsificare: perechi cu geometrie digitală cunoscută și capturi reale.

[IPOTEZĂ] **M2 — graf de propuneri simbolic-geometrice:** propune clase, coordonate, relații și „necunoscut”; nu impune decizia finală. Difuzia este opțională până când bate un baseline mai simplu. Falsificare: test comparativ sub aceeași limită de compute.

[IPOTEZĂ] **M3 — randare diferențiabilă și rafinare:** optimizează doar în regiuni de încredere și folosește căutare globală când primitiva este în afara bazinului local. Eșecul este detectat prin scăderea DID în timp ce erorile de relație cresc, sau prin Jacobian singular. Falsificare: perturbări sistematice de poziție, scală, rotație și captură pe scene independente.

[IPOTEZĂ] **M4 — reziduu fără pierderi:** conservă sursa exactă și se codifică împreună cu toate metadatele; nu este numit „înțelegere”. Falsificare: hash exact după round-trip și contabilitatea efectivă a tuturor octeților.

[IPOTEZĂ] **M5 — incertitudine:** păstrează alternative simbolice și relații, dar nu folosește terminologia de calcul cuantic ca afirmație fizică. Pentru început, distribuții explicite și grafuri de factori sunt suficiente; MPS se introduce numai după o măsurare a complexității. Falsificare: calibrare pe exemple ambigue etichetate independent.

[IPOTEZĂ] **M6 — 3D:** în v1 folosește z_sem numai pentru separare editorială de straturi. Eticheta UI trebuie să spună „exploded view semantic”, nu „adâncime fizică măsurată”, dacă nu există capturi și validare 3D. Falsificare: comparație cu măsurători de relief și geometrie multi-view.

[IPOTEZĂ] **Format:** graf nativ JSON cu ID-uri stabile și proveniență ca sursă de adevăr; MusicXML exportat ca proiecție compatibilă. Falsificare: round-trip prin implementări independente și verificarea conservării identității ancorate.

## 11. Antrenare și date

[IPOTEZĂ] Curriculumul începe cu gravor sintetic și adevăr geometric exact, dar split-ul se face după font, șablon și stil, nu aleator pe patch-uri. După validarea sintetică, modelul se testează pe perechi reale înregistrate; pseudo-etichetele sunt acceptate numai când randarea și verificarea independentă trec pragurile declarate.

[IPOTEZĂ] În loc să promit un exponent de scalare fără date, se măsoară performanța față de numărul de pagini unice, numărul de familii de stil, numărul de capturi și compute-ul, fiecare variabilă variată separat. Se potrivește o curbă error(N)=aN^-b+c doar dacă datele o susțin și se raportează intervalul pentru b și asimptota c. Falsificare: dacă extrapolarea nu prezice un set ținut separat, nu se folosește pentru planificarea compute-ului.

## 12. Evaluare și măsurarea metricii

[IPOTEZĂ] **Setul oarbă minim propus:** 100 pagini tipărite fotografiate și 100 pagini manuscrise; sursele, autorii și dispozitivele sunt separați de orice dezvoltare. Paginile born-digital sunt înregistrate cu sursa vectorială pentru ground truth geometric; manuscrisele fără sursă digitală primesc adnotare dublă și adjudicare. Acesta este un plan de eșantionare, nu un set deja colectat.

[IPOTEZĂ] **Baseline-uri obligatorii:** MusicXML de la OMR disponibil, detector/segmentator plus raster original, regravare simbolică, rafinare fără model de propunere și ablații ale capturii și reziduului. Fiecare primește aceleași intrări și buget comparabil; fiecare model folosește seed-uri versionate.

[IPOTEZĂ] **Studiu uman:** minimum 30 de muzicieni pentru o evaluare inițială oarbă, cu fragmente sursă și rezultate randomizate, task de potrivire/identitate și rating de încredere. Se raportează acordul inter-evaluator și corelația dintre DID și judecată; targetul preînregistrat poate fi kappa ≥0,70 și Spearman ρ ≥0,60, dar acestea sunt ținte de proiect, nu valori obținute.

[IPOTEZĂ] **Atacuri asupra DID:** creșterea pragului de zgomot declarat, estomparea ambelor imagini, îngroșarea contururilor și adăugarea de primitive duplicate pot schimba scorul fără să îmbunătățească fidelitatea reală. Apărarea este raportarea vectorului componentelor, numărarea tuturor fișierelor și o evaluare umană oarbă, plus teste de stres prestabilite.

[IPOTEZĂ] **Atacuri asupra MDL:** mutarea informației în alt fișier sau alegerea unui oracol slab face reziduul să pară mai ieftin. Apărarea este un singur artefact autocontent și măsurarea exactă a tuturor byte-urilor, a aceleiași compresii și a baseline-ului brut. Falsificare: dacă rezultatul depinde de contabilitatea incompletă, revendicarea MDL se retrage.

[IPOTEZĂ] **Distribuție în afara antrenării:** sunt excluse complet cel puțin trei familii de test: altă familie de font/gravor, alt tip de telefon/ISP și un stil real de manuscris necunoscut. Administratorul setului de test trebuie să fie diferit de autorul modelului, iar identitățile se țin secrete până la înghețarea codului. În afara distribuției, produsul trebuie să marcheze „necunoscut/încredere insuficientă”, nu să inventeze o etichetă.

## 13. Tier-uri și porți de produs

[IPOTEZĂ] **Tier A — v1, tipar standard, pagină unică, condiții bune:** ținte de acceptare: event F1 ≥0,98; precizie pitch ≥0,99; eroare geometrică p95 ≤0,10 sp; DID publicat împreună cu toate componentele; latență server ≤10 s/pagină pentru rezoluția benchmark-ului și cost operațional ≤0,05 EUR/pagină. Acestea sunt praguri de proiect, nu rezultate măsurate. Falsificare: setul oarbă nu atinge pragurile cu limitele intervalelor preînregistrate.

[IPOTEZĂ] **Tier B — manuscris și degradare moderată:** ținte: event F1 ≥0,95 pe regiunile acceptate; acoperire automată ≥0,70 cu abținere în rest; eroare geometrică p95 ≤0,15 sp; latență ≤30 s/pagină. Falsificare: măsurare pe autori necunoscuți și capturi noi; falsele certitudini sunt penalizate separat.

[IPOTEZĂ] **Tier C — multi-page, orchestrație densă, notație istorică/avangardă:** nu se promite automat un singur prag universal până când există seturi separate pentru fiecare subdomeniu. V2 trebuie să raporteze per clasă simbolică, număr de portative, densitate și pagină. Falsificare: teste pe subdomenii necunoscute și măsurarea derivei între pagini.

[IPOTEZĂ] Dacă doar Tier A este livrabil, povestea de produs este „digitizare cu păstrarea imaginii-sursă, editare și export MusicXML pentru partituri tipărite în condiții bune”, nu „digital twin perfect pentru orice manuscris”. Investitorului i se prezintă tracțiunea și metricile Tier A, nu se extrapolează la manuscrise fără rezultate.

## 14. Complicații de inginerie care trebuie urmărite

[IPOTEZĂ] **Fizica imaginii:** absorbție, împrăștiere, luciu, umbre, iluminare colorată, Bayer, zgomot Poisson–Gauss, rolling shutter, motion blur, JPEG, sharpening, tone mapping, denoise și HDR trebuie tratate drept parametri sau abateri de model. Detectarea se face prin reziduu structurat și capturi controlate; falsificare prin compararea modelului de captură cu RAW și cu sursa de referință.

[IPOTEZĂ] **Geometria paginii:** perspectivă, pliuri, curbură la cotor, pagini rupte, moiré și fotografii de pe ecran se testează ca subseturi separate. Dacă parametrii camerei și deformarea nu sunt identificabili, produsul cere recaptură sau scade încrederea.

[IPOTEZĂ] **Straturi fizice și adnotări:** transparența verso, pete, ștampile, corecturi lipite, adnotări, cerneluri multiple și mâini diferite nu se elimină ca zgomot în mod automat. Fiecare strat detectat trebuie păstrat ca mască/proveniență sau expus ca necunoscut.

[IPOTEZĂ] **Notație:** acorduri, voci, beam-uri cross-staff, grace notes, tuplets, tremolo, ornamente, accidentals de precauție, schimbări de cheie/metru, ossia, tabulatură, bas cifrat, notație mensurală, neume, grafică de avangardă, versuri multiple, diacritice și melisme sunt clase separate de evaluare, nu doar o singură etichetă „manuscris”. Falsificare: o clasă intră în Tier A/B numai după scoruri publice pentru acea clasă.

[IPOTEZĂ] **Interfață și produs:** editarea unei note trebuie să schimbe local doar primitivele și relațiile afectate, apoi să explice dacă a modificat MusicXML, geometria sau reziduul. Falsificare: suite de regresie cu hash-uri ale regiunilor neschimbate și diff al grafului.

[IPOTEZĂ] **Confidențialitate și drepturi:** procesarea locală trebuie să fie opțiunea implicită pentru documente private când performanța permite; upload-ul server trebuie explicat și separat. Trebuie stabilite proveniența sursei, politica de retenție și permisiunile înainte de ingestia arhivelor private sau protejate prin copyright. Falsificare: audit legal și de securitate înainte de beta publică.

## 15. Faze, porți și livrabile de execuție

[IPOTEZĂ] **Gate 0 — audit reproductibil:** blochează un commit; rulează testele și A–F într-un mediu curat; salvează hardware, dependențe, seed-uri, hash-uri și output-uri. GO numai dacă alt operator poate reproduce rezultatele în limite numerice declarate. NO-GO dacă rezultatele salvate nu corespund codului.

[IPOTEZĂ] **Gate 1 — benchmark real și metrică:** colectează perechi înregistrate și capturi reale, validează DID împotriva evaluatorilor și atacurilor adverse, apoi compară baseline-urile. GO numai dacă metricile prezic fidelitatea percepută și nu pot fi „jucate” prin reziduu sau praguri.

[IPOTEZĂ] **Gate 2 — Tier A end-to-end:** construiește ingestie, rectificare, graf de primitive, abținere, render overlay, editare și MusicXML. Nu se adaugă difuzie sau MPS înainte de un baseline funcțional care poate fi măsurat. GO numai dacă trece Tier A și testele de round-trip.

[IPOTEZĂ] **Gate 3 — transfer către manuscris:** adaugă auto-supervizare, variație procedurală și pseudo-etichete confirmate prin randare. GO numai dacă evaluarea oarbă Tier B trece pragurile; altfel păstrează manuscrisul ca mod asistat de utilizator.

[IPOTEZĂ] **Gate 4 — captură activă și 3D:** RAW, burst și lumină razantă se adaugă pe rând, măsurând câștigul de calitate versus costul UX. z_phys se lansează numai dacă există ground truth fizic; z_sem poate exista ca mod vizual separat.

[IPOTEZĂ] **Gate 5 — densitate/istoric:** partituri orchestrale, multi-page și notații istorice intră numai cu benchmark și buget de memorie/latență pe subdomeniu. Dacă apare explozie de stări, se limitează explicit acoperirea și se păstrează alternativele relevante cu abținere.

## 16. Kill criteria și ce tăiem pentru o livrare reală în șase luni

[IPOTEZĂ] **Kill 1:** dacă pe 100 de pagini manuscrise reale ținute deoparte, DID depășește 0,5 în peste 40% din cazuri sau corelația DID–evaluare umană este sub pragul preînregistrat, se abandonează teza că DID-ul actual este metrică de produs.

[IPOTEZĂ] **Kill 2:** dacă rafinarea dintr-o propunere produsă doar din imagine nu recuperează geometria în ≥90% din paginile Tier A, chiar dacă rafinarea din geometrie perturbată reușește, se abandonează presupunerea că rafinarea actuală poate fi nucleul pipeline-ului end-to-end.

[IPOTEZĂ] **Kill 3:** dacă MDL total (toți octeții incluși) este mai mare decât imaginea-sursă plus exportul standard fără a oferi o valoare simbolică/geometrică demonstrabilă, nu se vinde metoda ca un compresor și se păstrează conservarea raster ca arhivă fidelă.

[IPOTEZĂ] **Kill 4:** dacă modelul produce interpretări necunoscute cu încredere mare pe testele OOD, se oprește automatizarea pentru acele clase până la implementarea și calibrarea abținerii.

[IPOTEZĂ] **Kill 5:** dacă z_phys nu se corelează cu măsurători externe de relief, se elimină afirmația de profunzime fizică și se păstrează doar straturile semantice vizualizate.

[IPOTEZĂ] Pentru șase luni, se taie mai întâi: difuzia conjugată dacă nu bate baseline-ul, MPS, pretinsul z_phys, notațiile istorice, capturile multi-view și suportul orchestral. Se păstrează: Tier A, rasterul-sursă fără pierderi, rectificare, primitive explicabile, abținere, overlay, editare locală și export MusicXML.

## 17. Auto-critică meta

[DEMONSTRAT] **Punct orb al brief-ului 1:** „replică pixel cu pixel până la zgomotul senzorului” și „reziduu fără pierderi” sunt obiective diferite. Reziduul garantează reconstrucția datelor observate, dar poate ascunde lipsa explicației structurale; MDL trebuie să taxeze totul, nu doar entropia reziduului.

[DEMONSTRAT] **Punct orb al brief-ului 2:** adâncimea 3D semantică și adâncimea fizică nu trebuie reunite într-o singură afirmație de produs. Una este proiectată pentru explorare, cealaltă este o inferență fizică ce necesită observații suplimentare.

[DEMONSTRAT] **Punct orb al brief-ului 3:** un singur scalar DID cu ponderi egale amestecă unități și obiective. Componentele trebuie validate separat și agregate numai după calibrare față de sarcina umană.

[DEMONSTRAT] Cea mai solidă componentă actuală este conservarea fără pierderi a pixelilor observabili combinată cu un graf de primitive verificabil, în limitele codului existent.

[SPECULAȚIE] Cea mai puțin sigură componentă este transferul automat de la stiluri sintetice la manuscrise reale, în special pentru notație neparametrică și captură alterată de ISP.

[SPECULAȚIE] Cea mai grea propunere este inferența simultană și calibrată a geometriei, simbolurilor, capturii și ambiguităților pe manuscrise fără ground truth digital.

[IPOTEZĂ] Cu un an și 1 milion USD, aș finanța mai întâi evaluatorul independent, cohorta de date, metricile adversariale și Tier A complet; nu aș începe cu difuzia sau cu 3D fizic. Falsificare: dacă un baseline simplu depășește IDSR la aceeași calitate și cost pe setul oarbă, arhitectura trebuie simplificată.

[IPOTEZĂ] Aș abandona ideea unui latent unic dacă trei revizuiri independente ale metricii și două evaluări oarbe ar arăta că o reprezentare hibridă mai simplă oferă aceeași fidelitate, aceleași abțineri și un cost semnificativ mai mic.

## 18. Referințe inspectate

- [Repository faza 0 — DrFoxi14/IDSR](https://github.com/DrFoxi14/IDSR)
- [Arhitectura](https://github.com/DrFoxi14/IDSR/blob/master/docs/ARHITECTURA.md)
- [Runner experimente A–F](https://github.com/DrFoxi14/IDSR/blob/master/experiments/run_all.py)
- [Rezultate salvate](https://github.com/DrFoxi14/IDSR/blob/master/results/results.json)
- [Metrică DID](https://github.com/DrFoxi14/IDSR/blob/master/idsr/metrics.py)
- [Rafinare](https://github.com/DrFoxi14/IDSR/blob/master/idsr/refine.py)
- [Propunere geometrică folosită în experiment](https://github.com/DrFoxi14/IDSR/blob/master/idsr/proposal.py)
- [Superpoziție](https://github.com/DrFoxi14/IDSR/blob/master/idsr/superposition.py)
- [Teste](https://github.com/DrFoxi14/IDSR/blob/master/tests/test_idsr.py)

**Regulă de continuare:** niciun prag sau rezultat din acest raport nu devine „măsurat pe real” până când există fișierele sursă de test, hash-urile lor, codul de evaluare, logurile de rulare și raportul statistic complet.
