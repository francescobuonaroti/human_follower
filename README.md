# Robot inseguitore di persone

Un robot su ruote che **riconosce una persona con l'intelligenza artificiale e la segue**, fermandosi quando le arriva vicino.

Il progetto è pensato per un corso sull'intelligenza artificiale: tutta la parte tecnica (camera, rete neurale, motori, video) è chiusa in una "scatola nera", e gli studenti scrivono solo **l'algoritmo che decide come muoversi**.

Basato su *AI Robot – Human Following* di Jitesh Saini ([github.com/jiteshsaini](https://github.com/jiteshsaini), [helloworld.co.in](https://helloworld.co.in)).

---

## Indice

1. [Cosa fa il robot](#1-cosa-fa-il-robot)
2. [Componenti](#2-componenti)
3. [I file del progetto](#3-i-file-del-progetto)
4. [Collegamenti: Raspberry Pi → L298N](#4-collegamenti-raspberry-pi--l298n)
5. [Installazione del software](#5-installazione-del-software)
6. [Prima accensione: le prove](#6-prima-accensione-le-prove)
7. [Avviare il robot](#7-avviare-il-robot)
8. [Come ragiona il robot](#8-come-ragiona-il-robot)
9. [I parametri da regolare](#9-i-parametri-da-regolare)
10. [Problemi e soluzioni](#10-problemi-e-soluzioni)

---

## 1. Cosa fa il robot

Il robot ripete continuamente, circa 10 volte al secondo, tre passi:

```
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │  PERCEPIRE   │ ───► │   DECIDERE   │ ───► │    AGIRE     │
   │              │      │              │      │              │
   │ la camera    │      │ dov'è la     │      │ i motori     │
   │ scatta una   │      │ persona?     │      │ muovono le   │
   │ foto, la     │      │ quanto è     │      │ ruote alla   │
   │ rete neurale │      │ vicina?      │      │ velocità     │
   │ trova la     │      │ → velocità   │      │ scelta       │
   │ persona      │      │   delle ruote│      │              │
   └──────────────┘      └──────────────┘      └──────────────┘
          ▲                                           │
          └───────────────── e si ricomincia ─────────┘
```

1. **Percepire.** La camera scatta una foto. Una **rete neurale** (MobileNet SSD, addestrata a riconoscere 90 tipi di oggetti) trova le persone nell'immagine e le racchiude in un riquadro.
2. **Decidere.** L'algoritmo guarda dove si trova il riquadro della persona:
   - se è **a sinistra** dell'immagine, il robot curva a sinistra;
   - se è **a destra**, curva a destra;
   - se è **al centro**, va dritto;
   - se la persona **riempie quasi tutta l'immagine**, è arrivato e si ferma;
   - se **non vede nessuno**, si ferma.
3. **Agire.** Il robot non ha volante: ha due ruote motrici, una per lato, come un carro armato. Per curvare fa girare una ruota più veloce dell'altra.

Tutto quello che il robot "vede" e "pensa" si può guardare dal vivo da un browser, alla pagina `http://<ip-del-robot>:2204/`.

---

## 2. Componenti

| Componente | Note |
|---|---|
| Raspberry Pi 5 | con Raspberry Pi OS Bookworm o Trixie |
| Camera Module 3 (o altra camera CSI) | collegata al connettore della camera |
| Driver motori **L298N** | un solo modulo pilota entrambi i motori |
| 2 motori DC con riduttore e ruote | uno a sinistra, uno a destra |
| Batteria per i motori | per esempio 2S Li-ion (7,4 V) o 6 stilo (9 V), **massimo 12 V** |
| Alimentazione del Raspberry | powerbank USB-C o regolatore 5 V / 5 A, **separata** da quella dei motori |
| Cavetti jumper femmina-femmina | 7 fili tra Raspberry e L298N |
| Coral USB Accelerator (facoltativo) | rende la rete neurale circa 4 volte più veloce |

---

## 3. I file del progetto

Tutti nella cartella `~/robotics-level-4/earthrover/human_following/`:

| File | A cosa serve | Chi lo usa |
|---|---|---|
| `inseguitore.py` | **l'algoritmo del robot**, ben commentato | gli studenti lo leggono e lo modificano |
| `robot_toolkit.py` | la "scatola nera": camera, rete neurale, motori, video, log | non si tocca (si dà per buono) |
| `prova_camera.py` | prove dell'hardware: camera, rete neurale, motori | chi prepara il robot |
| `install.txt` | elenco delle librerie e comandi di installazione | chi prepara il robot |
| `README.md` | questo documento | tutti |

Il toolkit è **autonomo**: non ha bisogno di `common.py` né di `util.py` del progetto originale. Usa però gli stessi pin di `util.py`, quindi il pannello di controllo originale continua a funzionare con lo stesso cablaggio.

Servono inoltre, già presenti se il robot è stato installato con lo script originale:
- il modello e le etichette in `/var/www/html/all_models/`;
- `earthrover/config.txt` (facoltativo), che contiene l'orientamento della camera scelto nel pannello di controllo.

---

## 4. Collegamenti: Raspberry Pi → L298N

> ⚠️ **Fai tutti i collegamenti a robot SPENTO**: batteria dei motori scollegata e Raspberry spento.

### 4.1 Come è fatto il L298N

Disegno schematico: sul modulo reale le scritte sono le stesse, ma la disposizione può cambiare.

```
                 ┌─────────────────────────────────┐
   MOTORE        │  OUT1                     OUT4  │        MOTORE
   SINISTRO ═════│  OUT2        L298N        OUT3  │═════   DESTRO
                 │                                 │
                 │  [ENA] IN1 IN2 IN3 IN4 [ENB]    │   ← pin di comando
                 │                                 │
                 │  +12V    GND    +5V             │   ← morsetti di alimentazione
                 └─────────────────────────────────┘
```

- **OUT1 / OUT2**: motore A, la ruota **sinistra**.
- **OUT3 / OUT4**: motore B, la ruota **destra**.
- **IN1 / IN2**: comandano il motore A. **IN3 / IN4**: comandano il motore B.
- **ENA / ENB**: abilitano i due motori. Di serie hanno sopra un **ponticello** (jumper).
- **+12V**: arriva la batteria dei motori. **GND**: la massa. **+5V**: uscita a 5 V del regolatore interno (qui **non** si usa).

### 4.2 Tabella dei collegamenti

I numeri "GPIO" sono quelli usati nel codice; il "pin fisico" è la posizione sul connettore a 40 pin del Raspberry. Il pin 1 è all'estremità del connettore **lontana dalle porte USB** e ha la piazzola quadrata; i pin dispari sono sulla fila interna, i pari su quella verso il bordo.

| L298N | Raspberry Pi – GPIO | Raspberry Pi – pin fisico | Funzione |
|---|---|---|---|
| **IN1** | GPIO 8 | **24** | ruota sinistra – avanti |
| **IN2** | GPIO 11 | **23** | ruota sinistra – indietro |
| **ENA** | GPIO 20 | **38** | abilitazione ruota sinistra |
| **IN3** | GPIO 14 | **8** | ruota destra – indietro |
| **IN4** | GPIO 15 | **10** | ruota destra – avanti |
| **ENB** | GPIO 21 | **40** | abilitazione ruota destra |
| **GND** | GND | **39** (oppure 6) | massa comune – **obbligatoria** |

Sul connettore del Raspberry i pin usati sono questi (il resto è libero):

```
           pin dispari          pin pari
                 ┌────┬────┐
             1   │ ·  │ ·  │   2
             3   │ ·  │ ·  │   4
             5   │ ·  │ ·  │   6
             7   │ ·  │ ■  │   8   ← IN3 (GPIO 14)
             9   │ ·  │ ■  │  10   ← IN4 (GPIO 15)
                 │    ...  │
            21   │ ·  │ ·  │  22
  IN2 →     23   │ ■  │ ■  │  24   ← IN1 (GPIO 8)
                 │    ...  │
            37   │ ·  │ ■  │  38   ← ENA (GPIO 20)
  GND →     39   │ ■  │ ■  │  40   ← ENB (GPIO 21)
                 └────┴────┘
```

### 4.3 ENA ed ENB: due possibilità

Il codice tiene ENA ed ENB **sempre alti** e regola la velocità con un segnale PWM sugli ingressi IN1–IN4. Per questo vanno bene **entrambe** le soluzioni:

- **A (consigliata, come nella tabella):** togli i due ponticelli da ENA ed ENB e collega il pin "interno" di ciascuno al GPIO 20 e al GPIO 21. Così, quando il programma termina, il driver viene disabilitato del tutto.
- **B:** lascia i ponticelli al loro posto e non collegare i GPIO 20 e 21. Il robot funziona lo stesso.

> Se togli un ponticello restano due pin: uno è l'ingresso **ENA** (o ENB), l'altro è il **+5 V** della scheda. La posizione cambia da un modello all'altro: in caso di dubbio, con la batteria collegata misura con il tester la tensione di ciascun pin verso GND. **Quello con 5 V non va collegato al Raspberry**; il filo va sull'altro.

### 4.4 Alimentazione

```
   BATTERIA MOTORI (max 12 V)             ALIMENTAZIONE RASPBERRY (5 V / 5 A)
        +  ──────────► +12V  L298N               USB-C del Raspberry Pi
        −  ──────────► GND   L298N
                         │
                         └──────────────► GND del Raspberry (pin 39)
                                          ↑ MASSA COMUNE
```

1. **Batteria dei motori**: il **+** al morsetto **+12V**, il **−** al morsetto **GND** del L298N.
2. **Massa comune**: dallo **stesso morsetto GND** del L298N parte anche il filo verso il **GND del Raspberry** (pin 39). Senza massa comune i comandi non arrivano e i motori non si muovono, oppure si muovono a caso.
3. **Raspberry Pi**: alimentato a parte, con un powerbank USB-C o un regolatore da 5 V e almeno 3 A (meglio 5 A).
4. **Ponticello "5V-EN" del L298N** (vicino ai morsetti): lascialo inserito se la batteria è al massimo di 12 V.

> ⚠️ **Non alimentare il Raspberry dal morsetto +5V del L298N.** Il regolatore del L298N dà poca corrente e il Raspberry Pi 5 ne chiede molta: si riavvierebbe di continuo o si danneggerebbe il regolatore.

> ℹ️ Il L298N "mangia" circa 2 V: con una batteria da 7,4 V ai motori arrivano circa 5,4 V. Va bene per i comuni motori gialli da 3–6 V.

### 4.5 I motori

- Motore **sinistro** → morsetti **OUT1** e **OUT2**.
- Motore **destro** → morsetti **OUT3** e **OUT4**.

Il verso dei due fili di ciascun motore non si può sapere prima: si controlla con la prova dei motori (capitolo 6). Se una ruota gira al contrario, basta **invertire i suoi due fili** sui morsetti OUT.

### 4.6 Una cosa da fare sul Raspberry: liberare GPIO 14 e 15

I GPIO 14 e 15 (IN3 e IN4) sono anche la **porta seriale** del Raspberry. Se la console seriale è attiva, all'accensione il Raspberry ci manda dei dati e la ruota destra può dare degli scatti. Va disattivata una volta sola:

```bash
sudo raspi-config
```

→ *Interface Options* → *Serial Port* → alla domanda sulla login shell rispondi **No**, a quella sull'hardware seriale rispondi **No** → *Finish* → riavvia.

### 4.7 Riepilogo: lista di controllo

- [ ] IN1 → pin 24, IN2 → pin 23, IN3 → pin 8, IN4 → pin 10
- [ ] ENA → pin 38 e ENB → pin 40 (oppure ponticelli lasciati su ENA/ENB)
- [ ] GND del L298N → pin 39 del Raspberry
- [ ] Batteria motori su +12V e GND del L298N (massimo 12 V)
- [ ] Raspberry alimentato a parte, **non** dal +5V del L298N
- [ ] Motore sinistro su OUT1/OUT2, motore destro su OUT3/OUT4
- [ ] Console seriale disattivata con `raspi-config`

---

## 5. Installazione del software

Una sola riga da lanciare sul Raspberry (dettagli in `install.txt`):

```bash
sudo apt-get update && sudo apt-get install -y python3-numpy python3-pil python3-flask python3-picamera2 python3-libcamera python3-rpi-lgpio python3-lgpio && sudo pip3 install --break-system-packages opencv-python-headless==4.14.0.94 ai-edge-litert==2.2.0
```

Verifica:

```bash
python3 -c "import numpy, PIL, flask, picamera2, libcamera, RPi.GPIO, lgpio, cv2, ai_edge_litert; print('Tutte le librerie OK')"
```

---

## 6. Prima accensione: le prove

Prima di far muovere il robot da solo, si prova un pezzo alla volta. Tutte le prove si lanciano dalla cartella del progetto:

```bash
cd ~/robotics-level-4/earthrover/human_following
```

Per guardare il video, apri nel browser di un PC sulla stessa rete l'indirizzo che compare nel terminale, per esempio `http://192.168.1.50:2204/`. Per uscire da una prova premi **Ctrl+C**.

### Prova 1 – Camera

```bash
python3 prova_camera.py camera
```

Nel browser deve comparire l'immagine della camera.
**L'immagine deve essere dritta.** Se è capovolta, il robot sterzerà dalla parte sbagliata: vedi il capitolo 10.

### Prova 2 – Rete neurale

```bash
python3 prova_camera.py detection
```

Ogni oggetto riconosciuto viene incorniciato con il suo nome e la percentuale di sicurezza (es. `person 87%`). Mettiti davanti alla camera: devi vedere il riquadro `person`.

### Prova 3 – Motori

> ⚠️ **Solleva il robot** (per esempio su un libro) in modo che le ruote girino a vuoto.

```bash
python3 prova_camera.py motori
```

| Fase | Cosa deve succedere | Se non succede |
|---|---|---|
| 1 | entrambe le ruote **in avanti**: la **sinistra piano**, la **destra veloce** | vedi sotto |
| 2 | entrambe **in avanti**: la **sinistra veloce**, la **destra piano** | vedi sotto |
| 3 | rampa: entrambe le ruote accelerano dal 5% al 100% | annota a che percentuale **iniziano a girare** |

- Una ruota gira **all'indietro** → inverti i due fili di quel motore sui morsetti OUT del L298N.
- **Sinistra e destra sono scambiate** → scambia i motori: i fili su OUT1/OUT2 vanno su OUT3/OUT4 e viceversa.
- Le ruote vanno **sempre alla stessa velocità** → controlla i fili IN1–IN4.
- La ruota "piano" **non gira proprio** → metti in `PWM_MINIMO` (in `robot_toolkit.py`) il valore annotato nella fase 3.

---

## 7. Avviare il robot

```bash
python3 inseguitore.py
```

- **Video dal vivo:** l'indirizzo compare nel terminale (`>>> Video del robot: http://...:2204/`).
- **Per fermarlo:** Ctrl+C. I motori si fermano sempre, anche in caso di errore.

### Cosa si vede nel video

| Elemento | Significato |
|---|---|
| riquadro **verde spesso** | la persona che il robot sta seguendo |
| riquadri arancioni sottili | altre "persone" trovate ma scartate (troppo incerte) |
| punto rosso | centro della persona seguita |
| fascia verde verticale | zona "centrata": se la persona è qui, il robot va dritto |
| **X** (in basso) | quanto la persona è fuori centro (verde = centrata) |
| **A** (in basso) | quanta parte dell'immagine occupa la persona (verde = arrivato) |
| **L / R** | velocità della ruota sinistra e destra |
| in alto | foto al secondo (FPS) e tempi di camera e rete neurale |

### Il log

Per ogni foto il terminale stampa una riga come questa:

```
#   12 10.1fps cam   1ms inf  95ms | persone [0.88 b0.60 h0.55] | conf 0.88 errore_x +0.21 AREA 0.12 (...) | inseguo | L  20% R  40% (PWM  24/ 43)
```

Le stesse informazioni finiscono in un file CSV nella cartella `log/` (`inseguitore_<data_ora>.csv`), che si apre con Excel o LibreOffice: utile per studiare a posteriori cosa ha fatto il robot.

---

## 8. Come ragiona il robot

L'algoritmo è in `inseguitore.py`, nella funzione `decidi_e_muovi`, e segue quattro regole:

1. **Non vedo nessuno → mi fermo.** Meglio fermi che correre alla cieca.
2. **La persona riempie quasi tutta l'immagine → sono arrivato, mi fermo.** Più una persona è vicina, più appare grande: si misura l'*area* del suo riquadro (0,05 = lontana, 0,70 = vicinissima).
3. **Sterzo verso la persona.** Si misura l'*errore*: quanto la persona è lontana dal centro dell'immagine. La sterzata è **proporzionale** all'errore (*controllo proporzionale*): poco fuori centro = curva dolce, molto fuori centro = curva più stretta, con un limite massimo.
4. **Vado sempre avanti.** La sterzata toglie velocità a una ruota e la aggiunge all'altra:

```
   ruota sinistra = VELOCITA − sterzata
   ruota destra   = VELOCITA + sterzata

   persona a sinistra → sterzata > 0 → la destra va più forte → curva a sinistra
   persona a destra   → sterzata < 0 → la sinistra va più forte → curva a destra
   persona al centro  → sterzata = 0 → ruote uguali → dritto
```

Il robot non si ferma mai tra una foto e l'altra: si corregge un po' alla volta, in modo fluido.

### Cosa offre il toolkit agli studenti

| Comando | Cosa fa |
|---|---|
| `robot.cerca_persona()` | scatta una foto e restituisce la persona (o `None`) |
| `persona.errore_x` | > 0 persona a sinistra, < 0 a destra, 0 al centro |
| `persona.area` | parte dell'immagine occupata (0…1) |
| `persona.altezza`, `persona.larghezza`, `persona.basso` | misure del riquadro (0…1) |
| `persona.confidenza` | quanto la rete neurale è sicura (0…1) |
| `robot.muovi(sinistra, destra)` | velocità delle ruote, da −100 a 100 |
| `robot.ferma()` | frena |
| `robot.mostra(azione, tolleranza, soglia)` | aggiorna video e log |

---

## 9. I parametri da regolare

### In `inseguitore.py` (per gli studenti)

| Parametro | Valore | Effetto |
|---|---|---|
| `VELOCITA` | 30 | velocità di crociera (%). Troppo alta: il robot perde la persona |
| `GUADAGNO` | 30 | quanto forte sterza. Troppo alto: zigzaga. Troppo basso: la persona esce dall'immagine |
| `STERZATA_MAX` | 10 | limite della sterzata: evita curve troppo brusche |
| `TOLLERANZA` | 0.05 | zona centrale in cui il robot va dritto (±5% della larghezza) |
| `SOGLIA_VICINO` | 0.70 | area oltre la quale la persona è "raggiunta" |

**Come tarare `SOGLIA_VICINO`:** mettiti alla distanza a cui vuoi che il robot si fermi e leggi il valore `AREA` nel log.

### In `robot_toolkit.py` (per chi prepara il robot)

| Parametro | Effetto |
|---|---|
| `PWM_MINIMO` | PWM sotto il quale le ruote non girano (si trova con la prova motori) |
| `VARIAZIONE_MAX` | rampa di accelerazione: evita gli scatti |
| `CONFIDENZA_PERSONA` | sicurezza minima per seguire una persona (0,5) |
| `ORIENTAMENTO_CAMERA` | `None` = come il pannello; oppure `"rotate_180"` ecc. |
| `PIN_...` | i pin dei motori (vedi capitolo 4) |
| `LOG_TERMINALE`, `LOG_CSV` | accendono/spengono i log |

---

## 10. Problemi e soluzioni

| Problema | Causa probabile | Soluzione |
|---|---|---|
| Il robot sterza dalla parte **opposta** alla persona | immagine capovolta (camera montata al contrario) | imposta l'orientamento nel pannello di controllo, oppure `ORIENTAMENTO_CAMERA = "rotate_180"` nel toolkit |
| Dice "arrivato" anche quando sei lontano | `SOGLIA_VICINO` troppo bassa, oppure immagine capovolta | controlla `AREA` nel log e l'orientamento della camera |
| Va troppo veloce | `VELOCITA` alta | abbassa `VELOCITA` in `inseguitore.py` |
| Curva di scatto e ti perde | sterzata troppo forte | abbassa `GUADAGNO` o `STERZATA_MAX` |
| Curva troppo poco e ti perde ai lati | sterzata troppo debole | alza `STERZATA_MAX` (es. 15) |
| Le ruote ronzano ma non girano | PWM troppo basso | alza `PWM_MINIMO` (prova motori, fase 3) |
| La ruota destra dà scatti all'accensione | console seriale attiva su GPIO 14/15 | disattivala con `raspi-config` (capitolo 4.6) |
| I motori non si muovono affatto | manca la massa comune, oppure batteria scarica | collega GND del L298N al GND del Raspberry; controlla la batteria |
| `lgpio.error: 'GPIO not allocated'` | un altro programma usa gli stessi pin | il toolkit chiude da solo i programmi del pannello; se persiste, riavvia il Raspberry |
| La pagina web non si apre | porta 2204 occupata o rete diversa | il terminale lo segnala; PC e robot devono essere sulla stessa rete |
| Nessuna finestra sul Raspberry | OpenCV "headless" | normale: il video si guarda dal browser |
| Rete neurale lenta (~100–230 ms per foto) | gira sulla CPU | normale; con il Coral USB è circa 4 volte più veloce |