"""
robot_toolkit.py  -  LA "SCATOLA NERA" DEL ROBOT

Questo file NON fa parte della lezione: contiene tutto cio' che serve per far
funzionare il robot ma che non riguarda l'intelligenza artificiale:
camera (Picamera2), modello di rilevamento (TensorFlow Lite / Coral), motori
(driver L298N), finestra di anteprima OpenCV e streaming video via Flask.
E' autonomo: non usa common.py ne' util.py (le parti utili sono qui dentro).

Gli studenti lo usano soltanto cosi':

    from robot_toolkit import Robot

    with Robot() as robot:
        while robot.attivo():
            persona = robot.cerca_persona()      # "gli occhi"
            ...                                  # <-- il loro algoritmo
            robot.muovi(sinistra, destra) / robot.ferma()   # "le gambe"
            robot.mostra(azione, tolleranza, soglia_vicino)   # video

Prove dell'hardware (motori, camera, rete neurale): vedi in fondo al file.

Pagina web con la vista del robot:  http://<ip-del-robot>:2204/

Basato su "AI Robot - Human Following" di Jitesh Saini
(https://github.com/jiteshsaini, https://helloworld.co.in)
"""
import os
import time
import glob
import ctypes
import signal
import socket
from datetime import datetime
import logging
from dataclasses import dataclass
from threading import Thread, Condition

# --- messaggi di sistema superflui (impostati PRIMA di importare camera e OpenCV)
# libcamera: solo errori, niente INFO/WARN ad ogni avvio
os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")
# OpenCV da pip include Qt senza plugin wayland e senza font: usiamo X11 (xcb)
# e i font di sistema, cosi' spariscono gli avvisi "qt.qpa.plugin" e "QFontDatabase"
if os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
from PIL import Image
from picamera2 import Picamera2
from flask import Flask, Response, render_template
import RPi.GPIO as GPIO
import re
import collections
import numpy as np
import ai_edge_litert.interpreter as tflite

GPIO.setwarnings(False)


# Liberiamo camera e pin, come fa master.py del pannello di controllo:
#   cam_server.py   -> tiene occupata la camera (stream del pannello)
#   generate_pwm.py -> tiene occupati GPIO 20-21 (slider di velocita')
# Un pin GPIO puo' avere un solo proprietario: se un altro programma lo tiene,
# GPIO.setup fallisce con "lgpio.error: 'GPIO not allocated'".
# Se lanciato da master.py questi programmi sono gia' fermi e il comando non fa nulla.
for _prog in ("cam_server.py", "generate_pwm.py"):
    os.system("pkill -f " + _prog + " > /dev/null 2>&1")
time.sleep(0.5)   # tempo al kernel per rilasciare le linee GPIO


# "Stop" dal pannello = pkill = SIGTERM. Senza questo gestore Python
# terminerebbe di colpo, lasciando i motori nell'ultimo stato (es. avanti!).
# Cosi' invece viene eseguito Robot.chiudi(), che ferma i motori.
def _su_sigterm(signum, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, _su_sigterm)


# ---------------------------------------------------------------------------
# Impostazioni di sistema (gli studenti non devono toccarle)
# ---------------------------------------------------------------------------

# ---- Collegamento Raspberry Pi -> driver L298N (numeri GPIO "BCM") ---------
# Sono gli stessi pin di util.py del progetto originale, cosi' anche il
# pannello di controllo continua a funzionare senza ricablare nulla.
# Schema completo con i numeri dei pin fisici: vedi README.md.
#
#   Ruota SINISTRA = motore A del L298N (uscite OUT1/OUT2)
PIN_SINISTRA_AVANTI = 8          # -> IN1   (pin fisico 24)
PIN_SINISTRA_INDIETRO = 11       # -> IN2   (pin fisico 23)
PIN_ENABLE_SINISTRA = 20         # -> ENA   (pin fisico 38)
#   Ruota DESTRA = motore B del L298N (uscite OUT3/OUT4)
PIN_DESTRA_INDIETRO = 14         # -> IN3   (pin fisico 8)
PIN_DESTRA_AVANTI = 15           # -> IN4   (pin fisico 10)
PIN_ENABLE_DESTRA = 21           # -> ENB   (pin fisico 40)
#
# Se una ruota gira al contrario: inverti i due fili del motore sulle uscite
# OUTx del L298N (oppure scambia qui i due numeri AVANTI/INDIETRO di quella ruota).

DIMENSIONI_FRAME = (640, 480)

# Orientamento della camera. None = come il pannello di controllo: legge
# "flip_RPI_cam" da earthrover/config.txt (voce "Orientation" del pannello).
# Per forzarlo: "none", "rotate_180", "horizontal_flip", "vertical_flip".
# Se l'immagine e' sottosopra, anche destra e sinistra sono invertite e il
# robot sterza dalla parte sbagliata!
ORIENTAMENTO_CAMERA = None
SOGLIA_CONFIDENZA = 0.2          # minima "sicurezza" del modello per accettare un oggetto
MAX_OGGETTI = 5                  # quanti oggetti al massimo considerare per frame
# La persona da seguire deve avere almeno questa confidenza. Sotto, la rete
# produce spesso riquadri "person" sbagliati o parziali (solo gambe, riflessi...)
# che fanno saltare il robot da un riquadro all'altro. Tra piu' persone valide
# viene scelta quella con la confidenza piu' alta.
CONFIDENZA_PERSONA = 0.5

# Log di ogni fotogramma: riga nel terminale e file CSV nella cartella log/
# accanto a questo file (si apre con Excel / LibreOffice, separatore ';').
LOG_TERMINALE = True
LOG_CSV = True
OGGETTO_DA_SEGUIRE = 'person'

DIR_MODELLI = '/var/www/html/all_models'
MODELLO = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
MODELLO_EDGETPU = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
ETICHETTE = 'coco_labels.txt'

# Come si comandano i motori: la velocita' e' un PWM sugli ingressi IN1..IN4
# (il pin "avanti" o "indietro" della ruota), mentre ENA/ENB restano sempre
# ALTI. Funziona sia con ENA/ENB collegati ai GPIO 20/21, sia lasciando sul
# L298N i ponticelli (jumper) sopra ENA/ENB.
FREQ_PWM = 100                   # Hz

# Sotto una certa percentuale di PWM i motori ronzano ma non girano.
# Le velocita' degli studenti (1..100) vengono "spalmate" tra PWM_MINIMO e 100:
# cosi' anche velocita' basse muovono davvero il robot.
# Il valore giusto si trova con prova_motori() (fase "rampa").
PWM_MINIMO = 5                  # %

# Rampa di accelerazione: a ogni comando la velocita' di una ruota puo'
# cambiare al massimo di questi punti (con ~10 foto/s: da 0 a 30 in 0,3 s).
# Evita gli scatti. La frenata (velocita' 0) invece e' sempre immediata.
VARIAZIONE_MAX = 8

VELOCITA_CURVA = 15              # velocita' usata da gira_sinistra()/gira_destra()

PORTA_FLASK = 2204
QUALITA_JPEG = 80


# ---------------------------------------------------------------------------
# Impostazioni del pannello di controllo (config.txt) - era in util.py
# ---------------------------------------------------------------------------
_CARTELLA = os.path.dirname(os.path.realpath(__file__))
# config.txt si cerca accanto a questo file, poi nella cartella superiore
# (earthrover/config.txt, quella scritta dal pannello di controllo)
_FILE_CONFIG = [os.path.join(_CARTELLA, "config.txt"),
                os.path.join(os.path.dirname(_CARTELLA), "config.txt")]


def _impostazione(chiave, predefinito):
    """Un valore 'chiave=valore' da config.txt; se manca, il predefinito."""
    for percorso in _FILE_CONFIG:
        try:
            with open(percorso) as f:
                for riga in f:
                    k, _, v = riga.partition("=")
                    if k.strip() == chiave:
                        return type(predefinito)(v.strip())
        except (OSError, ValueError):
            continue
    return predefinito


# ---------------------------------------------------------------------------
# Coral USB Accelerator: rilevato automaticamente - era in util.py
# Vale 1 solo se la chiavetta e' collegata E la sua libreria e' installata.
# Per forzare: variabile d'ambiente EARTHROVER_EDGETPU=0 (CPU) oppure =1 (Coral).
# ---------------------------------------------------------------------------
_LIB_EDGETPU = 'libedgetpu.so.1'
_ID_CORAL = {("1a6e", "089a"), ("18d1", "9302")}   # prima e dopo il caricamento del firmware


def _coral_presente():
    ids = set(_ID_CORAL)
    # altri ID eventualmente dichiarati dalle regole udev di libedgetpu
    modello = re.compile(r'idVendor}=="([0-9a-fA-F]{4})".*?idProduct}=="([0-9a-fA-F]{4})"')
    for cartella in ("/usr/lib/udev/rules.d", "/lib/udev/rules.d", "/etc/udev/rules.d"):
        for regole in glob.glob(cartella + "/*edgetpu*.rules"):
            try:
                with open(regole) as f:
                    ids.update((v.lower(), p.lower()) for v, p in modello.findall(f.read()))
            except OSError:
                pass
    # dispositivi USB collegati (letti da /sys, senza bisogno di lsusb)
    for file_vendor in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            with open(file_vendor) as f:
                vid = f.read().strip().lower()
            with open(file_vendor[:-len("idVendor")] + "idProduct") as f:
                pid = f.read().strip().lower()
        except OSError:
            continue
        if (vid, pid) in ids:
            try:
                ctypes.CDLL(_LIB_EDGETPU)      # la libreria c'e'?
                return True
            except OSError:
                return False                   # chiavetta presente, libreria mancante
    return False


_forza_coral = os.environ.get("EARTHROVER_EDGETPU")
edgetpu = int(_forza_coral) if _forza_coral in ("0", "1") else int(_coral_presente())


# ---------------------------------------------------------------------------
# Pin dei motori bassi appena il toolkit viene importato: motori fermi
# (anche durante le prove di camera e rete neurale).
# ---------------------------------------------------------------------------
_PIN_DIREZIONE = (PIN_SINISTRA_AVANTI, PIN_SINISTRA_INDIETRO,
                  PIN_DESTRA_AVANTI, PIN_DESTRA_INDIETRO)
_PIN_ENABLE = (PIN_ENABLE_SINISTRA, PIN_ENABLE_DESTRA)

GPIO.setmode(GPIO.BCM)
for _pin in _PIN_DIREZIONE:
    # initial=LOW e' importante: senza, rpi-lgpio prova a leggere il pin prima
    # di averlo preso e fallisce con "lgpio.error: 'GPIO not allocated'"
    GPIO.setup(_pin, GPIO.OUT, initial=GPIO.LOW)


def _motori_off():
    """Tutti gli ingressi del L298N bassi: ruote ferme."""
    for pin in _PIN_DIREZIONE:
        try:
            GPIO.output(pin, False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rete neurale (TensorFlow Lite / LiteRT, con o senza Coral)
# Era common.py del progetto originale: ora e' qui dentro.
# ---------------------------------------------------------------------------

# Un oggetto rilevato: classe, confidenza, riquadro normalizzato 0..1
_Oggetto = collections.namedtuple('Oggetto', ['id', 'score', 'bbox'])
_BBox = collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])


def _carica_etichette(percorso):
    """coco_labels.txt -> {numero_classe: nome}, es. {0: 'person', ...}"""
    riga_valida = re.compile(r'\s*(\d+)(.+)')
    etichette = {}
    with open(percorso, 'r', encoding='utf-8') as f:
        for riga in f:
            m = riga_valida.match(riga)
            if m:
                etichette[int(m.group(1))] = m.group(2).strip()
    return etichette


def _carica_modello(cartella, modello, file_etichette, usa_coral):
    """Crea l'interprete TFLite (sul Coral se usa_coral == 1) e carica le etichette."""
    percorso_modello = os.path.join(cartella, modello)
    percorso_etichette = os.path.join(cartella, file_etichette)
    print("Modello: {}  ({})".format(percorso_modello, "Coral" if usa_coral else "CPU"))

    if usa_coral:
        interprete = tflite.Interpreter(
            model_path=percorso_modello,
            experimental_delegates=[tflite.load_delegate(_LIB_EDGETPU, {})])
    else:
        interprete = tflite.Interpreter(model_path=percorso_modello)
    interprete.allocate_tensors()
    return interprete, _carica_etichette(percorso_etichette)


def _imposta_ingresso(interprete, immagine_pil):
    """Ridimensiona la foto alla dimensione di ingresso della rete e la copia dentro."""
    _, altezza, larghezza, _ = interprete.get_input_details()[0]['shape']
    immagine = immagine_pil.resize((larghezza, altezza), Image.NEAREST)
    indice = interprete.get_input_details()[0]['index']
    interprete.tensor(indice)()[0][:, :] = immagine


def _tensore_uscita(interprete, i):
    """Uscita i-esima della rete, riportata a numeri reali se quantizzata."""
    dettagli = interprete.get_output_details()[i]
    dati = np.squeeze(interprete.tensor(dettagli['index'])())
    if 'quantization' not in dettagli:
        return dati
    scala, zero = dettagli['quantization']
    if scala == 0:
        return dati - zero
    return scala * (dati - zero)


def _leggi_oggetti(interprete, soglia, max_oggetti):
    """Elenco degli oggetti rilevati con confidenza >= soglia (modello SSD MobileNet)."""
    riquadri = _tensore_uscita(interprete, 0)
    classi = _tensore_uscita(interprete, 1)
    punteggi = _tensore_uscita(interprete, 2)
    quanti = int(_tensore_uscita(interprete, 3))

    oggetti = []
    for i in range(min(max_oggetti, quanti)):
        if punteggi[i] < soglia:
            continue
        ymin, xmin, ymax, xmax = riquadri[i]
        oggetti.append(_Oggetto(
            id=int(classi[i]),
            score=float(punteggi[i]),
            bbox=_BBox(xmin=max(0.0, float(xmin)), ymin=max(0.0, float(ymin)),
                       xmax=min(1.0, float(xmax)), ymax=min(1.0, float(ymax)))))
    return oggetti


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
_FLIP = {"none": (0, 0), "rotate_180": (1, 1),        # (hflip, vflip)
         "horizontal_flip": (1, 0), "vertical_flip": (0, 1)}


def _apri_camera():
    """
    Apre la camera CSI con Picamera2, orientata come nel pannello di controllo.
    "RGB888" in Picamera2 = ordine BGR in memoria = compatibile OpenCV.
    Il capovolgimento lo fa il sensore: non costa nulla per fotogramma.
    """
    from libcamera import Transform

    if ORIENTAMENTO_CAMERA is not None:
        orientamento, origine = ORIENTAMENTO_CAMERA, "ORIENTAMENTO_CAMERA nel toolkit"
    else:
        orientamento = _impostazione("flip_RPI_cam", "none")
        origine = "config.txt (pannello: Orientation)"
    if orientamento not in _FLIP:
        print("Orientamento '{}' sconosciuto: uso 'none'".format(orientamento))
        orientamento = "none"
    h, v = _FLIP[orientamento]
    print("Camera: orientamento '{}' (da {})".format(orientamento, origine))

    camera = Picamera2(0)
    config = camera.create_preview_configuration(
        main={"size": DIMENSIONI_FRAME, "format": "RGB888"},
        transform=Transform(hflip=h, vflip=v))
    camera.configure(config)
    camera.start()
    time.sleep(1)   # lascia assestare esposizione e bilanciamento del bianco
    return camera


# ---------------------------------------------------------------------------
# Motori: direzione e velocita' indipendenti per ruota sinistra e destra
# ---------------------------------------------------------------------------
class _Motori:
    """
    Pilota le due ruote separatamente tramite il driver L298N.
    Ogni ruota ha due ingressi (IN1/IN2 la sinistra, IN3/IN4 la destra):
      AVANTI    -> PWM sul pin "avanti", l'altro a 0
      INDIETRO  -> PWM sul pin "indietro", l'altro a 0
      FRENO     -> entrambi a 0 (con ENA/ENB alti il L298N frena)
    Valore di una ruota: da -100 (indietro a tutta) a +100 (avanti a tutta), 0 = freno.
    """

    def __init__(self):
        GPIO.setmode(GPIO.BCM)

        # --- ENA / ENB sempre alti ------------------------------------------
        for pin in _PIN_ENABLE:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

        # --- PWM sugli ingressi IN1..IN4 ------------------------------------
        self._pwm = {
            "sx": (GPIO.PWM(PIN_SINISTRA_AVANTI, FREQ_PWM),
                   GPIO.PWM(PIN_SINISTRA_INDIETRO, FREQ_PWM)),
            "dx": (GPIO.PWM(PIN_DESTRA_AVANTI, FREQ_PWM),
                   GPIO.PWM(PIN_DESTRA_INDIETRO, FREQ_PWM)),
        }
        for coppia in self._pwm.values():
            for p in coppia:
                p.start(0)

        self.duty = {"sx": 0.0, "dx": 0.0}     # ultimo PWM applicato (per i log)
        self.descrizione = (
            "Motori (L298N): SINISTRA avanti GPIO {} / indietro GPIO {}, ENA GPIO {}  |  "
            "DESTRA avanti GPIO {} / indietro GPIO {}, ENB GPIO {}".format(
                PIN_SINISTRA_AVANTI, PIN_SINISTRA_INDIETRO, PIN_ENABLE_SINISTRA,
                PIN_DESTRA_AVANTI, PIN_DESTRA_INDIETRO, PIN_ENABLE_DESTRA))
        print(self.descrizione)
        self.ferma()

    def ruota(self, lato, valore, grezzo=False):
        """
        Imposta una ruota. Con grezzo=True il valore e' il duty cycle PWM diretto
        (serve alla prova dei motori), altrimenti passa per PWM_MINIMO.
        Restituisce il valore effettivamente applicato (-100..100).
        """
        valore = max(-100.0, min(100.0, float(valore)))
        pwm_av, pwm_ind = self._pwm[lato]

        if abs(valore) < 1:
            # freno: entrambi gli ingressi bassi (enable alti)
            pwm_av.ChangeDutyCycle(0)
            pwm_ind.ChangeDutyCycle(0)
            self.duty[lato] = 0.0
            return 0.0

        if grezzo:
            duty = abs(valore)
        else:
            duty = PWM_MINIMO + (100 - PWM_MINIMO) * abs(valore) / 100

        if valore > 0:
            pwm_ind.ChangeDutyCycle(0)
            pwm_av.ChangeDutyCycle(duty)
        else:
            pwm_av.ChangeDutyCycle(0)
            pwm_ind.ChangeDutyCycle(duty)
        self.duty[lato] = duty
        return valore

    def muovi(self, sinistra, destra):
        return self.ruota("sx", sinistra), self.ruota("dx", destra)

    def ferma(self):
        self.muovi(0, 0)

    def spegni(self):
        """Uscita: freno, poi PWM fermati, ingressi ed enable bassi."""
        try:
            self.ferma()
            time.sleep(0.05)
        finally:
            for coppia in self._pwm.values():
                for p in coppia:
                    try:
                        p.ChangeDutyCycle(0)
                        p.stop()
                    except Exception:
                        pass
            _motori_off()
            for pin in _PIN_ENABLE:
                try:
                    GPIO.output(pin, False)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Cio' che il robot "vede": una persona
# ---------------------------------------------------------------------------
@dataclass
class Persona:
    """
    Descrive la persona rilevata nel fotogramma.

    Tutte le coordinate sono NORMALIZZATE tra 0 e 1:
      x: 0 = bordo sinistro dell'immagine, 1 = bordo destro
      y: 0 = bordo ALTO dell'immagine,     1 = bordo BASSO  (l'asse y va verso il basso!)
    """
    x_centro: float      # centro orizzontale della persona
    y_centro: float      # centro verticale della persona
    errore_x: float      # 0.5 - x_centro: > 0 persona a SINISTRA, < 0 persona a DESTRA, 0 = centrata
    basso: float         # y del bordo inferiore (i "piedi"): piu' e' vicino a 1, piu' la persona e' vicina
    altezza: float       # altezza del riquadro (0..1): una persona vicina appare grande
    larghezza: float     # larghezza del riquadro (0..1): anche questa cresce avvicinandosi
    area: float          # parte dell'immagine occupata dal riquadro (altezza x larghezza):
                         # 0.05 = persona piccola e lontana, 0.8 = riempie quasi tutto
    confidenza: float    # sicurezza del modello (0..1)
    bbox: tuple          # (x_min, y_min, x_max, y_max), usato per il disegno


# ---------------------------------------------------------------------------
# Streaming Flask (interno)
# ---------------------------------------------------------------------------
class _BufferStream:
    """Tiene l'ultimo JPEG prodotto; i client Flask lo leggono appena e' nuovo."""

    def __init__(self):
        self.cond = Condition()
        self.jpeg = None
        self.frame_id = 0
        self.clients = 0
        self.chiuso = False

    def pubblica(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.frame_id += 1
            self.cond.notify_all()

    def attendi(self, ultimo_id, timeout=2.0):
        with self.cond:
            self.cond.wait_for(
                lambda: self.frame_id != ultimo_id or self.chiuso, timeout=timeout)
            if self.chiuso or self.frame_id == ultimo_id:
                return ultimo_id, None
            return self.frame_id, self.jpeg

    def aggiungi_client(self):
        with self.cond:
            self.clients += 1

    def rimuovi_client(self):
        with self.cond:
            self.clients = max(0, self.clients - 1)

    def ha_client(self):
        return self.clients > 0

    def chiudi(self):
        with self.cond:
            self.chiuso = True
            self.cond.notify_all()


_stream = _BufferStream()
_app = Flask(__name__)

_PAGINA = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot inseguitore</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
 h1{font-size:1.1rem;padding:10px;margin:0}
 img{max-width:100%;height:auto;border:1px solid #444}
</style></head>
<body><h1>Robot inseguitore - vista della camera</h1>
<img src="/video_feed" alt="vista del robot"></body></html>"""


# endpoint espliciti: i nomi 'index' e 'video_feed' sono quelli che
# templates/index.html del progetto originale usa con url_for(...)
@_app.route('/', endpoint='index')
def _index():
    try:
        return render_template("index.html")
    except Exception as e:   # template assente o non compatibile
        print("templates/index.html non utilizzabile ({}): uso la pagina integrata".format(e))
        return _PAGINA


def _genera_mjpeg():
    _stream.aggiungi_client()
    ultimo = -1
    try:
        while True:
            ultimo, jpeg = _stream.attendi(ultimo)
            if _stream.chiuso:
                break
            if jpeg is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
    finally:
        _stream.rimuovi_client()


@_app.route('/video_feed', endpoint='video_feed')
def _video_feed():
    return Response(_genera_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _ip_locale():
    """IP del robot sulla LAN (nessun pacchetto viene davvero inviato)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "<ip-del-robot>"
    finally:
        s.close()


def _avvia_flask(porta):
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    def _esegui():
        try:
            _app.run(host='0.0.0.0', port=porta, threaded=True, use_reloader=False)
        except Exception as e:
            print("\n*** Flask NON avviato sulla porta {}: {}".format(porta, e))
            print("*** Probabilmente la porta e' gia' usata da un altro programma.")
            print("*** Controlla con:  sudo ss -ltnp | grep {}\n".format(porta))

    t = Thread(target=_esegui, daemon=True)
    t.start()
    time.sleep(0.5)
    if t.is_alive():
        print("\n>>> Video del robot:  http://{}:{}/\n".format(_ip_locale(), porta))


# ---------------------------------------------------------------------------
# Overlay video: stesso layout di append_text_img1 del progetto originale
# ---------------------------------------------------------------------------
def _disegna_overlay(cv2_im, oggetti, etichette, dur, persona, cmd, vel,
                     tolleranza, soglia_vicino):
    height, width = cv2_im.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # barra nera in alto
    cv2_im = cv2.rectangle(cv2_im, (0, 0), (width, 24), (0, 0, 0), -1)

    # durate di elaborazione
    cam = int(round(dur[0] * 1000))
    inference = int(round(dur[1] * 1000))
    other = int(round(dur[2] * 1000))
    text_dur = 'Camera: {}ms   Inference: {}ms   other: {}ms'.format(cam, inference, other)
    cv2_im = cv2.putText(cv2_im, text_dur, (int(width / 4) - 30, 16), font, 0.4, (255, 255, 255), 1)

    # FPS
    total = cam + inference + other
    fps = round(1000 / total, 1) if total else 0.0
    cv2_im = cv2.putText(cv2_im, 'FPS: {}'.format(fps), (10, 20), font, 0.7, (150, 150, 255), 2)

    # barra nera in basso
    cv2_im = cv2.rectangle(cv2_im, (0, height - 24), (width, height), (0, 0, 0), -1)

    # tolleranza e deviazioni
    cv2_im = cv2.putText(cv2_im, 'Tol : {}'.format(tolleranza), (10, height - 8),
                         font, 0.55, (150, 150, 255), 2)

    x_dev = persona.errore_x if persona else 0.0
    color_x = (0, 255, 0) if abs(x_dev) < tolleranza else (0, 0, 255)
    cv2_im = cv2.putText(cv2_im, 'X: {:+.2f}'.format(x_dev), (110, height - 8),
                         font, 0.55, color_x, 2)

    # area occupata dalla persona: diventa verde quando supera la soglia di arrivo
    area = persona.area if persona else 0.0
    color_a = (0, 255, 0) if area > soglia_vicino else (0, 0, 255)
    cv2_im = cv2.putText(cv2_im, 'A: {:.2f}'.format(area), (220, height - 8),
                         font, 0.55, color_a, 2)

    # comando, velocita' e stato
    cv2_im = cv2.putText(cv2_im, str(cmd), (int(width / 2) + 10, height - 8),
                         font, 0.68, (0, 255, 255), 2)
    # velocita' delle due ruote (sinistra / destra)
    str_sp = 'L {:.0f}%  R {:.0f}%'.format(vel[0], vel[1])
    cv2_im = cv2.putText(cv2_im, str_sp, (int(width / 2) + 165, height - 8),
                         font, 0.55, (150, 150, 255), 2)

    if cmd == 0:
        stato = "No object"
    elif cmd == 'Stop':
        stato = 'Acquired'
    else:
        stato = 'Tracking'
    cv2_im = cv2.putText(cv2_im, stato, (width - 140, 18), font, 0.7, (0, 255, 255), 2)

    # croce centrale
    cv2_im = cv2.rectangle(cv2_im, (0, int(height / 2) - 1), (width, int(height / 2) + 1), (255, 0, 0), -1)
    cv2_im = cv2.rectangle(cv2_im, (int(width / 2) - 1, 0), (int(width / 2) + 1, height), (255, 0, 0), -1)

    # punto rosso al centro della persona
    if persona:
        cv2_im = cv2.circle(cv2_im, (int(persona.x_centro * width), int(persona.y_centro * height)),
                            7, (0, 0, 255), -1)

    # fascia di tolleranza
    cv2_im = cv2.rectangle(cv2_im, (int(width / 2 - tolleranza * width), 0),
                           (int(width / 2 + tolleranza * width), height), (0, 255, 0), 2)

    # riquadri delle persone
    for obj in oggetti:
        nome = etichette.get(obj.id, obj.id)
        if nome != "person":
            continue
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)
        testo = '{}% {}'.format(int(100 * obj.score), nome)
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 150, 255), 1)
        cv2_im = cv2.putText(cv2_im, testo, (x0, y1 - 5), font, 0.5, (0, 255, 0), 1)

    # la persona scelta da seguire: riquadro verde spesso
    # (i riquadri arancioni sottili senza verde sono quelli scartati)
    if persona:
        x0, y0, x1, y1 = persona.bbox
        cv2_im = cv2.rectangle(cv2_im, (int(x0 * width), int(y0 * height)),
                               (int(x1 * width), int(y1 * height)), (0, 255, 0), 3)

    return cv2_im


# ---------------------------------------------------------------------------
# Il robot
# ---------------------------------------------------------------------------
class Robot:
    """
    Occhi:   cerca_persona()
    Gambe:   muovi(sinistra, destra)   velocita' delle due ruote, da -100 a 100
             ferma()
             avanti(velocita), gira_sinistra(secondi), gira_destra(secondi)
    Video:   mostra(azione, tolleranza, soglia_vicino)
    Ciclo:   attivo()   -> False quando si preme 'q' nella finestra di anteprima
    """

    def __init__(self, finestra=True, porta=PORTA_FLASK):
        # senza display (es. avvio dal pannello web o via SSH senza -X) niente finestra:
        # Qt in quel caso termina il programma invece di dare un errore gestibile
        if finestra and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("Nessun display: anteprima locale disattivata, video solo via web")
            finestra = False

        self._finestra = finestra
        self._attivo = True
        self._frame = None
        self._oggetti = []
        self._persona = None
        self._dur = [0.0, 0.0, 0.0]     # camera, inferenza, altro (secondi)
        self._fine_inferenza = time.time()
        self._cmd = 0                   # comando da mostrare nel video
        self._vel = (0.0, 0.0)          # velocita' applicate (sinistra, destra)
        self._motori = None
        self._persone = []              # tutte le persone viste: (confidenza, bbox)
        self._n_frame = 0
        self._t_frame_prec = time.time()
        self._fps = 0.0
        self._csv = None

        # 1) modello (e' il passo piu' probabile a fallire: lo facciamo per primo,
        #    quando i motori sono ancora spenti)
        nome_modello = MODELLO_EDGETPU if edgetpu == 1 else MODELLO
        self._interprete, self._etichette = _carica_modello(
            DIR_MODELLI, nome_modello, ETICHETTE, edgetpu)

        # 2) camera, con lo stesso orientamento del pannello di controllo
        self._camera = _apri_camera()

        # 3) motori: abilitati SOLO ora che tutto il resto e' pronto
        self._motori = _Motori()

        # 4) streaming video
        _avvia_flask(porta)

        # 5) file di log
        if LOG_CSV:
            self._apri_csv()

    # ----- log ----------------------------------------------------------------
    _COLONNE_CSV = ["ora", "frame", "fps", "camera_ms", "inferenza_ms",
                    "persone_viste", "persone (conf/basso/altezza)",
                    "scelta_conf", "x_centro", "errore_x", "basso", "altezza", "larghezza", "area",
                    "azione", "comando", "vel_sx", "vel_dx", "pwm_sx", "pwm_dx"]

    def _apri_csv(self):
        try:
            cartella = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log")
            os.makedirs(cartella, exist_ok=True)
            nome = os.path.join(cartella, datetime.now().strftime("inseguitore_%Y%m%d_%H%M%S.csv"))
            self._csv = open(nome, "w", encoding="utf-8")
            self._csv.write(";".join(self._COLONNE_CSV) + "\n")
            self._csv.flush()
            print(">>> Log CSV:", nome)
        except OSError as e:
            print("Log CSV disattivato:", e)
            self._csv = None

    @staticmethod
    def _num(v, cifre=3):
        """Numero per il CSV con la virgola decimale (Excel/LibreOffice in italiano)."""
        return "{:.{}f}".format(v, cifre).replace(".", ",")

    def _registra(self, azione):
        p = self._persona
        sx, dx = self._vel
        duty = self._motori.duty if self._motori else {"sx": 0, "dx": 0}
        cam_ms, inf_ms = int(self._dur[0] * 1000), int(self._dur[1] * 1000)
        persone_txt = " ".join("[{:.2f} b{:.2f} h{:.2f}]".format(c, b[3], b[3] - b[1])
                               for c, b in self._persone) or "-"

        if LOG_TERMINALE:
            if p:
                scelta = "conf {:.2f} errore_x {:+.2f} AREA {:.2f} (alt {:.2f} larg {:.2f} basso {:.2f})".format(
                    p.confidenza, p.errore_x, p.area, p.altezza, p.larghezza, p.basso)
            else:
                scelta = "nessuna persona valida (conf >= {})".format(CONFIDENZA_PERSONA)
            print("#{:5d} {:4.1f}fps cam {:3d}ms inf {:3d}ms | persone {} | {} | {} | "
                  "L {:3.0f}% R {:3.0f}% (PWM {:3.0f}/{:3.0f})".format(
                      self._n_frame, self._fps, cam_ms, inf_ms, persone_txt, scelta,
                      azione, sx, dx, duty["sx"], duty["dx"]))

        if self._csv:
            n = self._num
            riga = [datetime.now().strftime("%H:%M:%S.%f")[:-3], str(self._n_frame),
                    n(self._fps, 1), str(cam_ms), str(inf_ms),
                    str(len(self._persone)), persone_txt.replace(".", ",")]
            if p:
                riga += [n(p.confidenza, 2), n(p.x_centro), n(p.errore_x), n(p.basso),
                         n(p.altezza), n(p.larghezza), n(p.area)]
            else:
                riga += [""] * 7
            riga += [str(azione).replace(";", ","), str(self._cmd),
                     n(sx, 0), n(dx, 0), n(duty["sx"], 0), n(duty["dx"], 0)]
            try:
                self._csv.write(";".join(riga) + "\n")
                self._csv.flush()
            except OSError:
                pass

    # ----- gestione del ciclo di vita -------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, tipo, valore, traceback):
        self.chiudi()
        return tipo is KeyboardInterrupt   # Ctrl+C = uscita pulita, senza errori

    def attivo(self):
        return self._attivo

    def chiudi(self):
        """Ferma i motori e rilascia camera e finestre. Sicuro da chiamare piu' volte."""
        try:
            if self._motori is not None:
                self._motori.spegni()
            else:
                _motori_off()
        finally:
            _stream.chiudi()
            try:
                self._camera.stop()
                self._camera.close()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            if self._csv:
                try:
                    self._csv.close()
                except OSError:
                    pass
                self._csv = None

    # ----- occhi ------------------------------------------------------------
    def cerca_persona(self):
        """
        Scatta una foto, la passa alla rete neurale e restituisce la persona
        piu' sicura (oggetto Persona) oppure None se non c'e' nessuno.
        """
        adesso = time.time()
        dt = adesso - self._t_frame_prec
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._t_frame_prec = adesso
        self._n_frame += 1

        t0 = time.time()
        frame = self._camera.capture_array()
        self._frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._dur[0] = time.time() - t0

        t1 = time.time()
        _imposta_ingresso(self._interprete, pil)
        self._interprete.invoke()
        self._oggetti = _leggi_oggetti(
            self._interprete, soglia=SOGLIA_CONFIDENZA, max_oggetti=MAX_OGGETTI)
        self._fine_inferenza = time.time()
        self._dur[1] = self._fine_inferenza - t1

        # tutte le persone viste (per i log). float(): il modello restituisce
        # numpy.float32, che time.sleep() di Python 3.13 non accetta.
        self._persone = []
        for obj in self._oggetti:
            if self._etichette.get(obj.id, obj.id) == OGGETTO_DA_SEGUIRE:
                bbox = tuple(min(1.0, max(0.0, float(v))) for v in obj.bbox)
                self._persone.append((float(obj.score), bbox))
        self._persone.sort(key=lambda pb: pb[0], reverse=True)

        # la persona da seguire: la piu' sicura, se supera CONFIDENZA_PERSONA
        self._persona = None
        if self._persone and self._persone[0][0] >= CONFIDENZA_PERSONA:
            score, (x_min, y_min, x_max, y_max) = self._persone[0]
            x_centro = round((x_min + x_max) / 2, 3)
            y_centro = round((y_min + y_max) / 2, 3)
            self._persona = Persona(
                x_centro=x_centro,
                y_centro=y_centro,
                errore_x=round(0.5 - x_centro, 3),
                basso=round(y_max, 3),
                altezza=round(y_max - y_min, 3),
                larghezza=round(x_max - x_min, 3),
                area=round((x_max - x_min) * (y_max - y_min), 3),
                confidenza=round(score, 2),
                bbox=(x_min, y_min, x_max, y_max))

        return self._persona

    # ----- gambe ------------------------------------------------------------
    def muovi(self, sinistra, destra):
        """
        Imposta la velocita' delle due ruote, da -100 (indietro) a 100 (avanti).
        Il comando resta attivo finche' non ne arriva un altro: non blocca.
          muovi(50, 50)  -> dritto a meta' velocita'
          muovi(30, 70)  -> avanti curvando a SINISTRA (la ruota destra va piu' veloce)
          muovi(0, 0)    -> fermo
        Le variazioni di velocita' sono addolcite (VARIAZIONE_MAX per comando);
        la frenata invece e' immediata.
        """
        nuove = []
        for attuale, richiesta in zip(self._vel, (sinistra, destra)):
            richiesta = max(-100.0, min(100.0, float(richiesta)))
            if abs(richiesta) < 1:
                nuove.append(0.0)                          # frenata: subito
            else:
                passo = max(-VARIAZIONE_MAX, min(VARIAZIONE_MAX, richiesta - attuale))
                nuove.append(attuale + passo)
        self._vel = self._motori.muovi(*nuove)
        sx, dx = self._vel
        if sx == 0 and dx == 0:
            self._cmd = "Stop"
        elif dx - sx > 5:
            self._cmd = "Move Left"
        elif sx - dx > 5:
            self._cmd = "Move Right"
        else:
            self._cmd = "forward"

    def ferma(self):
        self.muovi(0, 0)

    def avanti(self, velocita=100):
        """Dritto alla velocita' indicata finche' non arriva un altro comando."""
        self.muovi(velocita, velocita)

    def gira_sinistra(self, secondi):
        """Rotazione sul posto a sinistra per 'secondi', poi si ferma."""
        self._vel = self._motori.muovi(-VELOCITA_CURVA, VELOCITA_CURVA)
        time.sleep(float(secondi))
        self.ferma()
        self._cmd = "Move Left"

    def gira_destra(self, secondi):
        """Rotazione sul posto a destra per 'secondi', poi si ferma."""
        self._vel = self._motori.muovi(VELOCITA_CURVA, -VELOCITA_CURVA)
        time.sleep(float(secondi))
        self.ferma()
        self._cmd = "Move Right"

    # ----- video ------------------------------------------------------------
    def mostra(self, azione="", tolleranza=0.1, soglia_vicino=0.9):
        """
        Disegna sul fotogramma quello che il robot ha capito (stesso layout del
        progetto originale) e lo mostra nella finestra locale e sulla pagina web.
          azione         : descrizione dell'azione scelta (finisce nei log)
          tolleranza     : larghezza della fascia centrale (riquadro verde)
          soglia_vicino  : soglia di arrivo (la "A" diventa verde quando l'area la supera)
        """
        if self._frame is None:
            return

        self._dur[2] = time.time() - self._fine_inferenza

        self._registra(azione)

        cmd =0 if self._persona is None else (self._cmd or "Stop")

        img = _disegna_overlay(self._frame.copy(), self._oggetti, self._etichette,
                               self._dur, self._persona, cmd, self._vel,
                               tolleranza, soglia_vicino)

        # finestra locale
        if self._finestra:
            try:
                cv2.imshow("Robot inseguitore", img)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    self._attivo = False
            except cv2.error as e:
                print("Anteprima locale disattivata:", e)
                self._finestra = False

        # stream web: lavoriamo solo se qualcuno sta guardando
        if _stream.ha_client():
            ok, jpeg = cv2.imencode('.jpg', img,
                                    [cv2.IMWRITE_JPEG_QUALITY, QUALITA_JPEG])
            if ok:
                _stream.pubblica(jpeg.tobytes())


# ===========================================================================
# PROVE DELL'HARDWARE
#
#   import robot_toolkit as rt
#   rt.prova_camera()       -> solo l'immagine della camera       (motori spenti)
#   rt.prova_detection()    -> camera + TUTTE le detection         (motori spenti)
#   rt.prova_motori()       -> ruota sinistra, ruota destra, rampa di PWM
#                              (ROBOT SOLLEVATO, ruote libere!)
#
# Video nella finestra locale (se c'e' un display) e sulla pagina web
# (porta 2204). Uscita: 'q' nella finestra oppure Ctrl+C.
# ===========================================================================


def prova_motori(lenta=20, veloce=100, secondi=3.0, passo_rampa=5):
    """
    Verifica dei motori. SOLLEVA IL ROBOT: le ruote devono girare a vuoto.
      1) entrambe AVANTI, SINISTRA piano e DESTRA veloce
      2) entrambe AVANTI, SINISTRA veloce e DESTRA piano
         -> una ruota va all'indietro        : inverti i due fili di quel motore
                                               sulle uscite OUTx del L298N
         -> sinistra e destra scambiate      : scambia i cavi dei motori
                                               (OUT1/OUT2 <-> OUT3/OUT4)
         -> la ruota "piano" non gira        : alza PWM_MINIMO
      3) rampa di PWM su entrambe (0 -> 100%): annota la % a cui le ruote
         iniziano DAVVERO a girare e mettila in PWM_MINIMO
    """
    print("\nPROVA MOTORI - solleva il robot! Inizio tra 3 secondi (Ctrl+C per annullare)")
    time.sleep(3)
    motori = _Motori()
    try:
        print("\n1) Entrambe AVANTI:  SINISTRA PIANO  -  DESTRA VELOCE")
        motori.muovi(lenta, veloce)
        time.sleep(secondi)
        motori.ferma()
        time.sleep(1)

        print("2) Entrambe AVANTI:  SINISTRA VELOCE  -  DESTRA PIANO")
        motori.muovi(veloce, lenta)
        time.sleep(secondi)
        motori.ferma()
        time.sleep(1)

        print("   Una ruota gira all'indietro     -> inverti i due fili di quel motore (OUTx del L298N)")
        print("   Sinistra e destra scambiate     -> scambia i motori: OUT1/OUT2 <-> OUT3/OUT4")
        print("   Stessa velocita' su entrambe    -> controlla i fili IN1..IN4 (vedi README.md)")
        print("   La ruota 'piano' non gira       -> alza PWM_MINIMO (vedi rampa)")

        print("\n3) Rampa: annota la percentuale a cui le ruote iniziano a girare")
        for duty in range(passo_rampa, 101, passo_rampa):
            motori.ruota("sx", duty, grezzo=True)
            motori.ruota("dx", duty, grezzo=True)
            print("   PWM {:3d}%".format(duty))
            time.sleep(0.8)
        motori.ferma()
        print("\nFine. Valore attuale di PWM_MINIMO: {}%".format(PWM_MINIMO))
    except KeyboardInterrupt:
        print("\nProva motori interrotta")
    finally:
        motori.spegni()

# un colore fisso per ogni classe (stessa classe = stesso colore), formato BGR
_PROVA_COLORI = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
                 (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255),
                 (255, 0, 128), (128, 255, 0)]


def _prova_finestra_disponibile(finestra):
    if finestra and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("Nessun display: anteprima locale disattivata, video solo via web")
        return False
    return finestra


def _prova_apri_camera():
    return _apri_camera()


def _prova_chiudi(camera):
    _stream.chiudi()
    try:
        camera.stop()
        camera.close()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print("Prova terminata")


def _prova_scritta(img, testo, pos, colore=(255, 255, 255), scala=0.55):
    """Testo con contorno nero, leggibile su qualunque sfondo."""
    cv2.putText(img, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, scala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, scala, colore, 1, cv2.LINE_AA)


def _prova_mostra(img, titolo, stato):
    """Finestra locale + stream web. Restituisce False se e' stato premuto 'q'."""
    continua = True
    if stato["finestra"]:
        try:
            cv2.imshow(titolo, img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                continua = False
        except cv2.error as e:
            print("Anteprima locale disattivata:", e)
            stato["finestra"] = False
    if _stream.ha_client():
        ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, QUALITA_JPEG])
        if ok:
            _stream.pubblica(jpeg.tobytes())
    return continua


def prova_camera(finestra=True, porta=PORTA_FLASK):
    """Mostra solo l'immagine della camera con risoluzione e FPS. Niente modello, niente motori."""
    stato = {"finestra": _prova_finestra_disponibile(finestra)}
    camera = _prova_apri_camera()
    _avvia_flask(porta)
    print("PROVA CAMERA avviata  ('q' nella finestra o Ctrl+C per uscire)")

    t_prec = time.time()
    try:
        while True:
            img = camera.capture_array()
            adesso = time.time()
            fps = 1.0 / (adesso - t_prec) if adesso > t_prec else 0.0
            t_prec = adesso

            h, w = img.shape[:2]
            _prova_scritta(img, "PROVA CAMERA", (8, 22), (0, 255, 255), 0.65)
            _prova_scritta(img, "{}x{}   FPS {:.1f}".format(w, h, fps), (8, h - 10))

            if not _prova_mostra(img, "Prova camera", stato):
                break
    except KeyboardInterrupt:
        pass
    finally:
        _prova_chiudi(camera)


def prova_detection(soglia=SOGLIA_CONFIDENZA, max_oggetti=10, finestra=True, porta=PORTA_FLASK):
    """
    Camera + rete neurale: disegna TUTTI gli oggetti riconosciuti (non solo le
    persone) con nome e confidenza. Niente motori.
      soglia      : confidenza minima per mostrare un oggetto (0..1)
      max_oggetti : quanti oggetti al massimo per fotogramma
    Nel terminale viene stampato l'elenco degli oggetti visti quando cambia.
    """
    stato = {"finestra": _prova_finestra_disponibile(finestra)}

    nome_modello = MODELLO_EDGETPU if edgetpu == 1 else MODELLO
    backend = "Coral" if edgetpu == 1 else "CPU"
    interprete, etichette = _carica_modello(DIR_MODELLI, nome_modello, ETICHETTE, edgetpu)

    camera = _prova_apri_camera()
    _avvia_flask(porta)
    print("PROVA DETECTION avviata su {}  ('q' nella finestra o Ctrl+C per uscire)".format(backend))

    ultimo_elenco = None
    t_prec = time.time()
    try:
        while True:
            t0 = time.time()
            img = camera.capture_array()
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            t_cam = time.time() - t0

            t1 = time.time()
            _imposta_ingresso(interprete, pil)
            interprete.invoke()
            oggetti = _leggi_oggetti(interprete, soglia=soglia, max_oggetti=max_oggetti)
            t_inf = time.time() - t1

            adesso = time.time()
            fps = 1.0 / (adesso - t_prec) if adesso > t_prec else 0.0
            t_prec = adesso

            h, w = img.shape[:2]
            nomi = []
            for obj in oggetti:
                nome = str(etichette.get(obj.id, obj.id))
                nomi.append(nome)
                colore = _PROVA_COLORI[int(obj.id) % len(_PROVA_COLORI)]

                x0, y0, x1, y1 = (float(v) for v in obj.bbox)
                p0 = (max(0, int(x0 * w)), max(0, int(y0 * h)))
                p1 = (min(w - 1, int(x1 * w)), min(h - 1, int(y1 * h)))
                cv2.rectangle(img, p0, p1, colore, 2)

                etichetta = "{} {}%".format(nome, int(float(obj.score) * 100))
                (tw, th), _ = cv2.getTextSize(etichetta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(p0[1], th + 4)
                cv2.rectangle(img, (p0[0], ty - th - 4), (p0[0] + tw + 4, ty), colore, -1)
                cv2.putText(img, etichetta, (p0[0] + 2, ty - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            _prova_scritta(img, "PROVA DETECTION - {}".format(backend), (8, 22), (0, 255, 255), 0.65)
            _prova_scritta(img, "Camera {}ms  Inferenza {}ms  FPS {:.1f}  oggetti {}  soglia {}".format(
                int(t_cam * 1000), int(t_inf * 1000), fps, len(oggetti), soglia), (8, h - 10), scala=0.5)

            elenco = ", ".join(sorted(nomi)) if nomi else "nessun oggetto"
            if elenco != ultimo_elenco:
                print("Visti:", elenco)
                ultimo_elenco = elenco

            if not _prova_mostra(img, "Prova detection", stato):
                break
    except KeyboardInterrupt:
        pass
    finally:
        _prova_chiudi(camera)