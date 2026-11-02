"""
prova_camera.py  -  VERIFICA DELL'HARDWARE PRIMA DELLA LEZIONE

Tre prove:
  "camera"     -> solo l'immagine della camera (con FPS)               motori spenti
  "detection"  -> camera + tutti gli oggetti riconosciuti dalla rete    motori spenti
  "motori"     -> ruota sinistra, ruota destra, rampa di velocita'      ROBOT SOLLEVATO!

Uso:
  - da Geany: cambia MODALITA qui sotto e premi Esegui
  - da terminale:  python3 prova_camera.py camera
                   python3 prova_camera.py detection
                   python3 prova_camera.py motori

Video nella finestra e su http://<ip-del-robot>:2204/
Per uscire: 'q' nella finestra oppure Ctrl+C.
"""
import sys
import robot_toolkit as rt

MODALITA = "camera"        # "camera", "detection" oppure "motori"
SOGLIA = 0.4               # solo per "detection": confidenza minima (0..1)

if len(sys.argv) > 1:      # un argomento da terminale ha la precedenza
    MODALITA = sys.argv[1].lower()

if MODALITA.startswith("det"):
    rt.prova_detection(soglia=SOGLIA)
elif MODALITA.startswith("mot"):
    rt.prova_motori()
else:
    rt.prova_camera()
