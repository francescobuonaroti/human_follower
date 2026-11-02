"""
inseguitore.py  -  L'ALGORITMO DEL ROBOT INSEGUITORE

Obiettivo: il robot deve SEGUIRE una persona, avanzando in modo continuo,
e fermarsi quando le e' abbastanza vicino.

Tutto cio' che non riguarda l'intelligenza artificiale (camera, motori, rete
neurale, streaming video) e' nascosto dentro robot_toolkit.py.
Qui dentro c'e' solo la parte interessante: DECIDERE cosa fare.

Come funziona un robot "intelligente" (il ciclo SENSE - THINK - ACT):

    1. PERCEPIRE  : la camera scatta una foto e una rete neurale cerca le
                    persone  ->  robot.cerca_persona()
    2. DECIDERE   : il NOSTRO algoritmo guarda dove si trova la persona
                    nell'immagine e sceglie la velocita' delle due ruote
                    ->  decidi_e_muovi()
    3. AGIRE      : i motori eseguono  ->  robot.muovi(sinistra, destra)
    4. RIPETERE   : piu' volte al secondo

Come sterza il robot?
    Non ha un volante: ha due ruote motrici, una a sinistra e una a destra
    (come un carro armato). Per curvare basta farle girare a velocita' diverse:
        sinistra = destra          -> va dritto
        destra piu' veloce         -> curva a SINISTRA
        sinistra piu' veloce       -> curva a DESTRA
"""
from robot_toolkit import Robot

# ---------------------------------------------------------------------------
# PARAMETRI DELL'ALGORITMO  (provate a cambiarli e osservate cosa succede!)
# ---------------------------------------------------------------------------

# Velocita' di crociera, in percentuale (0..100).
#   - troppo alta -> tra una foto e l'altra il robot fa troppa strada "alla
#                    cieca" e perde la persona
#   - troppo bassa -> il robot e' lento ad inseguire
VELOCITA = 30

# GUADAGNO: quanto "forte" sterza il robot per ogni unita' di errore.
# E' il cuore di un CONTROLLO PROPORZIONALE: piu' la persona e' lontana dal
# centro dell'immagine, piu' il robot sterza. Poco fuori centro = curva dolce.
#   - troppo alto  -> il robot sterza troppo, supera la persona e zigzaga
#   - troppo basso -> il robot curva troppo poco e la persona esce dall'immagine
GUADAGNO = 30

# Sterzata massima: anche se la persona e' al bordo dell'immagine, la curva
# non diventa mai piu' stretta di cosi'. Con VELOCITA = 30 e STERZATA_MAX = 10
# le ruote vanno al massimo 20% e 40%: curva decisa ma senza girare su se stesso.
STERZATA_MAX = 10

# Zona morta: se la persona e' entro questa distanza dal centro la
# consideriamo centrata e andiamo dritti (niente piccole correzioni inutili).
# E' una frazione della larghezza dell'immagine: 0.05 = +-5%
# (riquadro verde nel video).
TOLLERANZA = 0.05

# Come capiamo che la persona e' vicina? Piu' una persona e' vicina alla
# camera, piu' GRANDE appare nell'immagine. Misuriamo quanta parte
# dell'immagine occupa il suo riquadro (area = altezza x larghezza):
#     0.05 -> persona piccola, lontana
#     0.30 -> persona a qualche passo
#     0.70 -> la persona riempie quasi tutta l'immagine: siamo arrivati
# Nel video e' il valore "A" in basso, che diventa verde oltre la soglia.
# Per tararla: mettetevi alla distanza a cui il robot si deve fermare e
# leggete il valore AREA nel log del terminale.
SOGLIA_VICINO = 0.70


# ---------------------------------------------------------------------------
# FUNZIONI DELL'ALGORITMO
# ---------------------------------------------------------------------------

def calcola_sterzata(errore):
    """
    Trasforma l'errore (quanto la persona e' fuori centro) in una sterzata.

        sterzata = GUADAGNO * errore

    Con GUADAGNO = 30:
        errore = 0.10 (poco fuori centro)       ->  sterzata  3  (curva dolce)
        errore = 0.30 (molto fuori centro)      ->  sterzata  9
        errore = 0.45 (al bordo)                ->  27, ma limitata a STERZATA_MAX
        errore negativo (persona a destra)      ->  sterzata negativa

    Dentro la zona morta (TOLLERANZA) la sterzata e' zero: andiamo dritti.
    """
    if abs(errore) < TOLLERANZA:
        return 0
    sterzata = GUADAGNO * errore
    # limite: min/max "tagliano" il valore tra -STERZATA_MAX e +STERZATA_MAX
    return max(-STERZATA_MAX, min(STERZATA_MAX, sterzata))


def decidi_e_muovi(robot, persona):
    """
    IL CERVELLO DEL ROBOT.

    Riceve:
      robot    : per comandare i motori
      persona  : la persona vista dalla camera, oppure None se non c'e' nessuno.
                 Se c'e', ha questi valori utili (tutti tra 0 e 1):
                    persona.errore_x : > 0 persona a SINISTRA
                                       < 0 persona a DESTRA
                                       vicino a 0 = persona al centro
                    persona.area     : quanta immagine occupa (vicino a 1 = persona vicina)

    Restituisce una stringa che descrive cosa sta facendo (finisce nel log,
    insieme alla velocita' delle ruote).
    """

    # --- REGOLA 1: se non vedo nessuno, mi fermo --------------------------
    # Meglio fermi che correre alla cieca: e' la scelta piu' sicura.
    if persona is None:
        robot.ferma()
        return "non vista: fermo"

    # --- REGOLA 2: se la persona riempie quasi tutta l'immagine, mi fermo --
    # Abbiamo raggiunto l'obiettivo: restiamo fermi.
    if persona.area > SOGLIA_VICINO:
        robot.ferma()
        return "arrivato: fermo"

    # --- REGOLA 3: avanzo sterzando verso la persona ----------------------
    # La sterzata toglie velocita' a una ruota e la aggiunge all'altra.
    # Persona a SINISTRA (errore > 0, sterzata > 0):
    #     la ruota sinistra rallenta, la destra accelera -> curva a sinistra
    # Persona a DESTRA (errore < 0, sterzata < 0): il contrario.
    # Persona al centro (sterzata = 0): le ruote vanno uguali -> dritto.
    sterzata = calcola_sterzata(persona.errore_x)

    sinistra = VELOCITA - sterzata
    destra = VELOCITA + sterzata

    # Il comando resta attivo fino al prossimo giro del ciclo, quando la
    # situazione verra' rivalutata con una foto nuova: il robot non si ferma
    # mai tra una decisione e l'altra, quindi il movimento e' fluido.
    robot.muovi(sinistra, destra)

    return "inseguo"


# ---------------------------------------------------------------------------
# PROGRAMMA PRINCIPALE: il ciclo  PERCEPIRE -> DECIDERE -> AGIRE
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # "with" si occupa di spegnere i motori e chiudere la camera anche in caso
    # di errore o di Ctrl+C: non dobbiamo preoccuparcene.
    with Robot() as robot:

        # robot.attivo() diventa False quando si preme 'q' nella finestra video
        while robot.attivo():

            # 1. PERCEPIRE: la rete neurale cerca una persona nella foto
            persona = robot.cerca_persona()

            # 2. DECIDERE e 3. AGIRE: il nostro algoritmo comanda i motori
            azione = decidi_e_muovi(robot, persona)

            # 4. Mostriamo cosa ha capito il robot (finestra e pagina web).
            #    Passiamo i parametri per vedere le linee guida nel video.
            robot.mostra(azione, TOLLERANZA, SOGLIA_VICINO)