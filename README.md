# Verifica se un ritaglio appartiene a una foto (GUI + PDF Report)

Questo programma permette di verificare se un'immagine ritagliata proviene realmente da una foto originale.  
La verifica si basa su tecniche di computer vision affidabili e scientificamente valide, utilizzate in ambito forense, accademico e industriale.

Il programma fornisce:
- Interfaccia grafica (Tkinter)
- Visualizzazione affiancata di:
  - Foto originale
  - Ritaglio
  - Overlay del ritaglio trovato nella foto
- Report PDF automatico con timestamp, immagini e dettagli della verifica

---

# Come funziona
## Template Matching (OpenCV)

Funziona bene se il ritaglio è esattamente identico alla porzione originale (stesse dimensioni, nessuna rotazione, nessun resize).


Il programma utilizza tre concetti fondamentali di computer vision:

## 1. ORB (Oriented FAST and Rotated BRIEF)

ORB è un algoritmo che consente di individuare **punti caratteristici** in un’immagine.  
Questi punti sono zone molto informative (es. angoli, bordi particolari) che rimangono riconoscibili anche dopo trasformazioni come:

- Rotazioni
- Variazioni di illuminazione
- Piccoli cambiamenti di scala

ORB genera anche dei **descrittori**, cioè numeri che rappresentano localmente la struttura dell’immagine.  
Confrontando i descrittori del ritaglio e della foto completa, il programma cerca le corrispondenze.

### Nel programma:
ORB individua punti caratteristici sia nella foto originale sia nel ritaglio e ne confronta i descrittori per vedere se il ritaglio è presente.

---

## 2. Feature Matching (con BFMatcher)

Il programma usa un confronto basato sulla distanza tra descrittori.

Si considera un match affidabile se la distanza tra descrittori è sufficientemente bassa rispetto ad altri possibili abbinamenti.

### Nel programma:
Solo i match ritenuti sufficientemente buoni vengono considerati per i successivi step.

---

## 3. Omografia

L’omografia è una trasformazione geometrica che descrive come un piano viene mappato in un altro piano.  
Permette di modellare:

- Rotazioni
- Traslazioni
- Cambiamenti di prospettiva
- Ridimensionamenti

Se esiste un'omografia coerente tra ritaglio e foto, significa che il ritaglio può essere posizionato geometricamente all’interno della foto.

### Nel programma:
Una volta trovati i match buoni, si tenta di calcolare la matrice di omografia.  
Se riesce:
- Il ritaglio appartiene alla foto
- Il programma mostra un rettangolo verde nella posizione esatta

---

# Valenza scientifica

Le tecniche usate sono standard nel campo della computer vision e sono alla base di sistemi:

- di riconoscimento immagini
- analisi forense digitale
- robotica visiva
- ricostruzione 3D da immagini
- tracciamento oggetti

Il metodo ORB + Omografia è robusto e affidabile per verificare appartenenza di un ritaglio, purché l'immagine contenga sufficiente texture (non superfici completamente piatte).

---

# Dipendenze

Il programma richiede:

- opencv-python
- Pillow
- reportlab

Vedi `requirements.txt`.

Se si hanno problemi col Tkinter, in Windows il pacchetto arriva con l'istallazione di Python (https://www.python.org/)


---

# Come eseguire


python cutdetective.py 
