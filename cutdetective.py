import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os
from datetime import datetime


# ----------------------------------------------------
# Funzione principale di matching + overlay
# ----------------------------------------------------
def verifica_ritaglio(path_originale, path_ritaglio):
    img = cv2.imread(path_originale, cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(path_ritaglio, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return "Errore: impossibile caricare l'immagine originale.", None, None, None
    if crop is None:
        return "Errore: impossibile caricare il ritaglio.", None, None, None

    # ORB
    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(crop, None)

    if des1 is None or des2 is None:
        return "Nessuna feature trovata (immagini troppo uniformi).", None, img, crop

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des2, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    log = []
    log.append(f"Match trovati: {len(matches)}")
    log.append(f"Match buoni: {len(good)}")

    if len(good) < 10:
        log.append("RISULTATO: Il ritaglio NON appartiene alla foto.")
        return "\n".join(log), None, img, crop

    # Omografia
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        log.append("Omografia non trovata.")
        log.append("RISULTATO: Il ritaglio NON appartiene alla foto.")
        return "\n".join(log), None, img, crop

    # Trasformazione dei bordi del ritaglio
    h, w = crop.shape
    corners_crop = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    corners_transformed = cv2.perspectiveTransform(corners_crop, H)

    # Creazione overlay
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_color, [np.int32(corners_transformed)], True, (0,255,0), 3)

    overlay_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)

    log.append("Omografia valida.")
    log.append("RISULTATO: Il ritaglio APPARTIENE alla foto.")

    return "\n".join(log), overlay_pil, img, crop



# ----------------------------------------------------
# Generazione Report PDF con timestamp
# ----------------------------------------------------
def genera_report(log_text, path_originale, path_ritaglio, overlay_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"report_verifica_{timestamp}.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("<b>Report verifica ritaglio in foto</b>", styles["Title"]))
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("<b>Risultato e Analisi</b>", styles["Heading2"]))
    flow.append(Paragraph(log_text.replace("\n", "<br/>"), styles["BodyText"]))
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("<b>Immagini</b>", styles["Heading2"]))

    img1 = "tmp_originale.jpg"
    img2 = "tmp_ritaglio.jpg"
    img3 = "tmp_overlay.jpg"

    Image.fromarray(cv2.cvtColor(cv2.imread(path_originale), cv2.COLOR_BGR2RGB)).save(img1)
    Image.fromarray(cv2.cvtColor(cv2.imread(path_ritaglio), cv2.COLOR_BGR2RGB)).save(img2)
    if overlay_image is not None:
        overlay_image.save(img3)

    flow.append(Paragraph("Immagine originale:", styles["BodyText"]))
    flow.append(RLImage(img1, width=350, height=350 * 0.75))
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("Ritaglio:", styles["BodyText"]))
    flow.append(RLImage(img2, width=200, height=200 * 0.75))
    flow.append(Spacer(1, 20))

    if overlay_image is not None:
        flow.append(Paragraph("Overlay (ritaglio trovato):", styles["BodyText"]))
        flow.append(RLImage(img3, width=350, height=350 * 0.75))

    doc.build(flow)

    for f in [img1, img2, img3]:
        if os.path.exists(f):
            os.remove(f)

    return pdf_path



# ----------------------------------------------------
# GUI
# ----------------------------------------------------
class App:
    def __init__(self, master):
        self.master = master
        master.title("Verifica Ritaglio in Foto")

        self.path_originale = tk.StringVar()
        self.path_ritaglio = tk.StringVar()

        tk.Label(master, text="Foto Originale:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.path_originale, width=60).grid(row=0, column=1)
        tk.Button(master, text="Sfoglia", command=self.carica_originale).grid(row=0, column=2)

        tk.Label(master, text="Ritaglio:").grid(row=1, column=0, sticky="e")
        tk.Entry(master, textvariable=self.path_ritaglio, width=60).grid(row=1, column=1)
        tk.Button(master, text="Sfoglia", command=self.carica_ritaglio).grid(row=1, column=2)

        tk.Button(master, text="Verifica", command=self.avvia_verifica, width=20).grid(row=2, column=1, pady=10)

        self.output = scrolledtext.ScrolledText(master, width=80, height=10)
        self.output.grid(row=3, column=0, columnspan=3, pady=10)

        # Riquadri immagini + etichette
        self.img_originale_label = tk.Label(master)
        self.img_originale_label.grid(row=4, column=0)
        tk.Label(master, text="Originale").grid(row=5, column=0)

        self.img_ritaglio_label = tk.Label(master)
        self.img_ritaglio_label.grid(row=4, column=1)
        tk.Label(master, text="Ritaglio").grid(row=5, column=1)

        self.img_overlay_label = tk.Label(master)
        self.img_overlay_label.grid(row=4, column=2)
        tk.Label(master, text="Overlay").grid(row=5, column=2)

        tk.Button(master, text="Genera Report PDF", command=self.salva_pdf).grid(row=6, column=1, pady=20)

        self.tk_img_originale = None
        self.tk_img_ritaglio = None
        self.tk_img_overlay = None

        self.overlay_pil = None
        self.log_text = ""


    def carica_originale(self):
        file = filedialog.askopenfilename(filetypes=[("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file:
            self.path_originale.set(file)

    def carica_ritaglio(self):
        file = filedialog.askopenfilename(filetypes=[("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file:
            self.path_ritaglio.set(file)


    def avvia_verifica(self):
        originale = self.path_originale.get().strip()
        ritaglio = self.path_ritaglio.get().strip()

        if not originale or not ritaglio:
            messagebox.showerror("Errore", "Selezionare entrambe le immagini.")
            return

        risultato, overlay, img_originale_cv, crop_cv = verifica_ritaglio(originale, ritaglio)

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, risultato)
        self.log_text = risultato

        # Originale
        pil_orig = Image.fromarray(img_originale_cv).resize((300, 300))
        self.tk_img_originale = ImageTk.PhotoImage(pil_orig)
        self.img_originale_label.config(image=self.tk_img_originale)

        # Ritaglio
        pil_crop = Image.fromarray(crop_cv).resize((200, 200))
        self.tk_img_ritaglio = ImageTk.PhotoImage(pil_crop)
        self.img_ritaglio_label.config(image=self.tk_img_ritaglio)

        # Overlay
        if overlay is not None:
            overlay_resized = overlay.resize((300, 300))
            self.tk_img_overlay = ImageTk.PhotoImage(overlay_resized)
            self.img_overlay_label.config(image=self.tk_img_overlay)
            self.overlay_pil = overlay
        else:
            self.img_overlay_label.config(image="", text="Nessun overlay")
            self.overlay_pil = None


    def salva_pdf(self):
        if not self.log_text:
            messagebox.showerror("Errore", "Prima eseguire una verifica.")
            return

        pdf_path = genera_report(
            self.log_text,
            self.path_originale.get(),
            self.path_ritaglio.get(),
            self.overlay_pil
        )

        messagebox.showinfo("PDF generato", f"Report salvato come:\n{pdf_path}")



# Avvio applicazione
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
