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


def template_matching_multiscale(img_color, crop_color, threshold=0.8):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)

    (tH, tW) = crop_gray.shape[:2]
    found = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = cv2.resize(img_gray, (int(img_gray.shape[1] * scale), int(img_gray.shape[0] * scale)))
        r = img_gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        result = cv2.matchTemplate(resized, crop_gray, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    if found and found[0] > threshold:
        (maxVal, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        return (startX, startY, endX, endY), maxVal
    return None, None


# ============================================================
# FUNZIONE PRINCIPALE DI MATCHING
# ============================================================
def verifica_ritaglio(path_originale, path_ritaglio):

    # Carica immagini a colori
    img_color = cv2.imread(path_originale, cv2.IMREAD_COLOR)
    crop_color = cv2.imread(path_ritaglio, cv2.IMREAD_COLOR)

    if img_color is None:
        return "Errore: impossibile caricare l'immagine originale.", None, None, None, None
    if crop_color is None:
        return "Errore: impossibile caricare il ritaglio.", None, None, img_color, crop_color

    # Versioni in grayscale solo per ORB
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    crop = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)

    # Tentativo con SIFT e FLANN
    log = []
    sift_success = False
    overlay_pil = None
    overlay_rgb = None

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(crop, None)

    if des1 is not None and des2 is not None and len(kp1) >= 2 and len(kp2) >= 2:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des2, des1, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        log.append(f"SIFT - Match trovati: {len(matches)}")
        log.append(f"SIFT - Match buoni: {len(good)}")

        if len(good) >= 10:
            src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                h, w = crop.shape
                corners_crop = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
                corners_transformed = cv2.perspectiveTransform(corners_crop, H)

                overlay = img_color.copy()
                cv2.polylines(overlay, [np.int32(corners_transformed)], True, (0,255,0), 3)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                overlay_pil = Image.fromarray(overlay_rgb)

                log.append("SIFT - Omografia valida.")
                log.append("RISULTATO: Il ritaglio APPARTIENE alla foto (metodo SIFT).")
                sift_success = True
    
    if not sift_success:
        log.append("SIFT non ha trovato un match sufficiente o omografia fallita. Tentativo con Template Matching Multi-scala...")
        
        bbox, confidence = template_matching_multiscale(img_color, crop_color)
        
        if bbox is not None:
            (startX, startY, endX, endY) = bbox
            overlay = img_color.copy()
            cv2.rectangle(overlay, (startX, startY), (endX, endY), (0, 0, 255), 3) # Rosso per Template Matching
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            
            log.append(f"Template Matching - Trovato con confidenza: {confidence:.2f}")
            log.append("RISULTATO: Il ritaglio APPARTIENE alla foto (metodo Template Matching).")
        else:
            log.append("Template Matching Multi-scala non ha trovato un match sufficiente.")
            log.append("RISULTATO: Il ritaglio NON appartiene alla foto.")

    return "\n".join(log), overlay_pil, img_color, crop_color, overlay_rgb



# ============================================================
# REPORT PDF
# ============================================================
def genera_report(log_text, path_originale, path_ritaglio, overlay_image, pdf_path):



    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("<b>Report verifica ritaglio in foto</b>", styles["Title"]))
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("<b>Risultato e Analisi</b>", styles["Heading2"]))
    flow.append(Paragraph(log_text.replace("\n", "<br/>"), styles["BodyText"]))
    flow.append(Spacer(1, 20))

    # Salvataggio immagini temporanee
    img1 = "tmp_originale.jpg"
    img2 = "tmp_ritaglio.jpg"
    img3 = "tmp_overlay.jpg"

    Image.open(path_originale).save(img1)
    Image.open(path_ritaglio).save(img2)

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

    # Pulizia
    for file in [img1, img2, img3]:
        if os.path.exists(file):
            os.remove(file)

    return pdf_path



# ============================================================
# GUI TKINTER
# ============================================================
class App:
    def __init__(self, master):
        self.master = master
        master.title("Verifica Ritaglio in Foto")

        self.path_originale = tk.StringVar()
        self.path_ritaglio = tk.StringVar()

        # Form selezione file
        tk.Label(master, text="Foto Originale:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.path_originale, width=60).grid(row=0, column=1)
        tk.Button(master, text="Sfoglia", command=self.carica_originale).grid(row=0, column=2)

        tk.Label(master, text="Ritaglio:").grid(row=1, column=0, sticky="e")
        tk.Entry(master, textvariable=self.path_ritaglio, width=60).grid(row=1, column=1)
        tk.Button(master, text="Sfoglia", command=self.carica_ritaglio).grid(row=1, column=2)

        tk.Button(master, text="Verifica", command=self.avvia_verifica, width=20).grid(row=2, column=1, pady=10)

        # Log
        self.output = scrolledtext.ScrolledText(master, width=80, height=10)
        self.output.grid(row=3, column=0, columnspan=3, pady=10)

        # Aree per le immagini + etichette
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

        self.overlay_pil = None
        self.log_text = ""
        self.orig_color = None
        self.crop_color = None


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

        risultato, overlay_pil, orig_color, crop_color, overlay_rgb = verifica_ritaglio(originale, ritaglio)

        self.overlay_pil = overlay_pil
        self.orig_color = orig_color
        self.crop_color = crop_color

        # Log
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, risultato)
        self.log_text = risultato

        # Mostra immagini a colori
        if orig_color is not None:
            img_rgb = cv2.cvtColor(orig_color, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb).resize((300,300))
            tk_img = ImageTk.PhotoImage(pil)
            self.img_originale_label.config(image=tk_img)
            self.img_originale_label.image = tk_img

        if crop_color is not None:
            img_rgb = cv2.cvtColor(crop_color, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb).resize((200,200))
            tk_img = ImageTk.PhotoImage(pil)
            self.img_ritaglio_label.config(image=tk_img)
            self.img_ritaglio_label.image = tk_img

        if overlay_pil is not None:
            pil = overlay_pil.resize((300,300))
            tk_img = ImageTk.PhotoImage(pil)
            self.img_overlay_label.config(image=tk_img)
            self.img_overlay_label.image = tk_img
        else:
            self.img_overlay_label.config(image="", text="Nessun overlay")


    def salva_pdf(self):

        if not self.log_text:
            messagebox.showerror("Errore", "Prima eseguire una verifica.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"report_verifica_{timestamp}.pdf"

        file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                filetypes=[("PDF files", "*.pdf")],
                                                title="Salva Report PDF",
                                                initialfile=default_filename)
        if not file_path:
            return

        genera_report(
            self.log_text,
            self.path_originale.get(),
            self.path_ritaglio.get(),
            self.overlay_pil,
            file_path
        )

        messagebox.showinfo("Fatto", f"Report salvato come:\n{file_path}")



# ============================================================
# START PROGRAMMA
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
