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
# MAIN MATCHING FUNCTION
# ============================================================
def verify_crop(original_path, crop_path):

    # Load color images
    original_color = cv2.imread(original_path, cv2.IMREAD_COLOR)
    crop_color_image = cv2.imread(crop_path, cv2.IMREAD_COLOR)

    if original_color is None:
        return "Error: unable to load original image.", None, None, None, None
    if crop_color_image is None:
        return "Error: unable to load crop image.", None, None, original_color, crop_color_image

    # Grayscale versions only for feature detection (SIFT)
    img_gray = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_color_image, cv2.COLOR_BGR2GRAY)

    # Attempt with SIFT and FLANN
    log = []
    sift_success = False
    overlay_pil = None
    overlay_rgb = None

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    kp2, des2 = sift.detectAndCompute(crop_gray, None)

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

        log.append(f"SIFT - Matches found: {len(matches)}")
        log.append(f"SIFT - Good matches: {len(good)}")

        if len(good) >= 10:
            src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                h, w = crop_gray.shape
                corners_crop = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
                corners_transformed = cv2.perspectiveTransform(corners_crop, H)

                overlay = original_color.copy()
                cv2.polylines(overlay, [np.int32(corners_transformed)], True, (0,255,0), 3)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                overlay_pil = Image.fromarray(overlay_rgb)

                log.append("SIFT - Valid homography found.")
                log.append("RESULT: The crop BELONGS to the photo (SIFT method).")
                sift_success = True
    
    if not sift_success:
        log.append("SIFT did not find enough matches or homography failed. Attempting Multi-scale Template Matching...")
        
        bbox, confidence = template_matching_multiscale(original_color, crop_color_image)
        
        if bbox is not None:
            (startX, startY, endX, endY) = bbox
            overlay = original_color.copy()
            cv2.rectangle(overlay, (startX, startY), (endX, endY), (0, 0, 255), 3) # Red for Template Matching
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            
            log.append(f"Template Matching - Found with confidence: {confidence:.2f}")
            log.append("RESULT: The crop BELONGS to the photo (Template Matching method).")
        else:
            log.append("Multi-scale Template Matching did not find enough matches.")
            log.append("RESULT: The crop DOES NOT belong to the photo.")

    return "\n".join(log), overlay_pil, original_color, crop_color_image, overlay_rgb



# ============================================================
# PDF REPORT
# ============================================================
def generate_report(log_message, original_path, crop_path, overlay_image, pdf_path):



    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("<b>Crop Verification Report in Photo</b>", styles["Title"]))
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("<b>Result and Analysis</b>", styles["Heading2"]))
    flow.append(Paragraph(log_message.replace("\n", "<br/>"), styles["BodyText"]))
    flow.append(Spacer(1, 20))

    # Save temporary images
    img1 = "tmp_original.jpg"
    img2 = "tmp_crop.jpg"
    img3 = "tmp_overlay.jpg"

    Image.open(original_path).save(img1)
    Image.open(crop_path).save(img2)

    if overlay_image is not None:
        overlay_image.save(img3)

    flow.append(Paragraph("Original Image:", styles["BodyText"]))
    flow.append(RLImage(img1, width=350, height=350 * 0.75))
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("Crop:", styles["BodyText"]))
    flow.append(RLImage(img2, width=200, height=200 * 0.75))
    flow.append(Spacer(1, 20))

    if overlay_image is not None:
        flow.append(Paragraph("Overlay (found crop):", styles["BodyText"]))
        flow.append(RLImage(img3, width=350, height=350 * 0.75))

    doc.build(flow)

    # Cleanup
    for file in [img1, img2, img3]:
        if os.path.exists(file):
            os.remove(file)

    return pdf_path



# ============================================================
# TKINTER GUI
# ============================================================
class App:
    def __init__(self, master):
        self.master = master
        master.title("Verify Crop in Photo")

        self.original_path = tk.StringVar()
        self.crop_path = tk.StringVar()

        # File selection form
        tk.Label(master, text="Original Photo:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.original_path, width=60).grid(row=0, column=1)
        tk.Button(master, text="Browse", command=self.load_original).grid(row=0, column=2)

        tk.Label(master, text="Crop:").grid(row=1, column=0, sticky="e")
        tk.Entry(master, textvariable=self.crop_path, width=60).grid(row=1, column=1)
        tk.Button(master, text="Browse", command=self.load_crop).grid(row=1, column=2)

        tk.Button(master, text="Verify", command=self.start_verification, width=20).grid(row=2, column=1, pady=10)

        # Log
        self.output = scrolledtext.ScrolledText(master, width=80, height=10)
        self.output.grid(row=3, column=0, columnspan=3, pady=10)

        # Areas for images + labels
        self.original_image_label = tk.Label(master)
        self.original_image_label.grid(row=4, column=0)
        tk.Label(master, text="Original").grid(row=5, column=0)

        self.crop_image_label = tk.Label(master)
        self.crop_image_label.grid(row=4, column=1)
        tk.Label(master, text="Crop").grid(row=5, column=1)

        self.overlay_image_label = tk.Label(master)
        self.overlay_image_label.grid(row=4, column=2)
        tk.Label(master, text="Overlay").grid(row=5, column=2)

        tk.Button(master, text="Generate PDF Report", command=self.save_pdf).grid(row=6, column=1, pady=20)

        self.overlay_pil = None
        self.log_message = ""
        self.original_color = None
        self.crop_color_image = None


    def load_original(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file:
            self.original_path.set(file)

    def load_crop(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file:
            self.crop_path.set(file)


    def start_verification(self):

        original_image_path = self.original_path.get().strip()
        crop_image_path = self.crop_path.get().strip()

        if not original_image_path or not crop_image_path:
            messagebox.showerror("Error", "Please select both images.")
            return

        result, overlay_pil, original_color, crop_color_image, overlay_rgb = verify_crop(original_image_path, crop_image_path)

        self.overlay_pil = overlay_pil
        self.original_color = original_color
        self.crop_color_image = crop_color_image

        # Log
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, result)
        self.log_message = result

        # Display color images
        if original_color is not None:
            img_rgb = cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb).resize((300,300))
            tk_img = ImageTk.PhotoImage(pil)
            self.original_image_label.config(image=tk_img)
            self.original_image_label.image = tk_img

        if crop_color_image is not None:
            img_rgb = cv2.cvtColor(crop_color_image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb).resize((200,200))
            tk_img = ImageTk.PhotoImage(pil)
            self.crop_image_label.config(image=tk_img)
            self.crop_image_label.image = tk_img

        if overlay_pil is not None:
            pil = overlay_pil.resize((300,300))
            tk_img = ImageTk.PhotoImage(pil)
            self.overlay_image_label.config(image=tk_img)
            self.overlay_image_label.image = tk_img
        else:
            self.overlay_image_label.config(image="", text="No overlay")


    def save_pdf(self):

        if not self.log_message:
            messagebox.showerror("Error", "Please perform a verification first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"verification_report_{timestamp}.pdf"

        file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                filetypes=[("PDF files", "*.pdf")],
                                                title="Save PDF Report",
                                                initialfile=default_filename)
        if not file_path:
            return

        generate_report(
            self.log_message,
            self.original_path.get(),
            self.crop_path.get(),
            self.overlay_pil,
            file_path
        )

        messagebox.showinfo("Done", f"Report saved as:\n{file_path}")



# ============================================================
# START PROGRAM
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
