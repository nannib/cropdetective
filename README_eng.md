![Screenshot](screenshot_eng.jpg)

# Verify if a Crop Belongs to a Photo (GUI + PDF Report)

This program allows you to verify if a cropped image actually comes from an original photo.
The verification is based on reliable and scientifically valid computer vision techniques, used in forensic, academic, and industrial fields.

The program provides:
- Graphical Interface (Tkinter)
- Side-by-side display of:
  - Original Photo
  - Crop
  - Overlay of the found crop in the photo
- Automatic PDF Report with timestamp, images, and verification details

---

# How it Works

The program now uses a hybrid and robust strategy to verify if a crop belongs to a photo, combining different computer vision algorithms. The verification logic proceeds as follows:

1.  **Main attempt with SIFT (Scale-Invariant Feature Transform) and FLANN (Fast Library for Approximate Nearest Neighbors):**
    *   **SIFT:** This algorithm is more advanced than ORB and particularly effective in detecting keypoints that are invariant to scale, rotation, and lighting variations. It generates more robust descriptors, making it ideal for crops that might have been resized or rotated.
    *   **FLANN:** Used for efficient matching of SIFT descriptors. It is optimized to quickly find the best matches among a large number of descriptors.
    *   If SIFT and FLANN find a sufficient number of "good matches" and can calculate a valid homography, the crop is considered to belong to the photo.

2.  **Fallback with Multi-scale Template Matching:**
    *   If the SIFT/FLANN method does not produce sufficient results (e.g., few keypoints detected in the crop, as in the case of very small or uniform crops, or failure in homography calculation), the program attempts an alternative approach.
    *   Multi-scale Template Matching searches for the crop in the original image by trying different crop sizes. This is particularly useful for crops that are very similar to the original but might have been slightly resized, or for crops with little "texture" where keypoint-based algorithms struggle.
    *   If Template Matching finds a match with high confidence, the crop is considered to belong to the photo.

3.  **Final Result:**
    *   If neither method (SIFT/FLANN or Template Matching) can find a reliable match, the program concludes that the crop *does not* belong to the photo.

---

## Technical Details of the Algorithms Used:

### 1. SIFT (Scale-Invariant Feature Transform)

SIFT is an algorithm that allows the identification of **keypoints** in an image. These points are highly informative areas (e.g., corners, specific edges) that remain recognizable even after transformations such as:

-   Rotations
-   Lighting variations
-   Significant scale changes

SIFT also generates **descriptors**, which are numerical vectors that locally represent the image structure. By comparing the descriptors of the crop and the full photo, the program searches for correspondences.

### 2. Feature Matching (with FLANN)

The program uses a comparison based on the distance between descriptors. For SIFT descriptors (which are floating-point), FLANN is an optimized matcher that accelerates the search for "nearest neighbors" among descriptors, making the matching process much faster and more efficient than a brute-force matcher.

A match is considered reliable if the distance between descriptors is sufficiently low compared to other possible pairings (Lowe's Ratio Test).

### 3. Homography

Homography is a geometric transformation that describes how one plane is mapped to another plane. It allows modeling of:

-   Rotations
-   Translations
-   Perspective changes
-   Resizing

If a consistent homography exists between the crop and the photo, it means that the crop can be geometrically positioned within the photo.

### 4. Multi-scale Template Matching

This method directly compares the crop (template) with different portions of the original image at various scales. It is effective when the crop is an almost exact copy of a part of the image but may be present at slightly different sizes. The program searches for the position and scale that maximize the correlation between the crop and the original image.

---

# Scientific Validity

The techniques used are standard in the field of computer vision and are the basis of systems for:

-   Image recognition
-   Digital forensic analysis
-   Visual robotics
-   3D reconstruction from images
-   Object tracking

The hybrid SIFT/FLANN + Multi-scale Template Matching approach is robust and reliable for verifying crop belonging, even in the presence of small, low-resolution, or low-texture crops.

---

# Dependencies

The program requires:

- opencv-python
- Pillow
- reportlab

See `requirements.txt`.

If you have problems with Tkinter, on Windows the package comes with the Python installation (https://www.python.org/)

---

# How to Run

python cutdetective.py
