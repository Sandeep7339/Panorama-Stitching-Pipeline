import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from features import FeatureExtractor
from homography import HomographyEstimator
from blending import ImageBlender

def show_image(title, img):
    """Displays image in RGB for Matplotlib compatibility."""
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def crop_panorama(img):
    """
    Handles irregular borders by finding the largest internal rectangular area.
    This removes the black 'wedges' seen in warped results.
    """
    # Create a binary mask of the non-black pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Get initial bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Iteratively shrink the rectangle until no black pixels remain inside
    while cv2.countNonZero(thresh[y:y+h, x:x+w]) < (w * h):
        x += 1
        y += 1
        w -= 2
        h -= 2
        if w <= 0 or h <= 0:
            break

    return img[y:y+h, x:x+w]

def main():
    fe = FeatureExtractor()
    he = HomographyEstimator()
    ib = ImageBlender()

    # Configuration ( filenames from your previous code)
    img_names = ['img_left.jpeg', 'img_center.jpeg', 'img_right.jpeg']
    images = []

    for name in img_names:
        img = cv2.imread(name)
        if img is None:
            print(f"Error: Could not load {name}")
            sys.exit(1)
        # Resize to 50% for speed
        h, w = img.shape[:2]
        images.append(cv2.resize(img, (int(w * 0.5), int(h * 0.5))))

    img_left, img_center, img_right = images

    # --- STAGE 1: Left -> Center ---
    print("=== Processing Stage 1: Left -> Center ===")
    kp_l, des_l = fe.detect_and_describe(img_left)
    kp_c, des_c = fe.detect_and_describe(img_center)
    m1 = fe.match_features(des_l, des_c)
    H1, _ = he.estimate_homography(kp_l, kp_c, m1)
    
    if H1 is None: return

    warped_l, ref_c, _ = ib.warp_images(img_left, img_center, H1)
    pan_stage_1 = ib.seamless_blend(warped_l, ref_c)

    # --- STAGE 2: Right -> Stage 1 Result ---
    print("=== Processing Stage 2: Right -> Stage 1 Result ===")
    kp_r, des_r = fe.detect_and_describe(img_right)
    kp_p1, des_p1 = fe.detect_and_describe(pan_stage_1)
    m2 = fe.match_features(des_r, des_p1)
    H2, _ = he.estimate_homography(kp_r, kp_p1, m2)
    
    if H2 is not None:
        warped_r, pan_ref, _ = ib.warp_images(img_right, pan_stage_1, H2)
        
        # --- A. NAIVE STITCH (NO CROPPING) ---
        print("Generating Naive Stitch (Raw)...")
        mask = np.any(warped_r != [0, 0, 0], axis=-1)
        naive_raw = pan_ref.copy()
        naive_raw[mask] = warped_r[mask]
        
        cv2.imwrite('naive_stitch_raw.jpg', naive_raw)

        # --- B. FINAL SEAMLESS STITCH (WITH REFINED CROPPING) ---
        print("Generating Seamless Stitch...")
        final_raw = ib.seamless_blend(warped_r, pan_ref)
        
        print("Cropping Final Stitch to remove irregular borders...")
        final_cropped = crop_panorama(final_raw)
        cv2.imwrite('final_panorama_cropped.jpg', final_cropped)
        
        print("Done!")
        show_image("Naive Stitch (Raw)", naive_raw)
        show_image("Final Result (Cropped)", final_cropped)
    else:
        print("Stage 2 Homography failed.")

if __name__ == "__main__":
    main()