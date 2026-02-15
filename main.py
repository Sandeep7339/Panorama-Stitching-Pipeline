import cv2
import matplotlib.pyplot as plt
from features import FeatureExtractor
from homography import HomographyEstimator
from blending import ImageBlender

def show_image(title, img):
    """Helper to display images using Matplotlib (better for notebooks/scripts)"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def stitch_two_images(img1, img2, fe, he, ib):
    """
    Stitches img1 (Left) onto img2 (Right/Reference).
    Returns the stitched image.
    """
    print("--- Stitching Pair ---")
    
    # 1. Feature Extraction
    print("Detecting Features...")
    kp1, des1 = fe.detect_and_describe(img1)
    kp2, des2 = fe.detect_and_describe(img2)
    
    # 2. Feature Matching
    print("Matching Features...")
    matches = fe.match_features(des1, des2)
    print(f"Found {len(matches)} good matches.")
    
    # 3. Geometric Transformation (Homography)
    print("Estimating Homography...")
    H, mask = he.estimate_homography(kp1, kp2, matches)
    
    if H is None:
        print("Stitching failed: Could not estimate Homography.")
        return None

    # 4. Warping
    print("Warping Images...")
    warped_left, panorama_ref, _ = ib.warp_images(img1, img2, H)
    
    # 5. Blending (Artifact Mitigation)
    print("Blending...")
    result = ib.seamless_blend(warped_left, panorama_ref)
    
    return result

def main():
    # Initialize classes
    fe = FeatureExtractor()
    he = HomographyEstimator()
    ib = ImageBlender()

    # Load Images
    # Ensure you put the correct filenames here
    img_left = cv2.imread('img_left.jpeg')
    img_center = cv2.imread('img_center.jpeg')
    img_right = cv2.imread('img_right.jpeg')

    # Resize images slightly to speed up processing (optional but recommended for huge phone photos)
    scale_percent = 50 
    width = int(img_left.shape[1] * scale_percent / 100)
    height = int(img_left.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    img_left = cv2.resize(img_left, dim)
    img_center = cv2.resize(img_center, dim)
    img_right = cv2.resize(img_right, dim)

    print("Images loaded and resized.")

    # --- Iterative Processing ---
    # Strategy: 
    # 1. Stitch Left Image -> Center Image to create "Left_Center_Panorama"
    # 2. Stitch "Left_Center_Panorama" -> Right Image? 
    #    Actually, it's better to Stitch Right -> Left_Center_Panorama.
    #    The result of (1) becomes the new "Reference".
    
    # Step 1: Stitch Left to Center
    pan_stage_1 = stitch_two_images(img_left, img_center, fe, he, ib)
    
    if pan_stage_1 is not None:
        # Step 2: Stitch Right to the result of Step 1
        # Note: The result of stage 1 is now the "Left" side relative to the "Right" image.
        # However, purely geometrically, we want to map the Right image onto the Stage 1 Panorama.
        # So we treat Right as "img1" (source) and Stage 1 as "img2" (destination/ref).
        
        final_panorama = stitch_two_images(img_right, pan_stage_1, fe, he, ib)
        
        if final_panorama is not None:
            # Final Cleanup
            final_panorama = ib.crop_black_borders(final_panorama)
            
            # Save and Show
            cv2.imwrite('final_panorama.jpg', final_panorama)
            show_image("Final Panorama", final_panorama)
            print("Panorama stitching complete. Saved as 'final_panorama.jpg'")
        else:
            print("Second stage stitching failed.")
    else:
        print("First stage stitching failed.")

if __name__ == "__main__":
    main()