import cv2
import numpy as np

def crop_borders(image):
    """
    Crops the image to remove black borders/artifacts.
    
    Strategy:
    1. Crop to the Bounding Box of valid pixels (removes the large black canvas).
    2. Iteratively shrink inwards from all sides until no black pixels remain on the edges.
    3. Safety Check: If shrinking removes too much (>50%) or creates an empty image,
       it falls back to the Bounding Box crop to prevent crashing.
    """
    if image is None: return None
    
    # Convert to grayscale and threshold to find non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 1. Find the main Bounding Box (The "Safe" Crop)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image # Return original if no content found
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Perform the initial safe crop
    crop = image[y:y+h, x:x+w]
    thresh_crop = thresh[y:y+h, x:x+w]
    
    h_crop, w_crop = crop.shape[:2]
    original_area = h_crop * w_crop
    
    # 2. Iterative Shrinking (The "Strict" Crop)
    top = 0
    bottom = h_crop - 1
    left = 0
    right = w_crop - 1
    
    # Shrink Top
    while top < bottom and np.any(thresh_crop[top, left:right] == 0):
        top += 1
    
    # Shrink Bottom
    while bottom > top and np.any(thresh_crop[bottom, left:right] == 0):
        bottom -= 1
        
    # Shrink Left
    while left < right and np.any(thresh_crop[top:bottom, left] == 0):
        left += 1
        
    # Shrink Right
    while right > left and np.any(thresh_crop[top:bottom, right] == 0):
        right -= 1

    # 3. Calculate new dimensions
    new_h = bottom - top
    new_w = right - left
    new_area = new_h * new_w

    # 4. SAFETY CHECK:
    # If the strict crop creates an invalid image or deletes > 50% of the content,
    # we REJECT the strict crop and return the Safe Bounding Box instead.
    if new_h <= 0 or new_w <= 0 or new_area < (original_area * 0.5):
        print("    [Cropper] Warning: Irregular borders detected. Using standard crop to preserve image content.")
        return crop
    
    # Otherwise, return the clean strict crop
    return crop[top:bottom+1, left:right+1]