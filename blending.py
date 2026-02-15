import cv2
import numpy as np

class ImageBlender:
    def warp_images(self, img1, img2, H):
        """
        Warps img1 to align with img2 using Homography H.
        Returns the warped img1 and img2 placed on a large canvas.
        """
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        # Get the corners of img1 (to be warped)
        list_of_points_1 = np.float32([[0,0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        
        # Get the corners of img2 (reference)
        # We need to map img1 corners to the destination plane
        # Then calculate the size of the canvas needed to hold both
        list_of_points_2 = np.float32([[0,0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        
        # Transform img1 corners
        list_of_points_1_transformed = cv2.perspectiveTransform(list_of_points_1, H)
        
        # Combine all points to find the bounding box
        list_of_points = np.concatenate((list_of_points_1_transformed, list_of_points_2), axis=0)
        
        # Find the min and max x, y coordinates
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
        
        # Translation matrix to shift the image to positive coordinates
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        
        # Output image size
        output_shape = (x_max - x_min, y_max - y_min)
        
        # Warp img1
        warped_img1 = cv2.warpPerspective(img1, H_translation.dot(H), output_shape)
        
        # Place img2 on the canvas
        panorama_ref = np.zeros_like(warped_img1)
        # Create a region for img2
        panorama_ref[translation_dist[1]:rows2+translation_dist[1], translation_dist[0]:cols2+translation_dist[0]] = img2
        
        return warped_img1, panorama_ref, None

    def seamless_blend(self, warped_img1, panorama_ref):
        """
        Blends the warped image and the reference image.
        Simple alpha blending or max operator for now.
        For better results, multi-band blending can be used.
        """
        # Simple approach: where panorama_ref is black, use warped_img1. Where warped_img1 is black, use panorama_ref.
        # Where both overlap, we take the max (or average).
        
        # Create masks
        mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0)
        mask2 = (cv2.cvtColor(panorama_ref, cv2.COLOR_BGR2GRAY) > 0)
        
        # Use simple blending: if pixel in both, take max (usually preserves features better than average which might blur due to misalignment)
        # Alternatively, we can use a linear blend.
        
        # Using Max for simplicity and sharpness
        # Convert to float for blending if needed, but for max, uint8 is fine
        result = np.maximum(warped_img1, panorama_ref)
        
        return result

    def crop_black_borders(self, img):
        """
        Crops the largest interior axis-aligned rectangle from the image
        to remove black borders/corners using the Largest Rectangle in Histogram algorithm.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 1. First, crop to the bounding rect of the non-black content
        # This reduces the search space for the expensive algorithm
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
            
        c = max(contours, key=cv2.contourArea)
        x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(c)
        
        # Crop the image and the mask
        img_cropped = img[y_bound:y_bound+h_bound, x_bound:x_bound+w_bound]
        thresh_cropped = thresh[y_bound:y_bound+h_bound, x_bound:x_bound+w_bound]
        
        # 2. Find the largest interior rectangle in the cropped mask
        # Convert mask to 0 and 1
        matrix = (thresh_cropped > 0).astype(np.int32)
        rows, cols = matrix.shape
        
        # Array to hold height of histogram bars
        heights = np.zeros(cols, dtype=np.int32)
        
        max_area = 0
        best_rect = (0, 0, 0, 0) # x, y, w, h
        
        for r in range(rows):
            # Update heights: if pixel is 1, height increases. If 0, resets to 0.
            heights = (heights + 1) * matrix[r]
            
            # specific logic to find max area in this histogram
            area, rect = self._largest_rectangle_area(heights)
            
            if area > max_area:
                max_area = area
                # The histogram calculation returns width and x-offset
                # The height of the rectangle is what limited the bar, 
                # but we need to know the height of the rectangle that gave this area.
                # Actually, my helper returns area and (x, w).
                # The height of this rectangle implies it ends at row 'r'.
                # But what is its height? 
                # The helper needs to give us the height used for that area.
                
                # Let's verify the helper logic.
                rect_x, rect_y, rect_w, rect_h = rect
                # The helper 'rect_y' would be relative to the histogram base?
                # No, let's just make the helper return (h, x, w) corresponding to the max area.
                
                # Correct Logic:
                # The rectangle ends at current row 'r'.
                # The height is rect_h.
                # So top-left y is r - rect_h + 1.
                
                best_rect = (rect_x, r - rect_h + 1, rect_w, rect_h)

        bx, by, bw, bh = best_rect
        return img_cropped[by:by+bh, bx:bx+bw]

    def _largest_rectangle_area(self, heights):
        """
        Helper to find largest rectangle area in a histogram.
        Returns max_area, (x, y_relative, w, h)
        """
        stack = [-1] # Stack stores indices
        max_area = 0
        best_rect = (0, 0, 0, 0) # x, y (relative to 0?? No, y is just 0 for histogram), w, h
        
        # Append 0 to flush stack
        heights_extended = np.append(heights, 0)
        
        for i, h in enumerate(heights_extended):
            while stack[-1] != -1 and heights_extended[stack[-1]] >= h:
                height = heights_extended[stack.pop()]
                width = i - stack[-1] - 1
                area = height * width
                
                if area > max_area:
                    max_area = area
                    # x start is stack[-1] + 1
                    best_rect = (stack[-1] + 1, 0, width, height)
                    
            stack.append(i)
            
        return max_area, best_rect