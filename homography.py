import cv2
import numpy as np

class HomographyEstimator:
    def __init__(self):
        # Minimum matches required to attempt a stitch
        self.MIN_MATCH_COUNT = 10

    def estimate_homography(self, kp1, kp2, good_matches):
        """
        Calculates the Homography matrix using RANSAC.
        kp1: Keypoints from Image 1 (Query)
        kp2: Keypoints from Image 2 (Train)
        """
        if len(good_matches) > self.MIN_MATCH_COUNT:
            # Extract location of good matches
            # .reshape(-1, 1, 2) is required format for cv2.findHomography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find Homography
            # cv2.RANSAC is the Robust Estimation method.
            # 5.0 is the RANSAC reprojection threshold (in pixels).
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            return M, mask
        else:
            print(f"Not enough matches are found - {len(good_matches)}/{self.MIN_MATCH_COUNT}")
            return None, None