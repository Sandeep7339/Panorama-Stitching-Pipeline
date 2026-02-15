import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Initialize SIFT detector
        # SIFT is invariant to scale and rotation, making it ideal for stitching
        self.sift = cv2.SIFT_create()

    def detect_and_describe(self, image):
        """
        Detect keypoints and compute descriptors.
        """
        # Convert to grayscale as SIFT works on single channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        """
        Match features between two images using KNN (K-Nearest Neighbors).
        Includes Lowe's Ratio Test for robustness.
        """
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        
        # k=2 asks for the top 2 matches for every descriptor
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        # Apply Lowe's Ratio Test
        # If the best match is significantly better than the second best, keep it.
        # This removes ambiguous matches (noise).
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        return good_matches