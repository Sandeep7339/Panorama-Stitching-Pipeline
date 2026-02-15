# Panorama Image Stitching Pipeline

A modular Computer Vision pipeline built from scratch to stitch multiple overlapping images into a seamless panoramic view.

## 🚀 Key Features
- **SIFT Feature Detection**: Robust keypoint extraction.
- **Lowe's Ratio Test**: Filtering for high-quality feature correspondences.
- **RANSAC Homography**: Outlier-resilient geometric transformation.
- **Weighted Alpha Blending**: Seamless transitions and intensity normalization.
- **Auto-Rectangular Cropping**: Removes black borders to produce a clean final image.

## 📁 Project Structure
- `main.py`: Main entry point (coordinates the pipeline).
- `features.py`: SIFT detection and matching logic.
- `homography.py`: RANSAC and transformation matrix estimation.
- `blending.py`: Image warping, seamless blending, and final cropping.

## 🛠️ Usage
1. Place your images in the root directory (e.g., `img_left.jpeg`, `img_center.jpeg`, `img_right.jpeg`).
2. Run the pipeline:
   ```bash
   python main.py