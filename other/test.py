import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from joblib import Parallel, delayed

def load_image(filepath, resize=None):
    """Load and optionally resize the image."""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if resize:
        image = cv2.resize(image, resize)
    return image

def quantize_image(image, levels=64):
    """Reduce the number of gray levels in the image."""
    return (image / (256 // levels)).astype(np.uint8)

def compute_glcm_features(patch, distances, angles, levels):
    """Compute GLCM features for a single patch."""
    glcm = graycomatrix(patch, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    features = {
        "contrast": graycoprops(glcm, 'contrast').mean(),
        "correlation": graycoprops(glcm, 'correlation').mean(),
        "energy": graycoprops(glcm, 'energy').mean(),
        "homogeneity": graycoprops(glcm, 'homogeneity').mean()
    }
    return features

def extract_glcm_features(image, distances, angles, levels, window_size):
    """
    Perform GLCM computation with a single window size.
    """
    height, width = image.shape
    heatmaps = {
        "contrast": np.zeros((height, width), dtype=np.float32),
        "correlation": np.zeros((height, width), dtype=np.float32),
        "energy": np.zeros((height, width), dtype=np.float32),
        "homogeneity": np.zeros((height, width), dtype=np.float32)
    }

    # Extract features for the entire image with a single window size
    patches = [
        (image[y:y + window_size, x:x + window_size], distances, angles, levels)
        for y in range(0, height - window_size + 1, window_size)
        for x in range(0, width - window_size + 1, window_size)
    ]
    
    results = Parallel(n_jobs=-1)(delayed(compute_glcm_features)(patch[0], patch[1], patch[2], patch[3]) for patch in patches)
    
    # Apply the feature extraction to the corresponding heatmap regions
    idx = 0
    for y in range(0, height - window_size + 1, window_size):
        for x in range(0, width - window_size + 1, window_size):
            heatmaps["contrast"][y:y + window_size, x:x + window_size] += results[idx]['contrast']
            heatmaps["correlation"][y:y + window_size, x:x + window_size] += results[idx]['correlation']
            heatmaps["energy"][y:y + window_size, x:x + window_size] += results[idx]['energy']
            heatmaps["homogeneity"][y:y + window_size, x:x + window_size] += results[idx]['homogeneity']
            idx += 1

    return heatmaps

def show_heatmaps(heatmaps):
    """Display heatmaps for each feature."""
    feature_names = ["contrast", "correlation", "energy", "homogeneity"]
    for feature in feature_names:
        # Normalize heatmap for better visualization
        normed_heatmap = cv2.normalize(heatmaps[feature], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(f"{feature.capitalize()} Heatmap", normed_heatmap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    filepath = 'images\\captured\\original\\original (20241112_144628).jpg'  # Replace with your image path
    image = load_image(filepath)

    # Quantize the image to reduce gray levels
    quantized_image = quantize_image(image, levels=64)

    # Set a single window size to use
    window_size = 8  # Fixed window size for feature extraction
    
    # Generate heatmaps for all GLCM features
    heatmaps = extract_glcm_features(quantized_image, distances=[1], angles=[0], levels=64, window_size=window_size)
    
    # Show the heatmaps
    show_heatmaps(heatmaps)

if __name__ == "__main__":
    main()
