import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from Pylette import extract_colors

def color_histogram(img, channels=[0, 1, 2], bins=[16, 16, 16], ranges=[0, 256, 0, 256, 0, 256]):
    hist = cv2.calcHist(img, channels, None, bins, ranges)
    return hist.flatten()

def hsv_histogram(img, h_bins=18, s_bins=8, v_bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    return hist.flatten()

def lbp_texture_features(img, P=8, R=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    return lbp_hist

def find_mean(img):
    return np.mean(img, axis=(0, 1))

def find_stddev(img):
    return np.std(img, axis=(0, 1))

def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def shannon_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    return entropy

def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,2])

def green_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return np.sum(mask > 0) / mask.size


def color_palette(img, palette_size=5):
    """Extract dominant colors - reduced palette size for better generalization"""
    try:
        colors = extract_colors(image = img, palette_size=palette_size, mode="KMeans") # type: ignore
        rgb_array = np.array([color.rgb for color in colors], dtype=np.float64)  # shape Nx3
        freq_array = np.array([color.freq for color in colors], dtype=np.float64)  # shape N
        freq_array /= freq_array.sum()

        # Normalize RGB values to 0-1 range for better feature scaling
        rgb_array = rgb_array / 255.0
        
        # Interleave RGB and frequency values
        combined_features = []
        for i in range(len(colors)):
            combined_features.extend([rgb_array[i][0], rgb_array[i][1], rgb_array[i][2], freq_array[i]])
        
        # Pad with zeros if fewer colors found than expected
        expected_length = palette_size * 4
        while len(combined_features) < expected_length:
            combined_features.extend([0, 0, 0, 0])
            
        return np.array(combined_features[:expected_length])
    except Exception:
        # Fallback in case color extraction fails
        return np.zeros(palette_size * 4)

def hsv_palette(img, palette_size=5):
    """Extract HSV color palette - better for distinguishing biome colors"""
    try:
        colors = extract_colors(image = img, palette_size=palette_size, mode="KMeans") # type: ignore
        hsv_array = np.array([color.hsv for color in colors], dtype=np.float64)  # shape Nx3
        freq_array = np.array([color.freq for color in colors], dtype=np.float64)  # shape N
        freq_array /= freq_array.sum()
        
        # Normalize HSV values: H (0-360), S (0-100), V (0-100)
        hsv_array[:, 0] = hsv_array[:, 0] / 360.0  # Hue to 0-1
        hsv_array[:, 1] = hsv_array[:, 1] / 100.0  # Saturation to 0-1
        hsv_array[:, 2] = hsv_array[:, 2] / 100.0  # Value to 0-1
        
        combined_features = []
        for i in range(len(colors)):
            combined_features.extend([hsv_array[i][0], hsv_array[i][1], hsv_array[i][2], freq_array[i]])
        
        # Pad with zeros if fewer colors found than expected
        expected_length = palette_size * 4
        while len(combined_features) < expected_length:
            combined_features.extend([0, 0, 0, 0])
            
        return np.array(combined_features[:expected_length])
    except Exception:
        # Fallback in case color extraction fails
        return np.zeros(palette_size * 4)

def water_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue range for water detection
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return np.sum(mask > 0) / mask.size

def sand_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Yellow/beige range for sand detection
    lower_sand = np.array([15, 30, 100])
    upper_sand = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_sand, upper_sand)
    return np.sum(mask > 0) / mask.size

def snow_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # White/light gray range for snow detection
    lower_snow = np.array([0, 0, 200])
    upper_snow = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_snow, upper_snow)
    return np.sum(mask > 0) / mask.size


def red_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red range (considering HSV wrapping)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    return np.sum(mask > 0) / mask.size



def texture_complexity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def color_diversity_index(img):
    # Convert to LAB color space for better perceptual uniformity
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Calculate standard deviation across all channels
    return np.mean([np.std(lab[:,:,i]) for i in range(3)])

def dominant_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    return np.argmax(hue_hist)

def saturation_mean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,1])

def biome_specific_color_ratios(img):
    water_ratio = water_pixel_ratio(img)
    green_ratio = green_pixel_ratio(img)
    sand_ratio = sand_pixel_ratio(img)
    snow_ratio = snow_pixel_ratio(img)
    red_ratio = red_pixel_ratio(img)

    return np.array([water_ratio, green_ratio, sand_ratio, snow_ratio, red_ratio])

def block_pattern_detection(img, block_size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Divide image into blocks and calculate variance within each block
    block_variances = []
    for i in range(0, h-block_size, block_size):
        for j in range(0, w-block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            block_variances.append(np.var(block))
    
    return np.array([np.mean(block_variances), np.std(block_variances)])

