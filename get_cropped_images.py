import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from google.colab import drive
import shutil
import time

def mount_drive():
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")

def is_white_background(crop, white_threshold=200, white_percentage_threshold=80):
    """
    Check if the crop is mostly white background.
    Returns True if the crop is predominantly white.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Count pixels that are above the white threshold
    white_pixels = np.sum(gray >= white_threshold)
    total_pixels = gray.shape[0] * gray.shape[1]
    white_percentage = (white_pixels / total_pixels) * 100

    return white_percentage >= white_percentage_threshold

def has_sufficient_contrast(crop, min_contrast=20):
    """
    Check if the crop has sufficient contrast (not just uniform color).
    Returns True if there's enough variation in the image.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Calculate standard deviation of pixel values
    std_dev = np.std(gray)

    # Also check the range of values
    pixel_range = np.max(gray) - np.min(gray)

    return std_dev >= min_contrast or pixel_range >= 50

def detect_leaves(image):
    """
    Detects dried leaves in herbarium specimens using multiple techniques.
    Returns a binary mask where leaf pixels are white.
    """
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Check if the image is mostly envelope/paper
    if is_envelope(image):
        # Return empty mask for envelope/paper images
        return np.zeros(gray.shape, dtype=np.uint8)

    # 1. Expanded HSV ranges for various leaf colors
    # Range 1: Brown/tan dried leaves
    lower_brown = np.array([8, 20, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Range 2: Green to yellow-green leaves
    lower_green = np.array([25, 15, 30])
    upper_green = np.array([85, 255, 220])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Range 3: Very dark leaves (almost black)
    lower_dark = np.array([0, 0, 10])
    upper_dark = np.array([180, 255, 80])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    # Range 4: Reddish-brown leaves
    lower_red_brown1 = np.array([0, 30, 30])
    upper_red_brown1 = np.array([15, 255, 180])
    mask_red_brown1 = cv2.inRange(hsv, lower_red_brown1, upper_red_brown1)

    # Range 5: Higher red range
    lower_red_brown2 = np.array([160, 30, 30])
    upper_red_brown2 = np.array([180, 255, 180])
    mask_red_brown2 = cv2.inRange(hsv, lower_red_brown2, upper_red_brown2)

    # 2. LAB color space detection (good for brown/tan colors)
    # A channel: green-red axis, B channel: blue-yellow axis
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Detect brownish colors in LAB space
    mask_lab = np.zeros_like(gray)
    mask_lab[(a_channel > 128) & (b_channel > 128)] = 255  # Reddish-yellowish areas

    # 3. Enhanced grayscale thresholding
    # Adaptive threshold for varying lighting
    mask_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 8)

    # Regular threshold for darker areas
    _, mask_gray_dark = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    _, mask_gray_light = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 4. Edge-based detection to catch leaf boundaries
    edges = cv2.Canny(gray, 30, 100)
    # Dilate edges to create regions
    kernel_edge = np.ones((3, 3), np.uint8)
    mask_edges = cv2.dilate(edges, kernel_edge, iterations=2)

    # 5. Texture-based detection using local standard deviation
    kernel_texture = np.ones((9, 9), np.float32) / 81
    gray_blur = cv2.filter2D(gray.astype(np.float32), -1, kernel_texture)
    gray_sqr_blur = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel_texture)
    texture_var = gray_sqr_blur - gray_blur**2

    # Areas with high texture variation (leaves have more texture than paper)
    _, mask_texture = cv2.threshold(texture_var.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

    # Combine all masks
    combined_mask = cv2.bitwise_or(mask_brown, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_dark)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red_brown1)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red_brown2)
    combined_mask = cv2.bitwise_or(combined_mask, mask_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_adaptive)
    combined_mask = cv2.bitwise_or(combined_mask, mask_edges)
    combined_mask = cv2.bitwise_or(combined_mask, mask_texture)

    # Enhanced white background removal
    bright_mask = cv2.inRange(gray, 220, 255)  # Very bright areas (likely paper)

    # Calculate local texture variance to identify uniform areas
    kernel = np.ones((15, 15), np.float32) / 225
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2

    # Areas with very low variance are likely uniform background
    _, low_texture_mask = cv2.threshold(local_var.astype(np.uint8), 10, 255, cv2.THRESH_BINARY_INV)

    # Combine bright areas with low texture - these are likely paper background
    background_mask = cv2.bitwise_and(bright_mask, low_texture_mask)

    # Remove background areas from the leaf mask
    combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(background_mask))

    # Clean up the mask with morphological operations
    # Remove noise
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

    # Fill gaps
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Additional cleanup - remove very small artifacts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50  # Reduced minimum area to catch smaller leaves
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            cv2.drawContours(mask, [contour], 0, 0, -1)  # Fill small contours with black

    return mask

def is_envelope(image):
    """
    Enhanced envelope detection with additional checks specifically for uniform tan paper/envelopes.
    Uses multiple criteria to identify envelopes with very high confidence.
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Color check - specific to tan/beige envelopes
    avg_h = np.mean(hsv[:, :, 0])
    avg_s = np.mean(hsv[:, :, 1])
    avg_v = np.mean(hsv[:, :, 2])
    std_h = np.std(hsv[:, :, 0])
    std_s = np.std(hsv[:, :, 1])
    std_v = np.std(hsv[:, :, 2])

    # Strict color check for tan envelope
    # Expanded hue range to capture more envelope variations
    color_check = (10 <= avg_h <= 40 and  # Wider hue range for tan/beige
                  avg_s < 80 and         # Low saturation
                  avg_v > 140 and        # High brightness
                  std_h < 15 and         # Low hue variation
                  std_s < 25 and         # Low saturation variation
                  std_v < 30)            # Low brightness variation

    # If fails color check and has high color variation, it's probably not an envelope
    if not color_check or std_h > 20 or std_s > 30 or std_v > 40:
        return False

    # 2. Texture uniformity check
    # Calculate texture features using gradient magnitudes
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Mean and std dev of gradient magnitudes (low for uniform surfaces like envelopes)
    mean_gradient = np.mean(gradient_magnitude)
    std_gradient = np.std(gradient_magnitude)

    # Envelopes have low gradient magnitudes (very uniform texture)
    texture_check = mean_gradient < 10 and std_gradient < 20

    # 3. Edge sparsity check (envelopes have very few edges)
    # Edge detection with lower threshold for sensitivity
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100

    # If more than 5% of pixels are edges, probably not a blank envelope
    edge_check = edge_density < 5.0

    # 4. Color histogram check - envelopes have very concentrated histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)  # Normalize

    # Calculate histogram concentration (envelopes have peaks in narrow ranges)
    sorted_hist = np.sort(hist.flatten())
    # Sum of top 10 histogram bins - higher for concentrated histograms
    hist_concentration = np.sum(sorted_hist[-10:])

    # Envelopes have concentrated histograms (few distinct colors)
    histogram_check = hist_concentration > 0.5

    # 5. Count significant contours - envelopes typically have very few
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for significant contours
    significant_contours = 0
    min_contour_area = image.shape[0] * image.shape[1] * 0.01  # 1% of image area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            significant_contours += 1

    contour_check = significant_contours < 3  # Envelopes typically have 0-2 significant contours

    # 6. Count of darker pixels - leaves typically have many darker pixels than envelope
    # Count pixels significantly darker than average brightness
    brightness_threshold = np.mean(gray) * 0.8  # 80% of mean brightness
    dark_pixel_ratio = np.sum(gray < brightness_threshold) / (gray.shape[0] * gray.shape[1])

    # If more than 15% pixels are dark, probably has plant material
    darkness_check = dark_pixel_ratio < 0.15

    # Combine checks - more stringent requirements
    # Must pass color check AND at least 3 of the 5 other checks
    secondary_checks = sum([texture_check, edge_check, histogram_check, contour_check, darkness_check])

    return color_check and secondary_checks >= 3

def calculate_leaf_percentage(crop, mask):
    """
    Calculate the percentage of leaf pixels in the crop with additional validation.
    """
    # First check if the crop is mostly white background
    if is_white_background(crop):
        return 0.0

    # Check if there's sufficient contrast in the image
    if not has_sufficient_contrast(crop):
        return 0.0

    # Calculate leaf pixels from mask
    leaf_pixels = np.sum(mask == 255)
    total_pixels = crop.shape[0] * crop.shape[1]

    if total_pixels == 0:
        return 0.0

    leaf_percentage = (leaf_pixels / total_pixels) * 100

    # Additional validation: check if the detected "leaf" pixels actually correspond to non-white areas
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Create a mask for non-white pixels (pixels that are not close to white)
    non_white_mask = gray_crop < 200  # Pixels darker than 200 are considered non-white
    non_white_pixels = np.sum(non_white_mask)
    non_white_percentage = (non_white_pixels / total_pixels) * 100

    # If the leaf percentage is much higher than non-white percentage,
    # it means the detection is picking up white areas incorrectly
    if leaf_percentage > non_white_percentage * 1.5:
        # Return the more conservative non-white percentage
        return non_white_percentage

    return leaf_percentage

def crop_leaves(image_path, output_dir, crop_size=224, step_size=20, threshold=40, max_crops=None):
    """
    Crops the image into smaller images with sliding window approach.
    Saves crops that contain at least threshold% leaf pixels.
    """
    # Create output directory if it doesn't exist, clean it first if it exists
    if os.path.exists(output_dir):
        # Remove all existing files in the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        print(f"Cleaned existing files from {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 0

    # Get image dimensions
    height, width = image.shape[:2]

    # Check if whole image is an envelope
    if is_envelope(image):
        print(f"Image appears to be just an envelope or blank paper. Skipping: {image_path}")
        return 0

    # Detect leaves
    leaf_mask = detect_leaves(image)

    # For debugging - save the leaf mask to check detection quality
    cv2.imwrite(os.path.join(output_dir, "leaf_mask.jpg"), leaf_mask)

    count = 0
    crops_with_scores = []

    # Sliding window approach
    for y in range(0, height - crop_size + 1, step_size):
        for x in range(0, width - crop_size + 1, step_size):
            # Extract crop
            crop = image[y:y+crop_size, x:x+crop_size]
            crop_mask = leaf_mask[y:y+crop_size, x:x+crop_size]

            # Calculate leaf percentage with enhanced validation
            leaf_percent = calculate_leaf_percentage(crop, crop_mask)

            # Additional validation: ensure crop has meaningful content
            if leaf_percent >= threshold:
                # Double-check by analyzing the actual crop colors
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                # Check if crop has diverse colors (not just white/tan)
                unique_colors = len(np.unique(hsv_crop.reshape(-1, hsv_crop.shape[-1]), axis=0))

                # Additional check: ensure the crop is not mostly white background
                if not is_white_background(crop, white_threshold=200, white_percentage_threshold=70):
                    # Check if crop has sufficient contrast
                    if has_sufficient_contrast(crop, min_contrast=15):
                        # Skip crops with very few unique colors (likely uniform background)
                        if unique_colors > 15:  # Increased threshold for more diversity
                            crops_with_scores.append((leaf_percent, crop, x, y))
                            print(f"Valid crop found: {leaf_percent:.2f}% leaf content, {unique_colors} unique colors")
                        else:
                            print(f"Rejected crop: insufficient color diversity ({unique_colors} colors)")
                    else:
                        print(f"Rejected crop: insufficient contrast")
                else:
                    print(f"Rejected crop: too much white background")

    # Sort crops by leaf percentage (highest first)
    crops_with_scores.sort(reverse=True, key=lambda x: x[0])

    # Limit number of crops if specified and not zero
    if max_crops is not None and max_crops > 0:
        crops_with_scores = crops_with_scores[:max_crops]

    # Save the selected crops
    for i, (leaf_percent, crop, x, y) in enumerate(crops_with_scores):
        output_path = os.path.join(output_dir, f"leaf_crop_{i:04d}.jpg")
        cv2.imwrite(output_path, crop)
        print(f"Saved crop {i+1} with {leaf_percent:.2f}% leaf content (from position {x},{y})")

    print(f"Total crops saved: {len(crops_with_scores)}")
    return len(crops_with_scores)

def visualize_crops(image_path, output_dir, crop_size=224, step_size=20, threshold=40, max_crops=None):
    """
    Visualizes the crops on the original image.
    Shows yellow rectangles for crops that will actually be saved.
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 0

    image_vis = image.copy()

    # Check if whole image is an envelope
    if is_envelope(image):
        print(f"Image appears to be just an envelope or blank paper. Skipping visualization: {image_path}")
        # Add text showing image is skipped (but don't save the image)
        cv2.putText(image_vis, "SKIPPED - ENVELOPE/PAPER", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imwrite(os.path.join(output_dir, "crops_visualization.jpg"), image_vis)
        return 0

    # Detect leaves
    leaf_mask = detect_leaves(image)

    # Save the mask for debugging
    cv2.imwrite(os.path.join(output_dir, "crops_visualization_mask.jpg"), leaf_mask)

    # Get image dimensions
    height, width = image.shape[:2]

    # First pass: collect all valid crops (same logic as crop_leaves function)
    crops_with_scores = []
    for y in range(0, height - crop_size + 1, step_size):
        for x in range(0, width - crop_size + 1, step_size):
            # Extract crop
            crop = image[y:y+crop_size, x:x+crop_size]
            crop_mask = leaf_mask[y:y+crop_size, x:x+crop_size]

            # Calculate leaf percentage with enhanced validation
            leaf_percent = calculate_leaf_percentage(crop, crop_mask)

            # Additional validation: ensure crop has meaningful content
            if leaf_percent >= threshold:
                # Double-check by analyzing the actual crop colors
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                # Check if crop has diverse colors (not just white/tan)
                unique_colors = len(np.unique(hsv_crop.reshape(-1, hsv_crop.shape[-1]), axis=0))

                # Skip crops with very few unique colors (likely uniform background)
                if unique_colors > 15:  # At least 15 distinct colors
                    crops_with_scores.append((leaf_percent, x, y))

    # Sort crops by leaf percentage (highest first) - same as crop_leaves
    crops_with_scores.sort(reverse=True, key=lambda x: x[0])

    # Limit number of crops if specified and not zero
    if max_crops is not None and max_crops > 0:
        crops_with_scores = crops_with_scores[:max_crops]

    # Convert to set of coordinates for quick lookup
    selected_crops = set((x, y) for _, x, y in crops_with_scores)

    # Second pass: visualize all crops
    valid_crops = 0
    all_crops = []

    for y in range(0, height - crop_size + 1, step_size):
        for x in range(0, width - crop_size + 1, step_size):
            # Extract crop
            crop = image[y:y+crop_size, x:x+crop_size]
            crop_mask = leaf_mask[y:y+crop_size, x:x+crop_size]

            # Calculate leaf percentage
            leaf_percent = calculate_leaf_percentage(crop, crop_mask)

            if leaf_percent >= threshold:
                # Check color diversity
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                unique_colors = len(np.unique(hsv_crop.reshape(-1, hsv_crop.shape[-1]), axis=0))

                if unique_colors > 15:
                    valid_crops += 1
                    # Check if this crop will actually be saved
                    if (x, y) in selected_crops:
                        # Yellow for crops that will be saved - THICKNESS INCREASED FROM 4 TO 15
                        all_crops.append((x, y, 'yellow', 15))
                    else:
                        # Green for crops that meet threshold but won't be saved - THICKNESS INCREASED FROM 3 TO 6
                        all_crops.append((x, y, 'green', 6))
                else:
                    # Red for low diversity crops (likely white background) - THICKNESS INCREASED FROM 2 TO 5
                    all_crops.append((x, y, 'red', 5))
            else:
                # Red for those that don't meet threshold - THICKNESS INCREASED FROM 2 TO 5
                all_crops.append((x, y, 'red', 5))

    # Draw rectangles in order: red first, then green, then yellow (so yellow is on top)
    for x, y, color, thickness in all_crops:
        if color == 'red':
            cv2.rectangle(image_vis, (x, y), (x+crop_size, y+crop_size), (0, 0, 255), thickness)

    for x, y, color, thickness in all_crops:
        if color == 'green':
            cv2.rectangle(image_vis, (x, y), (x+crop_size, y+crop_size), (0, 255, 0), thickness)

    for x, y, color, thickness in all_crops:
        if color == 'yellow':
            cv2.rectangle(image_vis, (x, y), (x+crop_size, y+crop_size), (0, 255, 255), thickness)

    # Add filled overlay for valid crops (semi-transparent)
    overlay = image_vis.copy()
    for x, y, color, thickness in all_crops:
        if color in ['green', 'yellow']:  # Valid crops
            cv2.rectangle(overlay, (x, y), (x+crop_size, y+crop_size), (0, 255, 0), -1)  # Filled green

    # Blend with alpha
    alpha = 0.05
    image_vis = cv2.addWeighted(overlay, alpha, image_vis, 1 - alpha, 0)

    # Blend the overlay with original image (30% opacity)
    alpha = 0.15
    image_vis = cv2.addWeighted(image_vis, 1 - alpha, overlay, alpha, 0)

    # Add text showing count of valid crops and selected crops - FONT SIZE AND THICKNESS INCREASED
    cv2.putText(image_vis, f"Valid crops: {valid_crops}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)  # Font scale increased from 1 to 2, thickness from 2 to 4
    cv2.putText(image_vis, f"Selected crops: {len(selected_crops)}", (20, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)  # Font scale increased from 1 to 2, thickness from 2 to 4

    # Add legend with larger text
    legend_y_start = height - 120  # Start higher to accommodate larger text
    cv2.putText(image_vis, "Yellow = Will be saved", (20, legend_y_start),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)  # Font scale increased from 0.7 to 1.2, thickness from 2 to 3
    cv2.putText(image_vis, "Green = Valid but not saved", (20, legend_y_start + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Font scale increased from 0.7 to 1.2, thickness from 2 to 3
    cv2.putText(image_vis, "Red = Below threshold/white bg", (20, legend_y_start + 80),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Font scale increased from 0.7 to 1.2, thickness from 2 to 3

    # Save visualization
    cv2.imwrite(os.path.join(output_dir, "crops_visualization.jpg"), image_vis)
    print(f"Visualization saved to {os.path.join(output_dir, 'crops_visualization.jpg')}")
    print(f"Potential crops with this configuration: {valid_crops}")
    print(f"Crops that will be saved: {len(selected_crops)}")

    return valid_crops

def copy_to_drive_with_progress(src_dir, dest_dir):
    """
    Copy files from source directory to destination directory with progress updates,
    excluding visualization.jpg and leaf_mask.jpg files.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # List files in source directory and filter out unwanted files
    all_files = os.listdir(src_dir)
    files = [f for f in all_files if f not in ["visualization.jpg", "crops_visualization.jpg", "leaf_mask.jpg", "crops_visualization_mask.jpg"]]

    # Check if there are any crop files in the list
    crop_files = [f for f in files if f.startswith("leaf_crop_")]

    print(f"Debug: Total files in source: {len(all_files)}")
    print(f"Debug: Files after filtering: {len(files)}")
    print(f"Debug: Crop files found: {len(crop_files)}")

    # Show first few files for debugging
    print(f"Debug: First 5 files in source directory:")
    for i, f in enumerate(all_files[:5]):
        print(f"  {i+1}. {f}")

    if len(crop_files) == 0:
        print(f"No crop files found in {src_dir}. Skipping folder creation.")
        return False

    # Only copy crop files and visualization files, not all files
    files_to_copy = crop_files + [f for f in files if f in ["crops_visualization.jpg", "crops_visualization_mask.jpg"]]

    total_files = len(files_to_copy)

    print(f"Starting to copy {total_files} files to Google Drive at {dest_dir}")
    print(f"Destination directory exists: {os.path.exists(dest_dir)}")

    # Copy files with progress update
    for i, file in enumerate(files_to_copy):
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)

        # Copy file
        shutil.copy2(src_path, dest_path)

        # Update progress
        progress = (i + 1) / total_files * 100
        print(f"Copied {i+1}/{total_files} files ({progress:.1f}%): {file} -> {dest_path}")

        # Small delay to avoid overwhelming Google Drive
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    # Verify files were copied
    copied_files = os.listdir(dest_dir)
    copied_crop_files = [f for f in copied_files if f.startswith("leaf_crop_")]
    print(f"Finished copying files. Found {len(copied_files)} total files in destination directory.")
    print(f"Crop files in destination: {len(copied_crop_files)}")

    # List some files in the destination directory
    print("Files in destination directory (first 5):")
    for file in copied_files[:5]:
        print(f" - {file}")

    return True

def process_herbarium_images(input_path, crop_size=224, step_size=20, threshold=40, desired_crops=100):
    """
    Process herbarium images and save crops to Google Drive.

    Args:
        input_path: Path to the input image or directory of images
        crop_size: Size of each crop
        step_size: Step size for sliding window
        threshold: Minimum leaf content percentage
        desired_crops: Target number of crops to generate (approximate)
                     Set to 0 to save all valid crops
    """
    # Mount Google Drive
    mount_drive()

    # Create a temporary directory to work with (clean it first)
    temp_dir = "/content/temp_crops"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # Remove entire directory and contents
        print(f"Cleaned temp directory: {temp_dir}")
    os.makedirs(temp_dir, exist_ok=True)

    # Process single image
    if os.path.isfile(input_path):
        # Get filename without extension
        file_name = os.path.splitext(os.path.basename(input_path))[0]

        # Remove "Salinan" from file_name if present
        if "Salinan" in file_name:
            file_name = file_name.replace("Salinan ", "")

        # Create output folder with the format "leaf_crops_[filename]"
        output_folder_on_drive = f"leaf_crops_{file_name}_step_{step_size}_total_{desired_crops}"
        drive_output_path = f"/content/drive/MyDrive/{output_folder_on_drive}"

        print(f"Will save crops to: {drive_output_path}")

        # Process the image
        crop_count = crop_leaves(input_path, temp_dir, crop_size, step_size, threshold,
                              None if desired_crops == 0 else desired_crops)
        valid_crops = visualize_crops(input_path, temp_dir, crop_size, step_size, threshold,
                                    None if desired_crops == 0 else desired_crops)

        # If we didn't get enough crops, try adjusting parameters
        if crop_count < desired_crops and desired_crops > 0:
            print(f"Only generated {crop_count} crops. Trying with lower threshold...")

            # Try with lower threshold
            new_threshold = max(threshold - 15, 10)  # Lower minimum to 10%
            crop_count = crop_leaves(input_path, temp_dir, crop_size, step_size, new_threshold,
                                  None if desired_crops == 0 else desired_crops)

            if crop_count < desired_crops and desired_crops > 0:
                print(f"Still only generated {crop_count} crops. Using even more overlapping windows...")

                # Try with smaller step size
                new_step_size = max(step_size // 2, 5)  # Don't go below 5 pixels
                crop_count = crop_leaves(input_path, temp_dir, crop_size, new_step_size, new_threshold,
                                      None if desired_crops == 0 else desired_crops)

                # If still no crops, try with even lower threshold as last resort
                if crop_count == 0:
                    print("No crops generated. Trying with minimal threshold...")
                    crop_count = crop_leaves(input_path, temp_dir, crop_size, new_step_size, 5,
                                          None if desired_crops == 0 else desired_crops)

        # Only copy to Drive if crops were actually generated
        if crop_count > 0:
            # Copy results to Google Drive with progress
            copied = copy_to_drive_with_progress(temp_dir, drive_output_path)

            if copied:
                # Verify the folder exists in Google Drive
                if os.path.exists(drive_output_path):
                    print(f"Confirmed folder exists in Google Drive: {drive_output_path}")
                    print(f"Contents of {drive_output_path}:")
                    for item in os.listdir(drive_output_path)[:5]:  # Show first 5 items
                        print(f" - {item}")
                else:
                    print(f"ERROR: Folder not found in Google Drive: {drive_output_path}")
        else:
            print(f"No crops were generated for {file_name}. Skipping folder creation.")

    # Process directory of images
    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                file_path = os.path.join(input_path, file)
                print(f"\n\nProcessing image: {file}")

                # Get filename without extension
                file_name = os.path.splitext(file)[0]

                # Remove "Salinan" from file_name if present
                if "Salinan" in file_name:
                    file_name = file_name.replace("Salinan ", "")

                # Create a specific output folder for each image
                img_output_dir = os.path.join(temp_dir, file_name)
                os.makedirs(img_output_dir, exist_ok=True)

                # Create output folder with the format "leaf_crops_[filename]"
                output_folder_on_drive = f"leaf_crops_{file_name}_step_{step_size}_total_{desired_crops}"
                drive_img_output_dir = f"/content/drive/MyDrive/{output_folder_on_drive}"

                # Process the image
                crop_count = crop_leaves(file_path, img_output_dir, crop_size, step_size, threshold,
                                     None if desired_crops == 0 else desired_crops)
                valid_crops = visualize_crops(file_path, img_output_dir, crop_size, step_size, threshold,
                                            None if desired_crops == 0 else desired_crops)

                # If we didn't get enough crops, try adjusting parameters
                if crop_count < desired_crops and desired_crops > 0:
                    print(f"Only generated {crop_count} crops. Trying with lower threshold...")

                    # Try with lower threshold
                    new_threshold = max(threshold - 15, 10)  # Lower minimum to 10%
                    crop_count = crop_leaves(file_path, img_output_dir, crop_size, step_size, new_threshold,
                                         None if desired_crops == 0 else desired_crops)

                    if crop_count < desired_crops and desired_crops > 0:
                        print(f"Still only generated {crop_count} crops. Using even more overlapping windows...")

                        # Try with smaller step size
                        new_step_size = max(step_size // 2, 5)  # Don't go below 5 pixels
                        crop_count = crop_leaves(file_path, img_output_dir, crop_size, new_step_size, new_threshold,
                                             None if desired_crops == 0 else desired_crops)

                        # If still no crops, try with even lower threshold as last resort
                        if crop_count == 0:
                            print("No crops generated. Trying with minimal threshold...")
                            crop_count = crop_leaves(file_path, img_output_dir, crop_size, new_step_size, 5,
                                                 None if desired_crops == 0 else desired_crops)

                # Only copy to Drive if crops were actually generated
                if crop_count > 0:
                    # Copy results to Google Drive with progress
                    copied = copy_to_drive_with_progress(img_output_dir, drive_img_output_dir)
                    if copied:
                        print(f"Created folder for {file_name} with {crop_count} crops.")
                else:
                    print(f"No crops were generated for {file_name}. Skipping folder creation.")

    else:
        print(f"Input path {input_path} does not exist")

# Usage
input_path = "/content/drive/MyDrive/prediction_tool/baccata/"
crop_size = 224
step_size = 224
threshold = 70
desired_crops = 0

process_herbarium_images(input_path, crop_size, step_size, threshold, desired_crops)