# image_processing.py

import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Nueva función para redimensionar matrices
def matrix_resized(matrix, continuous_shape=(500, 500), m=20, n=20):
    # Redimensionar continua
    matrix_continuous = cv2.resize(matrix, (continuous_shape[1], continuous_shape[0]))

    # Crear heatmap discreto
    cell_height, cell_width = continuous_shape[0] // m, continuous_shape[1] // n
    discrete_heatmap = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            y_start, y_end = i * cell_height, (i + 1) * cell_height
            x_start, x_end = j * cell_width, (j + 1) * cell_width
            block = matrix_continuous[y_start:y_end, x_start:x_end]
            discrete_heatmap[i, j] = np.mean(block)

    return matrix_continuous, discrete_heatmap


def load_and_display_image(image_name, temperature):

    folder = f"T{int(temperature)}"
    image_path = os.path.join(folder, image_name)

    extension = image_name.split('.')[-1].lower()

    if extension in ['jpg', 'png']:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_name}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(f"Loaded Image: {image_name}")
        plt.axis('off')
        plt.show()
        return image

    elif extension == 'tif':
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not load .tif image {image_name}")
            return None
        plt.imshow(image, cmap='gray')
        plt.title(f"Loaded .tif Image: {image_name}")
        plt.axis('off')
        plt.show()
        return image

    else:
        print(f"Unsupported file format: {extension}")
        return None


def select_corners_jpg(image, calculate_rgb=True):
    corners = []

    def select_corners(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select PCB Corners', image)

    cv2.imshow('Select PCB Corners', image)
    cv2.setMouseCallback('Select PCB Corners', select_corners)

    print("Select 4 corners of the PCB by clicking on the image.")
    while len(corners) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(f"Corners selected: {corners}")

    width, height = 640, 480
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    corners = np.array(corners, dtype='float32')
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    aligned_image_jpg = cv2.warpPerspective(image, matrix, (width, height))

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(aligned_image_jpg, cv2.COLOR_BGR2RGB))
    plt.title("Aligned JPG Image")
    plt.axis('off')
    plt.show()

    if not calculate_rgb:
        return aligned_image_jpg, None, corners

    matrix_continuous, temperature_matrix_jpg = matrix_resized(aligned_image_jpg)

    plt.figure(figsize=(5, 5))
    plt.imshow(temperature_matrix_jpg, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (relative scale)')
    plt.title("Temperature Matrix - JPG")
    plt.axis('off')
    plt.show()

    print("Sample of the temperature matrix:")
    print(temperature_matrix_jpg[:5, :5])

    return aligned_image_jpg, temperature_matrix_jpg, corners


def select_corners_tif(tif_data, temperature):
    import os
    from PIL import Image

    folder = f"T{int(temperature)}"
    os.makedirs(folder, exist_ok=True)
    temp_png_path = os.path.join(folder, "temp_corner_selection.png")

    # Convertimos a Kelvin y normalizamos
    tif_data = tif_data + 273.15
    min_val = np.min(tif_data)
    max_val = np.max(tif_data)

    if min_val != max_val:
        normalized_data = np.interp(tif_data, (min_val, max_val), (0, 255))
    else:
        normalized_data = np.zeros_like(tif_data)

    # Aumentamos contraste y guardamos PNG temporal
    uint8_data = normalized_data.astype(np.uint8)
    enhanced_data = cv2.equalizeHist(uint8_data)
    temp_image = Image.fromarray(enhanced_data)
    temp_image.save(temp_png_path, format='PNG')
    print(f"Temporary PNG saved at: {temp_png_path}")

    png_image = cv2.imread(temp_png_path)
    if png_image is None:
        raise ValueError("Error loading the temporary .png image.")

    aligned_visual_tif, _, corners = select_corners_jpg(png_image, calculate_rgb=False)

    if corners is None or len(corners) != 4:
        raise ValueError("Corner selection error. Make sure to select all 4 corners.")

    width, height = 640, 480
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    corners = np.array(corners, dtype='float32')
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    aligned_radiometric_data = cv2.warpPerspective(tif_data, matrix, (width, height))

    return aligned_visual_tif, aligned_radiometric_data
