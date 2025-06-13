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


def warp_perspective_from_points(image, source_points, width=640, height=480):
    if len(source_points) != 4:
        raise ValueError("You must provide exactly 4 source points.")

    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    source = np.array(source_points, dtype='float32')
    matrix = cv2.getPerspectiveTransform(source, target_corners)
    aligned = cv2.warpPerspective(image, matrix, (width, height))
    return aligned
