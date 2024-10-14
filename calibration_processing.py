import numpy as np
import cv2
import matplotlib.pyplot as plt


m,n= 20,15

# calibration_processing.py

# calibration_processing.py

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Calibration for .jpg images (improved step: heatmap with two reference temperature points and emissivity)
def calibrate_jpg_temperature(image_rgb, temperature_matrix_jpg, emissivity_matrix, m, n):
    """
    Generate a heatmap from the RGB data of the image using two reference temperature points and emissivity adjustments.
    
    Args:
        image_rgb (np.array): The RGB image of the PCB.
        temperature_matrix_jpg (np.array): The RGB average temperature matrix (not used here).
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows for the grid.
        n (int): Number of columns for the grid.

    Returns:
        np.array: Temperature matrix (heatmap) with linear scaling based on two reference temperatures and emissivity.
    """
    # Step 1: Get the height and width of the image
    height, width, _ = image_rgb.shape

    # Step 2: Calculate the size of each cell in the grid
    cell_height = height // m
    cell_width = width // n

    # Step 3: Create a matrix to store the average brightness (grayscale) values for each cell
    temperature_matrix = np.zeros((m, n), dtype=np.float32)

    # Step 4: Loop through the grid and calculate the average brightness for each cell
    for i in range(m):
        for j in range(n):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Get the region of interest (ROI) for the current cell
            cell = image_rgb[y_start:y_end, x_start:x_end]

            # Convert the cell to grayscale (average of RGB channels)
            avg_brightness = np.mean(cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY))

            # Adjust the brightness by the emissivity value for this cell
            emissivity = emissivity_matrix[i, j]
            if emissivity > 0:  # Avoid division by zero
                avg_brightness /= emissivity
            else:
                avg_brightness = 0  # Handle cases where emissivity is zero or very low

            # Store the emissivity-corrected brightness in the temperature matrix
            temperature_matrix[i, j] = avg_brightness

    # Step 5: Input two reference temperature points and their known temperatures
    reference_points = []
    print("Click on two points on the PCB where you know the temperature.")
    
    def select_reference_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(reference_points) < 2:
            reference_points.append((x, y))
            cv2.circle(image_rgb, (x, y), 5, (0, 255, 0), -1)  # Mark the points
            cv2.imshow('Select Two Temperature Points', image_rgb)

    cv2.imshow('Select Two Temperature Points', image_rgb)
    cv2.setMouseCallback('Select Two Temperature Points', select_reference_points)
    
    # Wait until two points are selected
    while len(reference_points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Step 6: Get the known temperatures for the two points
    temp1 = float(input(f"Enter the temperature (in °C) for the first point {reference_points[0]}: "))
    temp2 = float(input(f"Enter the temperature (in °C) for the second point {reference_points[1]}: "))

    # Step 7: Get the grayscale brightness values at the reference points
    x1, y1 = reference_points[0]
    x2, y2 = reference_points[1]
    brightness1 = temperature_matrix[y1 // cell_height, x1 // cell_width]
    brightness2 = temperature_matrix[y2 // cell_height, x2 // cell_width]

    # Step 8: Create a linear mapping function
    def brightness_to_temperature(brightness, brightness1, brightness2, temp1, temp2):
        # Linearly map the brightness values to temperatures
        return temp1 + (brightness - brightness1) * (temp2 - temp1) / (brightness2 - brightness1)

    # Step 9: Apply the mapping to the entire temperature matrix
    for i in range(m):
        for j in range(n):
            temperature_matrix[i, j] = brightness_to_temperature(temperature_matrix[i, j], brightness1, brightness2, temp1, temp2)

    # Step 10: Print a sample of the temperature matrix for debugging
    print("Sample of the calibrated temperature matrix (first 5x5):")
    print(temperature_matrix[:5, :5])

    return temperature_matrix

# Visualization function for the temperature matrix (heatmap)
def visualize_temperature_matrix(temperature_matrix):
    plt.imshow(temperature_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Calibrated Heatmap with Emissivity Correction')
    plt.show()




# calibration_processing.py

# Calibration for .tif thermography
def calibrate_tif_temperature(radiometric_data, emissivity_matrix, m, n):
    """
    Calibrate the temperature matrix using the radiometric data and emissivity matrix (for .tif images).

    Args:
        radiometric_data (np.array): The radiometric temperature data from the .tif image.
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the emissivity matrix.
        n (int): Number of columns in the emissivity matrix.

    Returns:
        np.array: Calibrated temperature matrix.
    """
    # Verificar dimensiones
    print("Dimensiones de los datos radiométricos:", radiometric_data.shape)
    print("Dimensiones de la matriz de emisividad:", emissivity_matrix.shape)

    # Redimensionar la matriz de emisividad a las dimensiones de los datos radiométricos
    height, width = radiometric_data.shape
    emissivity_resized = cv2.resize(emissivity_matrix, (width, height), interpolation=cv2.INTER_LINEAR)

    # Inicializar la matriz de temperaturas calibradas
    temperature_values = np.zeros_like(radiometric_data, dtype=np.float32)

    # Aplicar corrección de emisividad
    for i in range(height):
        for j in range(width):
            radiometric_value = radiometric_data[i, j]
            emissivity = max(emissivity_resized[i, j], 0.1)  # Evitar divisiones por valores muy pequeños de emisividad
            temperature_values[i, j] = radiometric_value / emissivity  # Ajustar por emisividad

    # Imprimir un ejemplo de la matriz de temperatura calibrada
    print("Matriz de temperatura calibrada (primeros 5x5 valores):")
    print(temperature_values[:5, :5])

    return temperature_values

# Main function to handle calibration based on image type
def calibrate_temperature(image_type, image_data, temperature_matrix, emissivity_matrix, m, n):
    """
    Calibrate the temperature matrix based on whether the image is .jpg or .tif.

    Args:
        image_type (str): The type of image ('jpg' or 'tif').
        image_data (np.array): The image data (RGB for .jpg, radiometric data for .tif).
        temperature_matrix (np.array): The RGB matrix (for .jpg).
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the matrix (height).
        n (int): Number of columns in the matrix (width).

    Returns:
        np.array: Calibrated temperature matrix.
    """
    if image_type == 'jpg':
        # Call the JPG calibration method (ya implementado)
        return calibrate_jpg_temperature(image_data, temperature_matrix, emissivity_matrix, m, n)
    elif image_type == 'tif':
        # Call the TIF calibration method
        return calibrate_tif_temperature(image_data, emissivity_matrix, m, n)
    else:
        raise ValueError("Unsupported image type. Only 'jpg' and 'tif' are supported.")

# Function to visualize the heatmap of the temperature matrix
def visualize_temperature_matrix(temperature_matrix):
    plt.imshow(temperature_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')  # Actualizado a °C
    plt.title('Calibrated Temperature Heatmap')
    plt.show()