import numpy as np
import cv2
import matplotlib.pyplot as plt

# Calibration for .jpg images using two reference temperature points and emissivity adjustments
def calibrate_jpg_temperature(image_rgb, temperature_matrix_jpg, emissivity_matrix, m, n):
    """
    Generate a heatmap from the RGB data of the image using two reference temperature points and emissivity adjustments.
    
    Args:
        image_rgb (np.array): The RGB image of the PCB.
        temperature_matrix_jpg (np.array): The RGB average temperature matrix (not used here).
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the grid.
        n (int): Number of columns in the grid.

    Returns:
        np.array: Temperature matrix (heatmap) with linear scaling based on two reference temperatures and emissivity.
    """
    # Get the image dimensions and grid cell size
    height, width, _ = image_rgb.shape
    cell_height = height // m
    cell_width = width // n

    # Create matrix to store brightness (grayscale) values for each cell
    temperature_matrix = np.zeros((m, n), dtype=np.float32)

    # Calculate the average brightness for each cell, corrected by emissivity
    for i in range(m):
        for j in range(n):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Get the region of interest (ROI) and convert to grayscale
            cell = image_rgb[y_start:y_end, x_start:x_end]
            avg_brightness = np.mean(cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY))

            # Correct brightness using emissivity
            emissivity = emissivity_matrix[i, j]
            if emissivity > 0:
                avg_brightness /= emissivity
            else:
                avg_brightness = 0  # Handle very low emissivity

            temperature_matrix[i, j] = avg_brightness

    # Allow the user to select two reference points on the PCB
    reference_points = []
    print("Click on two points on the PCB where you know the temperature.")
    
    def select_reference_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(reference_points) < 2:
            reference_points.append((x, y))
            cv2.circle(image_rgb, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Two Temperature Points', image_rgb)

    cv2.imshow('Select Two Temperature Points', image_rgb)
    cv2.setMouseCallback('Select Two Temperature Points', select_reference_points)
    
    while len(reference_points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Input the known temperatures for the reference points
    temp1 = float(input(f"Enter the temperature (in °C) for the first point {reference_points[0]}: "))
    temp2 = float(input(f"Enter the temperature (in °C) for the second point {reference_points[1]}: "))

    # Get the brightness values at the reference points
    x1, y1 = reference_points[0]
    x2, y2 = reference_points[1]
    brightness1 = temperature_matrix[y1 // cell_height, x1 // cell_width]
    brightness2 = temperature_matrix[y2 // cell_height, x2 // cell_width]

    # Create a function to linearly map brightness to temperature
    def brightness_to_temperature(brightness, brightness1, brightness2, temp1, temp2):
        return temp1 + (brightness - brightness1) * (temp2 - temp1) / (brightness2 - brightness1)

    # Apply the temperature mapping to the entire matrix
    for i in range(m):
        for j in range(n):
            temperature_matrix[i, j] = brightness_to_temperature(temperature_matrix[i, j], brightness1, brightness2, temp1, temp2)

    # Print a sample of the calibrated matrix for debugging
    print("Sample of the calibrated temperature matrix (first 5x5):")
    print(temperature_matrix[:5, :5])

    return temperature_matrix

# Visualization function for the temperature matrix (heatmap)
def visualize_temperature_matrix(temperature_matrix):
    """
    Displays the calibrated temperature matrix as a heatmap.
    
    Args:
        temperature_matrix (np.array): The calibrated temperature matrix.
    """
    plt.imshow(temperature_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Calibrated Heatmap with Emissivity Correction')
    plt.show()

# Calibration for .tif thermographic images
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
    # Check the dimensions of the data
    print("Radiometric data dimensions:", radiometric_data.shape)
    print("Emissivity matrix dimensions:", emissivity_matrix.shape)

    # Resize the emissivity matrix to match the radiometric data dimensions
    height, width = radiometric_data.shape
    emissivity_resized = cv2.resize(emissivity_matrix, (width, height), interpolation=cv2.INTER_LINEAR)

    # Initialize the calibrated temperature matrix
    temperature_values = np.zeros_like(radiometric_data, dtype=np.float32)

    # Apply emissivity correction to each pixel
    for i in range(height):
        for j in range(width):
            radiometric_value = radiometric_data[i, j]
            emissivity = max(emissivity_resized[i, j], 0.1)  # Avoid division by very low emissivity values
            temperature_values[i, j] = radiometric_value / emissivity  # Adjust by emissivity

    # Print a sample of the calibrated matrix
    print("Calibrated temperature matrix (first 5x5 values):")
    print(temperature_values[:5, :5])

    return temperature_values

# Main function to calibrate temperature based on image type
def calibrate_temperature(image_type, image_data, temperature_matrix, emissivity_matrix, m, n):
    """
    Calibrate the temperature matrix based on whether the image is .jpg or .tif.

    Args:
        image_type (str): The type of image ('jpg' or 'tif').
        image_data (np.array): The image data (RGB for .jpg, radiometric data for .tif).
        temperature_matrix (np.array): The RGB matrix (for .jpg).
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the grid.
        n (int): Number of columns in the grid.

    Returns:
        np.array: Calibrated temperature matrix.
    """
    if image_type == 'jpg':
        # Call the JPG calibration method
        return calibrate_jpg_temperature(image_data, temperature_matrix, emissivity_matrix, m, n)
    elif image_type == 'tif':
        # Call the TIF calibration method
        return calibrate_tif_temperature(image_data, emissivity_matrix, m, n)
    else:
        raise ValueError("Unsupported image type. Only 'jpg' and 'tif' are supported.")

# Function to visualize the temperature heatmap
def visualize_temperature_matrix(temperature_matrix):
    """
    Displays the calibrated temperature matrix as a heatmap.
    
    Args:
        temperature_matrix (np.array): The calibrated temperature matrix.
    """
    plt.imshow(temperature_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Calibrated Temperature Heatmap')
    plt.show()
