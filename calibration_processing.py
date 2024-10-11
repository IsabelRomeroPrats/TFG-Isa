import numpy as np
import cv2
import matplotlib.pyplot as plt


m,n= 20,15

# Calibration for .jpg thermography
def calibrate_jpg_temperature(image_rgb, temperature_matrix_jpg, emissivity_matrix, m, n):
    """
    Calibrate the temperature matrix using the RGB data and a known reference temperature point (for .jpg images).

    Args:
        image_rgb (np.array): The RGB image to calibrate.
        temperature_matrix_jpg (np.array): The RGB matrix of the image (20x15 grid).
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the grid (height).
        n (int): Number of columns in the grid (width).

    Returns:
        np.array: Calibrated temperature matrix.
    """
    # Step 1: Allow the user to select a known temperature point
    image_rgb_copy = np.copy(image_rgb)
    temperature_point = []

    def select_temperature_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            temperature_point.append((x, y))
            cv2.circle(image_rgb_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Temperature Point', image_rgb_copy)

    # Show image and select the known temperature point
    cv2.imshow('Select Temperature Point', image_rgb_copy)
    cv2.setMouseCallback('Select Temperature Point', select_temperature_point)
    
    # Wait until a point is selected
    print("Click on a point where you know the temperature.")
    while len(temperature_point) == 0:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Step 2: Get the known temperature from the user
    known_temperature_kelvin = float(input(f"Enter the known temperature (in Kelvin) for the point {temperature_point}: "))
    print(f"The known temperature is: {known_temperature_kelvin} K at {temperature_point}.")

    # Step 3: Map the selected point to the grid
    x_full, y_full = temperature_point[0]  # Full image coordinates
    height, width, _ = image_rgb.shape  # Dimensions of the full image

    # Calculate the cell height and width
    cell_height = height / m
    cell_width = width / n

    # Map the full image coordinates to the grid indices in the 20x15 matrix
    x_grid = int(x_full // cell_width)
    y_grid = int(y_full // cell_height)

    # Ensure the grid coordinates are within bounds
    x_grid = min(x_grid, n - 1)
    y_grid = min(y_grid, m - 1)

    # Extract the reference RGB value at the selected point
    reference_rgb_value = temperature_matrix_jpg[y_grid, x_grid].astype(np.float32)
    print(f"Reference RGB Value at ({x_grid}, {y_grid}): {reference_rgb_value}")

    # Step 4: Define the calibration function
    def rgb_to_temperature_calibration(rgb_value, ref_rgb_value, ref_temperature, emissivity):
        epsilon = 1e-6
        r_ratio = rgb_value[0] / (ref_rgb_value[0] + epsilon)
        g_ratio = rgb_value[1] / (ref_rgb_value[1] + epsilon)
        b_ratio = rgb_value[2] / (ref_rgb_value[2] + epsilon)
        ratios = [r_ratio, g_ratio, b_ratio]
        average_ratio = np.mean([r for r in ratios if r < 10])
        temperature = average_ratio * ref_temperature
        temperature_adjusted = temperature / (emissivity + epsilon)
        return temperature_adjusted

    # Step 5: Create a matrix to store the temperature values
    temperature_values = np.zeros((m, n), dtype=np.float32)

    # Step 6: Loop over the grid and apply the calibration function
    for i in range(m):
        for j in range(n):
            rgb_value = temperature_matrix_jpg[i, j].astype(np.float32)
            emissivity = emissivity_matrix[i, j]
            temperature = rgb_to_temperature_calibration(rgb_value, reference_rgb_value, known_temperature_kelvin, emissivity)
            temperature_values[i, j] = temperature

     # Step 7: Print a sample of the calibrated temperature matrix (first 5x5 elements)
    print("Sample of calibrated temperature matrix (first 5x5):")
    print(temperature_values[:5, :5])  # Prints the top-left 5x5 part of the matrix

    # Step 8: Return the calibrated temperature matrix
    return temperature_values


# Calibration for .tif thermography
def calibrate_tif_temperature(radiometric_data, emissivity_matrix, m, n):
    """
    Calibrate the temperature matrix using the radiometric data and emissivity matrix (for .tif images).

    Args:
        radiometric_data (np.array): The radiometric temperature data from the .tif image.
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the matrix (height).
        n (int): Number of columns in the matrix (width).

    Returns:
        np.array: Calibrated temperature matrix.
    """
    # Step 1: Adjust the radiometric data by the emissivity matrix
    temperature_values = np.zeros((m, n), dtype=np.float32)
    
    # Step 2: Loop through the grid to apply emissivity correction
    for i in range(m):
        for j in range(n):
            radiometric_value = radiometric_data[i, j]
            emissivity = emissivity_matrix[i, j]
            temperature_values[i, j] = radiometric_value / emissivity  # Adjust using emissivity

    # Step 3: Print a sample of the calibrated temperature matrix (first 5x5 elements)
    print("Sample of calibrated temperature matrix (first 5x5):")
    print(temperature_values[:5, :5])  # Prints the top-left 5x5 part of the matrix

    # Step 4: Return the calibrated temperature matrix
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
        # Call the JPG calibration method
        return calibrate_jpg_temperature(image_data, temperature_matrix, emissivity_matrix, m, n)
    elif image_type == 'tif':
        # Call the TIF calibration method
        return calibrate_tif_temperature(image_data, emissivity_matrix, m, n)
    else:
        raise ValueError("Unsupported image type. Only 'jpg' and 'tif' are supported.")

# Function to visualize the heatmap of the temperature matrix
def visualize_temperature_matrix(temperature_matrix):
    plt.imshow(temperature_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (K)')
    plt.title('Calibrated Temperature Heatmap')
    plt.show()
