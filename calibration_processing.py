import numpy as np
import cv2
import matplotlib.pyplot as plt

# Calibration for .jpg images
def calibrate_jpg_temperature(image_rgb, temperature_matrix_jpg, emissivity_matrix):
    """
    Generate a full-resolution heatmap from the RGB data of the image using two reference temperature points 
    and emissivity adjustments. This will return a matrix matching the original image dimensions.

    Args:
        image_rgb (np.array): The RGB image of the PCB.
        temperature_matrix_jpg (np.array): The RGB average temperature matrix (not used here).
        emissivity_matrix (np.array): The emissivity matrix.

    Returns:
        np.array: Full-resolution temperature matrix (heatmap) with linear scaling based on two reference temperatures.
    """
    # Get the image dimensions and create a full-resolution matrix
    height, width, _ = image_rgb.shape  # Example: 480x640
    temperature_matrix = np.zeros((height, width), dtype=np.float32)

    # Loop over the image and calculate brightness (grayscale) for each pixel, corrected by emissivity
    for i in range(height):
        for j in range(width):
            # Convert the pixel to grayscale (brightness)
            brightness = np.mean(cv2.cvtColor(image_rgb[i:i+1, j:j+1], cv2.COLOR_RGB2GRAY))

            # Apply emissivity correction
            emissivity = emissivity_matrix[i // (height // 20), j // (width // 20)]  # Use the 20x20 emissivity matrix
            if emissivity > 0:
                brightness /= emissivity
            else:
                brightness = 0  # Handle low emissivity

            temperature_matrix[i, j] = brightness

    # Allow the user to select two reference points to calibrate the brightness to temperature
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
    temp1 = float(input(f"Enter the temperature (in °C) for the first point {reference_points[0]}: ")) + 273.15
    temp2 = float(input(f"Enter the temperature (in °C) for the second point {reference_points[1]}: ")) + 273.15

    # Get the brightness values at the reference points
    x1, y1 = reference_points[0]
    x2, y2 = reference_points[1]
    brightness1 = temperature_matrix[y1, x1]
    brightness2 = temperature_matrix[y2, x2]

    # Create a function to map brightness to temperature
    def brightness_to_temperature(brightness, brightness1, brightness2, temp1, temp2):
        return temp1 + (brightness - brightness1) * (temp2 - temp1) / (brightness2 - brightness1)

    # Apply the temperature calibration to the entire matrix
    for i in range(height):
        for j in range(width):
            temperature_matrix[i, j] = brightness_to_temperature(temperature_matrix[i, j], brightness1, brightness2, temp1, temp2)

    # Return the full-resolution temperature matrix (480x640 or whatever the image size is)
    return temperature_matrix


def visualize_temperature_matrix_jpg(temperature_matrix):
    """
    Displays the calibrated full-resolution temperature matrix (480x640) as a heatmap, then discretizes it 
    to a 20x20 matrix and displays the discretized matrix as a separate heatmap.

    Args:
        temperature_matrix (np.array): The calibrated temperature matrix (480x640).
    """
    # --- Full-resolution heatmap: Use the full original matrix ---
    full_resolution_matrix = temperature_matrix.copy()  # Ensure we don't overwrite the original matrix

    # Plot the full-resolution heatmap (480x640)
    plt.figure(figsize=(8, 6))
    plt.imshow(full_resolution_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (K)')
    plt.title('Full-Resolution Temperature Heatmap (480x640)')
    plt.show()

    # --- Discretized heatmap: Now discretize the full-resolution matrix ---
    height, width = full_resolution_matrix.shape  # Get the original matrix dimensions
    
    # Define the grid dimensions for discretization (10x20)
    m, n = 20, 20
    
    # Calculate the size of each grid cell (32x32 for example)
    cell_height = height // m
    cell_width = width // n
    
    # Initialize the discretized temperature matrix
    discretized_matrix = np.zeros((m, n))

    # Loop over the temperature matrix and calculate the average temperature in each grid cell
    for i in range(m):
        for j in range(n):
            # Calculate the start and end indices for the current block
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Extract the 32x32 block of the temperature matrix
            block = full_resolution_matrix[y_start:y_end, x_start:x_end]

            if block.size > 0:  # Ensure block is not empty
                # Calculate the average temperature in the block
                avg_temperature = np.mean(block)
                discretized_matrix[i, j] = avg_temperature
            else:
                discretized_matrix[i, j] = np.nan  # Mark as nan if no data

    # Plot the heatmap for the discretized matrix (10x20)
    plt.figure(figsize=(8, 6))
    plt.imshow(discretized_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (K)')
    plt.title('Discretized Temperature Heatmap (20x20)')
    plt.show()

    # Print the discretized temperature matrix as an array
    print("Discretized temperature matrix:")
    print(discretized_matrix)

    # Print the dimensions of the original and discretized matrices
    print(f"Original matrix dimensions: {full_resolution_matrix.shape}")
    print(f"Discretized matrix dimensions: {discretized_matrix.shape}")


# Calibration for .tif thermographic images
def calibrate_tif_temperature(image_rgb, emissivity_matrix, m, n, is_kelvin=True):
    """
    Calibrate the temperature matrix using the radiometric data and emissivity matrix (for .tif images).

    Args:
        image_rgb (np.array): The radiometric temperature data from the .tif image.
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the emissivity matrix.
        n (int): Number of columns in the emissivity matrix.
        is_kelvin (bool): Whether the input data is already in Kelvin.

    Returns:
        np.array: Calibrated temperature matrix.
    """
    # Check the dimensions of the data
    print("Radiometric data dimensions:", image_rgb.shape)
    print("Emissivity matrix dimensions:", emissivity_matrix.shape)

    # Resize the emissivity matrix to match the radiometric data dimensions
    height, width = image_rgb.shape
    emissivity_resized = cv2.resize(emissivity_matrix, (width, height), interpolation=cv2.INTER_LINEAR)

    # Initialize the calibrated temperature matrix
    temperature_values = np.zeros_like(image_rgb, dtype=np.float32)

    # Apply emissivity correction to each pixel
    for i in range(height):
        for j in range(width):
            radiometric_value = image_rgb[i, j]

            # Convert to Kelvin only if the input is not already in Kelvin
            if not is_kelvin:
                radiometric_value += 273.15

            emissivity = max(emissivity_resized[i, j], 0.1)  # Avoid division by very low emissivity values
            temperature_values[i, j] = radiometric_value / emissivity  # Adjust by emissivity

    # Print a sample of the calibrated matrix
    print("Calibrated temperature matrix (first 5x5 values):")
    print(temperature_values[:5, :5])

    return temperature_values


def visualize_temperature_matrix_tif(temperature_matrix):
    """
    Displays both the full-resolution and discretized heatmaps for .tif images.

    Args:
        temperature_matrix (np.array): The calibrated temperature matrix (480x640).
    """
    # First, check if the matrix is valid
    if temperature_matrix is None or temperature_matrix.size == 0:
        print("Error: Empty temperature matrix.")
        return

    # Plot the original full-resolution heatmap (480x640)
    plt.figure(figsize=(8, 6))
    plt.imshow(temperature_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (K)')
    plt.title('Full-Resolution Temperature Heatmap (480x640) for .tif')
    plt.show()

    # Now discretize the matrix to a 15x20 size
    height, width = temperature_matrix.shape
    
    # Define the grid dimensions for discretization
    m, n = 15, 20  # Target matrix size
    
    # Calculate the size of each grid cell (32x32)
    cell_height = height // m
    cell_width = width // n
    
    # Initialize the discretized temperature matrix
    discretized_matrix = np.zeros((m, n))

    # Loop over the temperature matrix and calculate the average temperature in each grid cell
    for i in range(m):
        for j in range(n):
            # Calculate the start and end indices for the current block
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Extract the 32x32 block of the temperature matrix
            block = temperature_matrix[y_start:y_end, x_start:x_end]

            if block.size > 0:  # Ensure block is not empty
                # Calculate the average temperature in the block
                avg_temperature = np.mean(block)
                discretized_matrix[i, j] = avg_temperature
            else:
                discretized_matrix[i, j] = np.nan  # Mark as nan if no data

    # Plot the heatmap for the discretized matrix (15x20)
    plt.figure(figsize=(8, 6))
    plt.imshow(discretized_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (K)')
    plt.title('Discretized Temperature Heatmap (15x20) for .tif')
    plt.show()

    # Print the discretized temperature matrix as an array
    print("Discretized temperature matrix for .tif:")
    print(discretized_matrix)

    # Print the dimensions of the original and discretized matrices
    print(f"Original matrix dimensions for .tif: {temperature_matrix.shape}")
    print(f"Discretized matrix dimensions for .tif: {discretized_matrix.shape}")


# Main function to calibrate temperature based on image type
def calibrate_temperature(image_type, image_data, temperature_matrix, emissivity_matrix, m, n, is_kelvin=True):
    """
    Calibrate the temperature matrix based on whether the image is .jpg or .tif.

    Args:
        image_type (str): The type of image ('jpg' or 'tif').
        image_data (np.array): The image data (RGB for .jpg, radiometric data for .tif).
        temperature_matrix (np.array): The RGB matrix (for .jpg).
        emissivity_matrix (np.array): The emissivity matrix.
        m (int): Number of rows in the grid.
        n (int): Number of columns in the grid.
        is_kelvin (bool): Whether the input data is already in Kelvin.

    Returns:
        np.array: Calibrated temperature matrix.
    """
    if image_type == 'jpg':
        # Call the JPG calibration method
        return calibrate_jpg_temperature(image_data, temperature_matrix, emissivity_matrix)
    elif image_type == 'tif':
        # Call the TIF calibration method with the Kelvin flag
        return calibrate_tif_temperature(image_data, emissivity_matrix, m, n, is_kelvin)
    else:
        raise ValueError("Unsupported image type. Only 'jpg' and 'tif' are supported.")