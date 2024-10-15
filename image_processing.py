# image_processing.py

import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_and_display_image(image_name):
    """
    Loads and displays an image from the 'Images' folder based on its extension.
    
    Args:
        image_name (str): The name of the image file, including extension.
    
    Returns:
        image (np.array or None): The loaded image if successful, otherwise None.
    """
    # Define the path to the "Images" folder
    images_folder = "Images"
    image_path = os.path.join(images_folder, image_name)

    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file {image_name} was not found in the Images folder.")
        return None

    # Detect the file extension
    extension = image_name.split('.')[-1].lower()

    # Load and display the image based on extension
    if extension in ['jpg', 'png']:
        # Load the image using OpenCV (for non-radiometric images)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_name}")
            return None
        # Convert color from BGR to RGB for correct display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(f"Loaded Image: {image_name}")
        plt.axis('off')
        plt.show()
        return image
    
    elif extension == 'tif':
        # Load .tif image (may need specific handling for radiometric .tif files)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not load .tif image {image_name}")
            return None
        # Display .tif image in grayscale
        plt.imshow(image, cmap='gray')
        plt.title(f"Loaded .tif Image: {image_name}")
        plt.axis('off')
        plt.show()
        return image
    
    else:
        print(f"Unsupported file format: {extension}")
        return None


def select_corners_jpg(image, calculate_rgb=True):
    """
    Allows the user to manually select corners in .jpg or .png images.
    Aligns the image based on the selected corners and resizes it to 640x480 pixels.
    If calculate_rgb is True, it also calculates the RGB temperature matrix.

    Args:
        image (np.array): The loaded image.
        calculate_rgb (bool): Whether to calculate the RGB temperature matrix.

    Returns:
        aligned_image_jpg (np.array): The transformed and aligned image.
        temperature_matrix_jpg (np.array or None): RGB temperature matrix or None if calculate_rgb is False.
        corners (list): Coordinates of the selected corners.
    """
    corners = []

    def select_corners(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select PCB Corners', image)

    # Display the image for corner selection
    cv2.imshow('Select PCB Corners', image)
    cv2.setMouseCallback('Select PCB Corners', select_corners)

    print("Select 4 corners of the PCB by clicking on the image.")
    while len(corners) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    print(f"Corners selected: {corners}")

    # Set fixed dimensions: 640 pixels wide and 480 pixels high
    width, height = 640, 480

    # Define target corners to align the image to a 640x480 space
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    corners = np.array(corners, dtype='float32')

    # Create the transformation matrix using the selected corners
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    
    # Apply the transformation to align the image to 640x480 pixels
    aligned_image_jpg = cv2.warpPerspective(image, matrix, (width, height))

    # Display the aligned image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(aligned_image_jpg, cv2.COLOR_BGR2RGB))
    plt.title("Aligned JPG Image")
    plt.axis('off')
    plt.show()

    # If no RGB matrix is needed, return without processing the temperature matrix
    if not calculate_rgb:
        return aligned_image_jpg, None, corners

    # Define grid dimensions
    m, n = 20, 15  # m: rows, n: columns

    # Calculate the size of each grid cell
    cell_height = height / m
    cell_width = width / n

    # Create a matrix to store the average RGB values (temperature)
    temperature_matrix_jpg = np.zeros((m, n, 3))  # 3 channels for RGB

    # Loop over the image and calculate the average color in each grid cell
    for i in range(m):
        for j in range(n):
            y_start = int(i * cell_height)
            y_end = int((i + 1) * cell_height) if i != m - 1 else height
            x_start = int(j * cell_width)
            x_end = int((j + 1) * cell_width) if j != n - 1 else width

            # Extract the region of interest (ROI) for the current grid cell
            cell_jpg = aligned_image_jpg[y_start:y_end, x_start:x_end]

            # Calculate the average color in the cell
            avg_color_jpg = np.mean(cell_jpg, axis=(0, 1))

            # Store the average color in the temperature matrix
            temperature_matrix_jpg[i, j] = avg_color_jpg

    # Display the temperature matrix (average RGB values) as an image
    plt.figure(figsize=(8, 6))
    plt.imshow(temperature_matrix_jpg.astype(int))
    plt.title("Temperature Matrix (Average Color per Cell) - JPG")
    plt.axis('off')
    plt.show()

    # Print a portion of the RGB matrix to ensure it's not grayscale
    print("Sample of the RGB matrix from the JPG image:")
    print(temperature_matrix_jpg[:5, :5, :])  # Print the first 5x5 elements of the RGB matrix

    return aligned_image_jpg, temperature_matrix_jpg, corners


def select_corners_tif(image_path, tif_data, temp_png_path):
    """
    Creates a temporary .png file to allow corner selection for .tif images,
    and then applies the same transformations to the original radiometric image, resizing to 640x480 pixels.
    Displays both the aligned visual image and the radiometric data heatmap.

    Args:
        image_path (str): Path to the original .tif file.
        tif_data (np.array): Radiometric data from the .tif file.
        temp_png_path (str): Path for the temporary .png file.

    Returns:
        aligned_visual_tif (np.array): Visually aligned image from the .tif file.
        aligned_radiometric_data (np.array): Aligned radiometric data.
    """
    try:
        # Open the .tif image using PIL to generate the .png
        tif_image = Image.open(image_path)

        # Convert image mode if needed
        if tif_image.mode == 'F':
            print("Converting 'F' mode image to 'L' (grayscale) for .png generation")
            tif_image = tif_image.convert('L')  # Convert to grayscale for visual .png
        elif tif_image.mode != 'RGB':
            tif_image = tif_image.convert('RGB')  # Convert to RGB if needed

        # Save the converted image as a temporary .png (preserving quality)
        tif_image.save(temp_png_path, format='PNG')  # PNG preserves quality better than JPEG

        # Load the temporary .png image for corner selection
        png_image = cv2.imread(temp_png_path)

        # Ensure the temporary image loaded correctly
        if png_image is None:
            raise ValueError("Error loading the temporary .png image.")

        # Use the corner selection function (no RGB matrix calculation)
        aligned_visual_tif, _, corners = select_corners_jpg(png_image, calculate_rgb=False)

        # Ensure corners were selected properly
        if corners is None or len(corners) != 4:
            raise ValueError("Corner selection error. Make sure to select all 4 corners.")

        # Apply the same transformations to the radiometric .tif data
        width, height = 640, 480

        # Define target corners to align the image to a 640x480 space
        target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
        corners = np.array(corners, dtype='float32')

        # Create the transformation matrix based on the selected corners
        matrix = cv2.getPerspectiveTransform(corners, target_corners)

        # Apply the transformation to the radiometric .tif data
        aligned_radiometric_data = cv2.warpPerspective(tif_data, matrix, (width, height))

        return aligned_visual_tif, aligned_radiometric_data

    except Exception as e:
        print(f"Error during .tif image processing: {e}")
        return None, None
