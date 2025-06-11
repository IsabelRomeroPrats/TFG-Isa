import numpy as np
import cv2
from scipy.constants import sigma  # Boltzmann constant (sigma)
from PIL import Image

# Define the constants for the new E_ref equation
k1 = 1.0026
k2 = 60.392

def calculate_radiation(E_tot, e, T_real):
    """
    Calculate E_ref using the formula:
    E_ref = E_tot - E_emit

    Args:
        E_tot (float): Total radiation energy at the selected point.
        e (float): Real emissivity of the material.
        T_real (float): Real temperature (in Kelvin) from the thermocouple.

    Returns:
        float: Reflected radiation energy (E_ref).
    """
    # Calculate E_emit
    E_emit = e * sigma * T_real**4

    # Calculate E_ref
    E_ref = E_tot - E_emit

    return E_ref

def calculate_Tref(E_ref):
    """
    Calculate T_ref from E_ref using the formula:
    E_ref = k1 * sigma * T_ref^4 + k2

    Args:
        E_ref (float): Reflected radiation energy.

    Returns:
        float: Calculated reference temperature (T_ref) in Kelvin.
    """
    # Rearrange the equation to solve for T_ref
    T_ref = ((E_ref - k2) / (k1 * sigma)) ** 0.25
    return T_ref

def normalize_tif_for_display(tif_data):
    """
    Normalize .tif data to the 0-255 range for visualization.

    Args:
        tif_data (np.array): Radiometric .tif data.

    Returns:
        np.array: Normalized uint8 image for display.
    """
    min_val = np.min(tif_data)
    max_val = np.max(tif_data)

    if min_val != max_val:
        normalized_data = np.interp(tif_data, (min_val, max_val), (0, 255))
    else:
        # Edge case: if all values are identical, create a uniform black image
        normalized_data = np.zeros_like(tif_data)

    return normalized_data.astype(np.uint8)

def get_user_inputs(image, aligned_radiometric_data):
    """
    Allow the user to click on specific points on the image and input the required values
    to calculate Eref (Etot, real emissivity, and real temperature).

    Args:
        image (np.array): The image to display for point selection.
        aligned_radiometric_data (np.array): Matrix containing aligned radiometric data.

    Returns:
        list: List of dictionaries with Etot, Eemit, Eref, Tref, and Tfin values for each selected point.
    """
    # Convert radiometric data to a displayable image
    display_image = normalize_tif_for_display(aligned_radiometric_data)

    user_points = []  # To store the pixel coordinates selected by the user
    Etot_values = []  # To store Etot values for the selected points
    Eemit_values = []  # To store Eemit values for the selected points
    Eref_values = []   # To store the calculated Eref values
    Tref_values = []   # To store the calculated Tref values
    Tfin_values = []   # To store the calculated Tfin values

    def click_event(event, x, y, flags, param):
        """Callback function for mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Record the selected point
            user_points.append((x, y))
            
            # Get the radiometric value from the aligned radiometric data matrix
            T_camera = aligned_radiometric_data[y, x]  # Assuming the matrix matches image coordinates

            # Calculate Etot
            Etot = sigma * T_camera**4
            print(f"Selected point: ({x}, {y}), T_camera: {T_camera} K, Etot: {Etot}")

            # Get real emissivity and temperature from the user
            e_real = float(input(f"Enter the real emissivity (e) for point ({x}, {y}): "))
            T_real = float(input(f"Enter the real temperature (T_real in Kelvin) for point ({x}, {y}): "))

            # Calculate Eemit and Eref
            E_emit = e_real * sigma * T_real**4
            E_ref = Etot - E_emit

            # Calculate T_ref from E_ref
            T_ref = calculate_Tref(E_ref)

            # Calculate T_fin based on the formula
            T_fin = (T_camera - T_ref) / e_real

            # Store the values
            Etot_values.append(Etot)
            Eemit_values.append(E_emit)
            Eref_values.append(E_ref)
            Tref_values.append(T_ref)
            Tfin_values.append(T_fin)

            print(f"E_emit: {E_emit}, E_ref: {E_ref}, T_ref: {T_ref}, T_fin: {T_fin}")

    # Display the normalized radiometric image and set up the click event
    cv2.imshow('Select Points', display_image)
    cv2.setMouseCallback('Select Points', click_event)

    print("Click on points where thermocouples are located. Press 'q' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(user_points) >= 5:  # Quit when 'q' is pressed or enough points selected
            break

    cv2.destroyAllWindows()

    # Package the data into a list of dictionaries
    data = []
    for i in range(len(user_points)):
        data.append({
            'point': user_points[i],
            'Etot': Etot_values[i],
            'Eemit': Eemit_values[i],
            'Eref': Eref_values[i],
            'Tref': Tref_values[i],
            'Tfin': Tfin_values[i]
        })

    return data
