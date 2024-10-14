import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure the function is defined before it is called
def initialize_emissivity_matrix(base_emissivity, m, n):
    """
    Initializes an emissivity matrix with the base emissivity value.
    
    Args:
        base_emissivity (float): The base emissivity value for the PCB.
        m (int): Number of rows in the matrix (height).
        n (int): Number of columns in the matrix (width).
    
    Returns:
        np.array: Emissivity matrix of size m x n.
    """
    emissivity_matrix = np.full((m, n), base_emissivity)
    return emissivity_matrix



# Global variables to store shapes and emissivity
shapes = []  # Stores (type, points), where type is 'polygon' or 'circle'
shape_emissivities = []  # Stores emissivity values for each shape

# Callback functions to select points for polygons and circles
def select_polygon(event, x, y, flags, param):
    """Callback function to select corners of a polygon."""
    global polygon_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_coordinates.append((x, y))
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # Mark the selected point
        cv2.imshow('Select Polygon', image_copy)

def select_circle(event, x, y, flags, param):
    """Callback function to select points on the perimeter of a circle."""
    global circle_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        circle_coordinates.append((x, y))
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # Mark the selected point
        cv2.imshow('Select Circle', image_copy)

def define_shape(image_rgb):
    """Allows the user to define polygons (3-6 corners) or circles."""
    global image_copy, polygon_coordinates, circle_coordinates

    while True:
        # Removed the .lower() so the input is case-sensitive, and "No" will be required to stop
        shape_type = input("Enter the number of corners for a polygon (3-6) or 9 for a circle (type 'No' to stop): ")

        if shape_type == 'No':  # Now it's case-sensitive
            break
        elif shape_type in ['3', '4', '5', '6']:  # For polygons
            corners = int(shape_type)
            print(f"Please click on {corners} points to define the polygon.")

            polygon_coordinates = []
            image_copy = image_rgb.copy()

            # Show the image and set the mouse callback for selecting points
            cv2.imshow('Select Polygon', image_copy)
            cv2.setMouseCallback('Select Polygon', select_polygon)

            # Wait until the required number of points are selected
            while len(polygon_coordinates) < corners:
                if cv2.waitKey(1) & 0xFF == 27:  # Break if 'Esc' is pressed
                    break

            # Store the polygon and emissivity if all corners are selected
            if len(polygon_coordinates) == corners:
                shapes.append(('polygon', polygon_coordinates))
                cv2.destroyAllWindows()  # Close the pop-up window after selection
                polygon_emissivity = float(input(f"Enter the emissivity value for this polygon: "))
                shape_emissivities.append(polygon_emissivity)

        elif shape_type == '9':  # For circles
            print("Please click five points on the perimeter of the circle.")

            circle_coordinates = []
            image_copy = image_rgb.copy()

            # Show the image and set the mouse callback for selecting points
            cv2.imshow('Select Circle', image_copy)
            cv2.setMouseCallback('Select Circle', select_circle)

            # Wait until five points are selected
            while len(circle_coordinates) < 5:
                if cv2.waitKey(1) & 0xFF == 27:  # Break if 'Esc' is pressed
                    break

            # Once five points are selected, calculate the circle and store it
            if len(circle_coordinates) == 5:
                circle_coordinates_np = np.array(circle_coordinates, dtype=np.float32)
                (center_x, center_y), radius = cv2.minEnclosingCircle(circle_coordinates_np)
                center = (int(center_x), int(center_y))
                radius = int(radius)
                shapes.append(('circle', (center, radius)))
                cv2.destroyAllWindows()  # Close the pop-up window after selection
                circle_name = input(f"Enter the name for this circle (e.g., material name): ")
                circle_emissivity = float(input(f"Enter the emissivity value for {circle_name}: "))
                shape_emissivities.append(circle_emissivity)

def update_emissivity_matrix(emissivity_matrix, shapes, m, n, height, width, base_emissivity):
    """Updates the emissivity matrix based on defined shapes (polygons and circles)."""
    # Calculate the size of each cell in the grid
    cell_height = height / m
    cell_width = width / n

    for i, (shape_type, shape_data) in enumerate(shapes):
        shape_emissivity = shape_emissivities[i]

        if shape_type == 'circle':
            center, radius = shape_data
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, center, radius, 1, -1)  # Fill the circle with 1

        else:  # For polygons
            polygon = np.array(shape_data, dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)  # Fill the polygon with 1

        # Iterate over the grid nodes and update emissivity
        for i in range(m):
            for j in range(n):
                y_start = int(i * cell_height)
                y_end = int((i + 1) * cell_height) if i != m - 1 else height
                x_start = int(j * cell_width)
                x_end = int((j + 1) * cell_width) if j != n - 1 else width

                # Extract the region of the mask corresponding to the current cell
                cell_mask = mask[y_start:y_end, x_start:x_end]
                area_inside = np.sum(cell_mask)
                total_area = cell_mask.size

                # If part of the cell is inside the shape, update the emissivity
                if area_inside > 0:
                    fraction_inside = area_inside / total_area
                    weighted_emissivity = base_emissivity * (1 - fraction_inside) + shape_emissivity * fraction_inside
                    emissivity_matrix[i, j] = weighted_emissivity

    return emissivity_matrix

def draw_shapes(image_rgb):
    """
    Draws and fills all the defined shapes (polygons and circles) on the image for user verification.
    
    Args:
        image_rgb (np.array): The RGB image to draw shapes on.
    """
    image_with_shapes = image_rgb.copy()  # Copy the original image to draw on

    # Draw filled circles
    for shape_type, shape_data in shapes:
        if shape_type == 'circle':
            center, radius = shape_data
            cv2.circle(image_with_shapes, center, radius, (0, 255, 0), -1)  # Fill circle in green
            cv2.circle(image_with_shapes, center, radius, (255, 0, 0), 2)  # Outline circle in blue

        elif shape_type == 'polygon':
            # Draw filled polygons
            pts = np.array(shape_data, np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv2.fillPoly(image_with_shapes, [pts], (0, 255, 0))  # Fill polygon in green
            cv2.polylines(image_with_shapes, [pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Outline in blue

    # Display the image with the filled shapes
    plt.imshow(cv2.cvtColor(image_with_shapes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Filled Shapes (Polygons and Circles)')
    plt.show()

# Visualize the updated emissivity matrix
def visualize_emissivity_matrix(emissivity_matrix):
    plt.imshow(emissivity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Emissivity')
    plt.title('Emissivity Matrix')
    plt.show()

# Print the names and emissivity values of shapes
def print_shape_info():
    print("\nEmissivity information for defined shapes:")
    for i, (shape_type, emissivity) in enumerate(zip(shapes, shape_emissivities)):
        shape_name = f"{'Polygon' if shape_type[0] == 'polygon' else 'Circle'} {i+1}"
        print(f"{shape_name}: Emissivity = {emissivity}")

def process_emissivity(image_rgb, base_emissivity, m=20, n=15):
    height, width, _ = image_rgb.shape

    # Initialize the emissivity matrix
    emissivity_matrix = initialize_emissivity_matrix(base_emissivity, m, n)

    # Define shapes
    define_shape(image_rgb)

    # **Draw the shapes for user verification**
    draw_shapes(image_rgb)

    # Update the emissivity matrix based on defined shapes
    emissivity_matrix = update_emissivity_matrix(emissivity_matrix, shapes, m, n, height, width, base_emissivity)

    # Visualize the updated emissivity matrix
    visualize_emissivity_matrix(emissivity_matrix)

    # Print the names and emissivity values of the shapes
    print_shape_info()

    return emissivity_matrix

