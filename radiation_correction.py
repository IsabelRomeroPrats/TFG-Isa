import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy.constants import sigma  # Boltzmann constant (sigma)

# Define the resolution for both continuous and discrete heatmaps
continuous_shape = (500, 500)  # Continuous 480x640 heatmap
m, n = 20, 20  # Discrete heatmap 15x20

# Define the default parameters
# tau = 0.89


def temperature_to_radiance(temp_matrix):
    return sigma * (temp_matrix ** 4)


def visualize_heatmap(matrix, title, colorbar_label, cmap='hot'):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(label=colorbar_label)
    plt.xlabel("Pixel X-axis")
    plt.ylabel("Pixel Y-axis")
    plt.title(title)
    plt.show()


# Convert to discrete heatmap
def convert_discrete(continuous_heatmap):
    cell_height, cell_width = continuous_shape[0] // m, continuous_shape[1] // n
    discrete_heatmap = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            y_start, y_end = i * cell_height, (i + 1) * cell_height
            x_start, x_end = j * cell_width, (j + 1) * cell_width
            block = continuous_heatmap[y_start:y_end, x_start:x_end]
            discrete_heatmap[i, j] = np.mean(block)

    return discrete_heatmap


def multiply_emissivity(radiance_heatmap, emissivity_matrix):

    # Resize emissivity matrix to match the continuous heatmap
    emissivity_resized = cv2.resize(emissivity_matrix, (continuous_shape[1], continuous_shape[0]), interpolation=cv2.INTER_LINEAR)

    assert radiance_heatmap.shape == continuous_shape, "La imagen de radiancia no tiene el tamaño esperado"
    assert emissivity_matrix.shape == (m, n), "La matriz de emisividad discreta no tiene el tamaño esperado"
 
    # Create Discrete Heatmap
    radiance_discrete_heatmap = convert_discrete(radiance_heatmap)

    # Apply emissivity correction to both heatmaps
    radiance_heatmap_continuous = radiance_heatmap * emissivity_resized
    radiance_heatmap_discrete = radiance_discrete_heatmap * emissivity_matrix

    return radiance_heatmap_continuous, radiance_heatmap_discrete


def divide_emissivity(radiance_heatmap, emissivity_matrix):
 
    # Resize emissivity matrix to match the continuous heatmap
    emissivity_resized = cv2.resize(emissivity_matrix, (continuous_shape[1], continuous_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Create Discrete Heatmap
    radiance_discrete_heatmap = convert_discrete(radiance_heatmap)

    # Apply emissivity correction to both heatmaps
    radiance_heatmap_continuous = radiance_heatmap / emissivity_resized
    radiance_heatmap_discrete = radiance_discrete_heatmap / emissivity_matrix

    return radiance_heatmap_continuous, radiance_heatmap_discrete


## CORRECTION IMAGE

def correction_image(temperature, heatmap, emissivity_matrix):

    " R =  sigma * T_captured^4 - epsilon * sigma * T_ideal^4 "

    " R = sigma * T_ideal^4 - epsilon * sigma * T_captured^4"

    # Generate both heatmaps
    nonradiance_continuous_ideal_heatmap = np.full(continuous_shape, temperature, dtype=np.float32)
    ideal_heatmap = temperature_to_radiance(nonradiance_continuous_ideal_heatmap)

    radiance_heatmap = temperature_to_radiance(heatmap)
    radiance_heatmap = cv2.resize(radiance_heatmap, (continuous_shape[1], continuous_shape[0]), interpolation=cv2.INTER_LINEAR)

    visualize_heatmap(radiance_heatmap, f"Primera Radiancia (T={temperature}K)", "Energy difference (W/m²)")

    # Apply emissivity correction
    ideal_heatmap_continuous, _ = multiply_emissivity(ideal_heatmap, emissivity_matrix)

    # Math
    correction_T =  radiance_heatmap - ideal_heatmap_continuous

    # Convert to discrete
    correction_T_discrete = convert_discrete(correction_T)

    # Visualize the results
    visualize_heatmap(correction_T, f"Error Radiation Heatmap Conitnuous (T={temperature}K)", "Energy difference (W/m²)")
    visualize_heatmap(correction_T_discrete, f"Error Radiation Heatmap Discrete (T={temperature}K)", "Energy difference (W/m²)")

    # Saving the results
    folder_name = f"T{int(temperature)}"
    os.makedirs(folder_name, exist_ok=True)

    file_path = os.path.join(folder_name, f"correction_T{int(temperature)}.npy")
    np.save(file_path, correction_T)

    file_path = os.path.join(folder_name, f"correction_T{int(temperature)}_discrete.npy")
    np.save(file_path, correction_T_discrete)

    return correction_T, correction_T_discrete


## TRUE TEMPERATURE

def final_image(temperature, heatmap, correction_image, emissivity_matrix, tau):

    " T_real = [(sigma * T_heatmap^4 - R)/(tau * epsilon * sigma)]^(1/4)"

    # Generate both heatmaps
    radiance_heatmap = temperature_to_radiance(heatmap) # captured by the camera === J
    radiance_heatmap = cv2.resize(radiance_heatmap, (continuous_shape[1], continuous_shape[0]), interpolation=cv2.INTER_LINEAR)

    visualize_heatmap(radiance_heatmap, f"Radiance map before correction (T={temperature}K)", "Energy difference (W/m²)")
    visualize_heatmap(correction_image, f"Radiance reflection used for correction (T={temperature}K)", "Energy difference (W/m²)")


    # Apply emissivity correction
    minus = radiance_heatmap - correction_image
    radiometric_heatmap, _ = divide_emissivity(minus, emissivity_matrix)
    
    visualize_heatmap(radiometric_heatmap, f"Radiance heatmap after emissivity (T={temperature}K)", "Energy difference (W/m²)")

    # Apply transmissivity
    true_radiometric_heatmap = radiometric_heatmap / tau # true radiance

    # Obtain temperature
    true_temperature = (true_radiometric_heatmap / sigma)**(1/4)

    # Convert to discrete
    true_radiometric_heatmap_discrete = convert_discrete(true_radiometric_heatmap)
    true_temperature_discrete = convert_discrete(true_temperature)

    # Visualize the results

    visualize_heatmap(true_radiometric_heatmap, f"Radiometric Heatmap Continuous (T={temperature}K)", "Energy difference (W/m²)")
    visualize_heatmap(true_radiometric_heatmap_discrete, f"Radiometric Heatmap Discrete (T={temperature}K)", "Energy difference (W/m²)")

    visualize_heatmap(true_temperature, f"True Temperature Heatmap Continuous (T={temperature}K)", "(K)")
    visualize_heatmap(true_temperature_discrete, f"True Temperature Heatmap Discrete (T={temperature}K)", "(K)")

   # Saving the results
    folder_name = f"T{int(temperature)}"
    os.makedirs(folder_name, exist_ok=True)

    file_path = os.path.join(folder_name, f"radiometric_T{int(temperature)}.npy")
    np.save(file_path, true_radiometric_heatmap)
    file_path = os.path.join(folder_name, f"radiometric_T{int(temperature)}_discrete.npy")
    np.save(file_path, true_radiometric_heatmap_discrete)

    file_path = os.path.join(folder_name, f"true_temperature_T{int(temperature)}.npy")
    np.save(file_path, true_temperature)
    file_path = os.path.join(folder_name, f"true_temperature_T{int(temperature)}_discrete.npy")
    np.save(file_path, true_temperature_discrete)