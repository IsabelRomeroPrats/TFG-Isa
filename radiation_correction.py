import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
from scipy.constants import sigma  # Boltzmann constant (sigma)

# Define the resolution for both continuous and discrete heatmaps
continuous_shape = (500, 500)  # Continuous 480x640 heatmap
m, n = 20, 20  # Discrete heatmap 15x20

# Define the default parameters
# tau = 0.89

def save_dual_maps(prefix, radiance_map, out_dir, cmap="hot"):
    """
    Guarda dos versiones del mismo mapa:
    - Radiancia (W/m^2): archivo *_radiance_*.npy y *_radiance_*.png
    - Kelvin (temperatura de brillo): archivo *_kelvin_*.npy y *_kelvin_*.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # Radiancia
    np.save(os.path.join(out_dir, f"{prefix}_radiance.npy"), radiance_map)
    fig_r, ax_r = plt.subplots(figsize=(6, 5))
    im_r = ax_r.imshow(radiance_map, cmap=cmap)
    fig_r.colorbar(im_r, ax=ax_r, label="Radiance (W/m²)")
    ax_r.set_title(f"{prefix.replace('_',' ').title()} — Radiance")
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(out_dir, f"{prefix}_radiance.png"), dpi=150)
    plt.close(fig_r)

    # Kelvin (temperatura de brillo equivalente)
    kelvin_map = np.maximum(radiance_map, 0.0)**0.25 / (sigma**0.25)  # (R/sigma)**0.25
    np.save(os.path.join(out_dir, f"{prefix}_kelvin.npy"), kelvin_map)
    fig_k, ax_k = plt.subplots(figsize=(6, 5))
    im_k = ax_k.imshow(kelvin_map, cmap=cmap)
    fig_k.colorbar(im_k, ax=ax_k, label="Temperature (K)")
    ax_k.set_title(f"{prefix.replace('_',' ').title()} — Kelvin")
    fig_k.tight_layout()
    fig_k.savefig(os.path.join(out_dir, f"{prefix}_kelvin.png"), dpi=150)
    plt.close(fig_k)


def _save_rgb_with_colorbar(arr, title, label, out_png, cmap="jet"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)
    ax.set_title(title)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    fig.text(0.5, -0.02, f"min={vmin:.6g}   max={vmax:.6g}",
        ha="center", va="top", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _save_radiance_and_kelvin(prefix, R_map, folder):
    """
    Guarda radiancia (W/m^2) y Kelvin (brillo) en:
      - {prefix}_radiance.tif  (float32)
      - {prefix}_radiance_rgb.png
      - {prefix}_kelvin.tif    (float32)
      - {prefix}_kelvin_rgb.png
    """
    os.makedirs(folder, exist_ok=True)

    R = np.asarray(R_map, dtype=np.float32)

    # Radiancia .tif
    cv2.imwrite(os.path.join(folder, f"{prefix}_radiance.tif"), R)
    # Radiancia .png (RGB)
    _save_rgb_with_colorbar(
        R,
        f"{prefix} — Radiance",
        "Radiance (W/m²)",
        os.path.join(folder, f"{prefix}_radiance_rgb.png")
    )

    # Kelvin (temperatura de brillo)
    K = (np.maximum(R, 0.0) / sigma) ** 0.25
    K = K.astype(np.float32)

    # Kelvin .tif
    cv2.imwrite(os.path.join(folder, f"{prefix}_kelvin.tif"), K)
    # Kelvin .png (RGB)
    _save_rgb_with_colorbar(
        K,
        f"{prefix} — Kelvin",
        "Temperature (K)",
        os.path.join(folder, f"{prefix}_kelvin_rgb.png")
    )

########################################

########### CÓDIGO PRINCIPAL ###########

########################################

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

def correction_image(temperature, heatmap, tau, emissivity_matrix):

    " J = tau * [epsilon * sigma * T_ideal^4 + R] = sigma * T_captured^4 "

    " R =  (sigma * T_captured^4) / tau - epsilon * sigma * T_ideal^4 "

    # Generate both heatmaps
    nonradiance_continuous_ideal_heatmap = np.full(continuous_shape, temperature, dtype=np.float32)
    ideal_heatmap = temperature_to_radiance(nonradiance_continuous_ideal_heatmap)

    radiance_heatmap = temperature_to_radiance(heatmap)
    radiance_heatmap = cv2.resize(radiance_heatmap, (continuous_shape[1], continuous_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply emissivity correction
    ideal_heatmap_continuous, _ = multiply_emissivity(ideal_heatmap, emissivity_matrix)

    # Apply tau correction
    radiance_heatmap = radiance_heatmap / tau
    # ideal_heatmap_continuous = ideal_heatmap_continuous * tau

    # Math
    correction_T =  radiance_heatmap - ideal_heatmap_continuous

    # Convert to discrete
    correction_T_discrete = convert_discrete(correction_T)

    folder_name = f"T{int(temperature)}"
    # 1) Captured (J)
    _save_radiance_and_kelvin(prefix=f"captured_T{int(temperature)}",
                            R_map=radiance_heatmap,
                            folder=folder_name)
    # 2) CorrectionImage (ε σ T_ideal^4)
    _save_radiance_and_kelvin(prefix=f"correctionImage_T{int(temperature)}",
                            R_map=ideal_heatmap_continuous,
                            folder=folder_name)
    # 3) Reflexión R (sin τ)
    _save_radiance_and_kelvin(prefix=f"reflexion_R_T{int(temperature)}",
                            R_map=correction_T,
                            folder=folder_name)


    return correction_T, correction_T_discrete


## TRUE TEMPERATURE

def final_image(temperature, heatmap, correction_image, emissivity_matrix, tau, name_suffix=""):

    " T_heatmap^4 * sigma = tau * [(sigma * T_real^4 * epsilon) + R]"

    " T_real = [((sigma * T_heatmap^4) / tau - R))/(epsilon * sigma)]^(1/4)"

    # Generate both heatmaps
    radiance_heatmap = temperature_to_radiance(heatmap) # captured by the camera === J
    radiance_heatmap = cv2.resize(radiance_heatmap, (continuous_shape[1], continuous_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply transmissivity
    radiometric_heatmap = radiance_heatmap / tau # true radiance

    # Apply emissivity correction
    #╔ correction_image = correction_image * tau
    minus = radiometric_heatmap - correction_image
    true_radiometric_heatmap, _ = divide_emissivity(minus, emissivity_matrix)
 
    # Obtain temperature
    true_temperature = (true_radiometric_heatmap / sigma)**(1/4)

    # Convert to discrete
    true_radiometric_heatmap_discrete = convert_discrete(true_radiometric_heatmap)
    true_temperature_discrete = convert_discrete(true_temperature)

    return true_temperature, true_temperature_discrete