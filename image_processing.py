# image_processing.py

import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_and_display_image(image_name):
    """
    Loads and displays the image from the 'Images' folder based on its extension.
    
    Args:
        image_name (str): The name of the image file, including extension (e.g., 'image.jpg').
    
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

    # Load and display image based on extension
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
        # Load .tif image (this is a placeholder, might need specific handling for radiometric .tif)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not load .tif image {image_name}")
            return None
        plt.imshow(image, cmap='gray')  # Display .tif as grayscale image for now
        plt.title(f"Loaded .tif Image: {image_name}")
        plt.axis('off')
        plt.show()
        return image
    
    else:
        print(f"Unsupported file format: {extension}")
        return None

import numpy as np
import cv2
import matplotlib.pyplot as plt

def select_corners_jpg(image, calculate_rgb=True):
    """
    Permite al usuario seleccionar las esquinas manualmente en imágenes .jpg o .png.
    Luego, alinea la imagen en base a las esquinas seleccionadas y la adapta a 640x480 píxeles.
    Si calculate_rgb es True, también calcula y muestra la matriz de temperatura RGB.

    Args:
        image (np.array): La imagen cargada.
        calculate_rgb (bool): Si es True, calcula la matriz de temperatura RGB.

    Returns:
        aligned_image_jpg (np.array): La imagen transformada y alineada.
        temperature_matrix_jpg (np.array or None): Matriz de temperatura (RGB promedio) para cada celda, o None si calculate_rgb es False.
        corners (list): Las coordenadas de las esquinas seleccionadas.
    """
    corners = []

    def select_corners(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select PCB Corners', image)

    # Mostrar la imagen para seleccionar las esquinas
    cv2.imshow('Select PCB Corners', image)
    cv2.setMouseCallback('Select PCB Corners', select_corners)

    print("Seleccione 4 esquinas del PCB haciendo clic en la imagen.")
    while len(corners) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    print(f"Esquinas seleccionadas: {corners}")

    # Definir las medidas fijas: 640 píxeles de ancho y 480 píxeles de alto
    width, height = 640, 480

    # Definir las esquinas objetivo para alinear la imagen en un espacio de 640x480 píxeles
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    corners = np.array(corners, dtype='float32')

    # Crear la matriz de transformación usando las esquinas seleccionadas
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    
    # Aplicar la transformación para alinear la imagen a 640x480 píxeles
    aligned_image_jpg = cv2.warpPerspective(image, matrix, (width, height))

    # Mostrar la imagen alineada
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(aligned_image_jpg, cv2.COLOR_BGR2RGB))
    plt.title("Imagen JPG Alineada")
    plt.axis('off')
    plt.show()

    # Si no se debe calcular la matriz RGB, regresar sin procesar la matriz de temperatura
    if not calculate_rgb:
        return aligned_image_jpg, None, corners

    # Dimensiones de la malla
    m, n = 20, 15  # m: filas, n: columnas

    # Calcular el tamaño de cada celda en la malla
    cell_height = height / m
    cell_width = width / n

    # Crear una matriz para almacenar los valores promedio de color (temperatura RGB)
    temperature_matrix_jpg = np.zeros((m, n, 3))  # 3 canales para RGB

    # Recorrer la imagen y calcular el color promedio en cada celda de la malla
    for i in range(m):
        for j in range(n):
            y_start = int(i * cell_height)
            y_end = int((i + 1) * cell_height) if i != m - 1 else height
            x_start = int(j * cell_width)
            x_end = int((j + 1) * cell_width) if j != n - 1 else width

            # Extraer la región de interés (ROI) para la celda actual
            cell_jpg = aligned_image_jpg[y_start:y_end, x_start:x_end]

            # Calcular el color promedio en la celda (sin promediar entre canales RGB)
            avg_color_jpg = np.mean(cell_jpg, axis=(0, 1))

            # Almacenar el color promedio en la matriz de temperatura
            temperature_matrix_jpg[i, j] = avg_color_jpg

    # Mostrar la matriz de temperatura (valores RGB promedio) como una imagen
    plt.figure(figsize=(8, 6))
    plt.imshow(temperature_matrix_jpg.astype(int))
    plt.title("Matriz de Temperatura (Color Promedio por Celda) - JPG")
    plt.axis('off')
    plt.show()

    # Imprimir un tramo de la matriz RGB para asegurar que no está en escala de grises
    print("Tramo de la matriz RGB de la imagen JPG:")
    print(temperature_matrix_jpg[:5, :5, :])  # Imprimir los primeros 5x5 elementos de la matriz RGB

    return aligned_image_jpg, temperature_matrix_jpg, corners




def select_corners_tif(image_path, tif_data, temp_jpg_path):
    """
    Crea un archivo .jpg temporal para permitir la selección de esquinas en imágenes .tif
    y luego aplica las mismas transformaciones a la imagen radiométrica original adaptada a 640x480 píxeles.
    Muestra tanto la imagen visual alineada como el heatmap de los datos radiométricos.

    Args:
        image_path (str): Ruta del archivo .tif original.
        tif_data (np.array): Datos radiométricos del archivo .tif.
        temp_jpg_path (str): Ruta del archivo .jpg temporal.

    Returns:
        aligned_visual_tif (np.array): Imagen visualmente alineada del .tif.
        aligned_radiometric_data (np.array): Datos radiométricos alineados.
    """
    try:
        # Abrir la imagen .tif usando PIL para generar el .jpg
        tif_image = Image.open(image_path)

        # Verificar el modo de la imagen y convertirla si es necesario
        if tif_image.mode == 'F':
            print("Convirtiendo la imagen de modo 'F' a 'L' (escala de grises) para generar .jpg")
            tif_image = tif_image.convert('L')  # Convertir a escala de grises (solo para visualización .jpg)
        elif tif_image.mode != 'RGB':
            tif_image = tif_image.convert('RGB')  # Convertir a RGB si es necesario

        # Guardar la imagen convertida como un archivo .jpg temporal (solo para selección de esquinas)
        tif_image.save(temp_jpg_path)

        # Cargar la imagen .jpg temporal para la selección de esquinas (no se usará para procesamiento RGB)
        jpg_image = cv2.imread(temp_jpg_path)

        # Verifica que la imagen temporal se haya cargado correctamente
        if jpg_image is None:
            raise ValueError("Error al cargar la imagen temporal .jpg.")

        # Usar la función de selección de esquinas en la imagen .jpg (sin calcular matriz RGB)
        aligned_visual_tif, _, corners = select_corners_jpg(jpg_image, calculate_rgb=False)

        # Verifica que se hayan seleccionado las esquinas correctamente
        if corners is None or len(corners) != 4:
            raise ValueError("Error en la selección de esquinas. Asegúrate de seleccionar las 4 esquinas.")

        # Aplicar las mismas transformaciones a los datos radiométricos .tif
        width, height = 640, 480

        # Definir las esquinas objetivo para alinear la imagen en un espacio de 640x480 píxeles
        target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
        corners = np.array(corners, dtype='float32')

        # Crear la matriz de transformación basada en las esquinas seleccionadas
        matrix = cv2.getPerspectiveTransform(corners, target_corners)

        # Aplicar la transformación a los datos radiométricos .tif (manteniendo los datos radiométricos)
        aligned_radiometric_data = cv2.warpPerspective(tif_data, matrix, (width, height))

        # # Mostrar la imagen visualmente alineada
        # plt.figure(figsize=(8, 6))
        # plt.imshow(cv2.cvtColor(aligned_visual_tif, cv2.COLOR_BGR2RGB))
        # plt.title("Imagen .tif visualmente alineada")
        # plt.axis('off')
        # plt.show()

        return aligned_visual_tif, aligned_radiometric_data

    except Exception as e:
        print(f"Error durante el procesamiento de la imagen .tif: {e}")
        return None, None