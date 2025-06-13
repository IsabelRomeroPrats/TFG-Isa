import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QWidget, QHBoxLayout,
                             QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.constants import sigma
from image_processing import warp_perspective_from_points

import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QWidget, QHBoxLayout,
                             QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.constants import sigma
from image_processing import warp_perspective_from_points
from radiation_correction import final_image, correction_image


class ClickableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.image_np = None
        self.points = []
        self.shapes = []
        self.shape_emissivities = []
        self.callback = None
        self.mode = 'corners'
        self.expected_points = 4
        self.canvas_callback = None
        self.emissivity_matrix_callback = None

    def set_numpy_image(self, image):
        self.image_np = image.copy()
        self.points.clear()
        self.shapes.clear()
        self.shape_emissivities.clear()
        self.update_display()
        if self.canvas_callback:
            self.canvas_callback()

    def mousePressEvent(self, event):
        if self.image_np is None or self.pixmap() is None:
            return

        pixmap = self.pixmap()
        label_size = self.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_w, scaled_h = scaled_pixmap.width(), scaled_pixmap.height()
        x_offset = (label_size.width() - scaled_w) // 2
        y_offset = (label_size.height() - scaled_h) // 2

        x_click = event.pos().x()
        y_click = event.pos().y()
        if not (x_offset <= x_click <= x_offset + scaled_w and y_offset <= y_click <= y_offset + scaled_h):
            return

        img_h, img_w = self.image_np.shape
        x_rel = (x_click - x_offset) / scaled_w
        y_rel = (y_click - y_offset) / scaled_h
        x_real = int(x_rel * img_w)
        y_real = int(y_rel * img_h)

        self.points.append((x_real, y_real))
        self.update_display()

        if len(self.points) == self.expected_points:
            if self.mode == 'corners':
                if self.callback:
                    self.callback(self.points)
                return

            from PyQt5.QtWidgets import QInputDialog
            emissivity, ok = QInputDialog.getDouble(
                self, "Emissivity", "Enter emissivity value (0-1):", min=0.01, max=1.0, decimals=3)
            if not ok:
                self.points.clear()
                self.update_display()
                return

            if self.mode == 'circle':
                pts = np.array(self.points, dtype=np.float32)
                (cx, cy), radius = cv2.minEnclosingCircle(pts)
                self.shapes.append(('circle', ((int(cx), int(cy)), int(radius))))
            elif self.mode == 'polygon':
                if len(self.points) == 3:
                    self.shapes.append(('triangle', self.points.copy()))
                else:
                    self.shapes.append(('polygon', self.points.copy()))

            self.shape_emissivities.append(emissivity)
            if self.callback:
                self.callback(self.shapes[-1], emissivity)

            if self.canvas_callback:
                self.canvas_callback()
            if self.emissivity_matrix_callback:
                self.emissivity_matrix_callback()

            self.points = []
            self.update_display()

    def update_display(self):
        if self.image_np is None:
            return

        img_display = cv2.normalize(self.image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        for i, (tipo, datos) in enumerate(self.shapes):
            color = (0, 255, 0)
            if tipo == 'circle':
                center, radius = datos
                cv2.circle(img_display, center, radius, color, -1)
                cv2.putText(img_display, str(i+1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            elif tipo in ('polygon', 'triangle'):
                pts = np.array(datos, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img_display, [pts], color)
                centroid = np.mean(pts[:, 0, :], axis=0).astype(int)
                cv2.putText(img_display, str(i+1), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for pt in self.points:
            cv2.circle(img_display, pt, 6, (0, 0, 255), -1)

        h, w, _ = img_display.shape
        qimg = QImage(img_display.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio))

class IRCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR Correction Tool")
        self.setGeometry(100, 100, 800, 600)

        self.image_path = None
        self.image_data = None
        self.temperature = None
        self.emissivity = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)  # más estrecho
        layout.setSpacing(2)  # muy poco espacio vertical

        image_container = QVBoxLayout()
        image_container.setSpacing(0)
        image_container.setContentsMargins(0, 0, 0, 0)

        self.image_label = ClickableImageLabel()
        self.image_label.setMinimumSize(500, 400)
        image_container.addWidget(self.image_label)

        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignCenter)
        self.image_name_label.setStyleSheet("font-size: 9pt; color: gray;")
        image_container.addWidget(self.image_name_label)

        layout.addLayout(image_container)

        load_button = QPushButton("Load .tif Image")
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        temp_layout = QHBoxLayout()
        self.temp_input = QLineEdit()
        self.temp_input.setPlaceholderText("Enter temperature (K)")
        temp_layout.addWidget(QLabel("Temperature:"))
        temp_layout.addWidget(self.temp_input)
        layout.addLayout(temp_layout)

        select_button = QPushButton("Select Corners")
        select_button.clicked.connect(self.select_corners)
        layout.addWidget(select_button)

        emis_layout = QHBoxLayout()
        self.emiss_input = QLineEdit()
        self.emiss_input.setPlaceholderText("Enter base emissivity (0-1)")
        emis_layout.addWidget(QLabel("Emissivity:"))
        emis_layout.addWidget(self.emiss_input)
        layout.addLayout(emis_layout)

        # Sección de título Emissivity Shape
        emis_title = QLabel("Add Emissivity Shape")
        emis_title.setAlignment(Qt.AlignCenter)
        emis_title.setStyleSheet("font-weight: bold; font-size: 8pt; margin-top: 10px;")
        layout.addWidget(emis_title)
        # Botón: Iniciar forma
        self.start_button = QPushButton("Start Drawing Shape")
        self.start_button.clicked.connect(self.start_shape)
        layout.addWidget(self.start_button)
        # Botón: Terminar forma
        self.finish_button = QPushButton("Finish Shape")
        self.finish_button.clicked.connect(self.finish_shape)
        layout.addWidget(self.finish_button)

        tau_layout = QHBoxLayout()
        self.tau_input = QLineEdit()
        self.tau_input.setPlaceholderText("Enter tau for crystal (0-1)")
        emis_layout.addWidget(QLabel("Tau:"))
        emis_layout.addWidget(self.tau_input)
        layout.addLayout(tau_layout)

        # Contenedor para editar emisividades
        self.shapes_layout = QVBoxLayout()
        self.shapes_layout.setSpacing(3)
        layout.addLayout(self.shapes_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        show_matrix_button = QPushButton("Show Emissivity Matrix")
        show_matrix_button.clicked.connect(self.show_emissivity_matrix)
        layout.addWidget(show_matrix_button)

        process_button = QPushButton("Apply Correction")
        process_button.clicked.connect(self.apply_correction)
        layout.addWidget(process_button)


    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open TIF image', '', 'Image files (*.tif *.jpg *.png)')
        if fname:
            self.image_path = fname
            image_name = os.path.basename(fname)
            self.image_name_label.setText(f"Loaded image: {image_name}")

            self.image_data = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

            if self.image_data is not None:
                self.image_label.set_numpy_image(self.image_data)

    def normalize_image(self, image):
        image = image.astype(np.float32) + 273.15  # Convert to Kelvin
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def select_corners(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return

        def on_4_points(points):
            from image_processing import warp_perspective_from_points
            aligned = warp_perspective_from_points(self.image_data, points)
            self.image_data = aligned
            self.image_label.set_numpy_image(aligned)
            QMessageBox.information(self, "Aligned", "Image has been aligned based on corners.")

        self.image_label.points.clear()
        self.image_label.expected_points = 4
        self.image_label.mode = 'corners'
        self.image_label.callback = on_4_points
        self.image_label.update_display()
        QMessageBox.information(self, "Select", "Click 4 corners on the image.")

    def activate_shape_selection(self):
        self.image_label.points.clear()
        self.image_label.expected_points = 4  # por ahora, polígono de 4 lados
        self.image_label.mode = 'polygon'

        def on_shape_added(shape_data, emissivity):
            index = len(self.image_label.shapes)
            QMessageBox.information(self, "Shape Added", f"Shape {index} added with emissivity {emissivity:.2f}")
            # Aquí podrías actualizar una lista visual de formas si quieres

        self.image_label.callback = on_shape_added
        self.image_label.update_display()
        QMessageBox.information(self, "Select", "Click polygon points on the image.")

    def start_shape(self):
        self.image_label.points.clear()
        self.image_label.mode = 'polygon'
        self.image_label.expected_points = 9999  # número indefinido
        self.image_label.update_display()
        QMessageBox.information(self, "Shape Mode", "Click to add points. Then click 'Finish Shape'.")

    def finish_shape(self):
        points = self.image_label.points.copy()

        if len(points) < 3:
            QMessageBox.warning(self, "Too few points", "You need at least 3 points to define a shape.")
            return

        shape_type = 'triangle' if len(points) == 3 else 'polygon'

        if len(points) == 5:
            reply = QMessageBox.question(
                self, "Shape Type",
                "Is this shape a circle?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                shape_type = 'circle'

        from PyQt5.QtWidgets import QInputDialog
        emissivity, ok = QInputDialog.getDouble(self, "Emissivity", "Enter emissivity (0-1):", min=0.01, max=1.0, decimals=3)

        if not ok:
            return

        if shape_type == 'circle':
            pts = np.array(points, dtype=np.float32)
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            self.image_label.shapes.append(('circle', ((int(cx), int(cy)), int(radius))))
        else:
            self.image_label.shapes.append(('polygon', points))

        self.image_label.shape_emissivities.append(emissivity)
        self.image_label.points.clear()
        self.image_label.update_display()
        shape_type_label = shape_type.capitalize()
        self.add_shape_entry(len(self.image_label.shapes) - 1, shape_type_label, emissivity)


    def add_shape_entry(self, shape_index, shape_type, current_emissivity):
        layout = QHBoxLayout()
        label = QLabel(f"Shape {shape_index + 1} ({shape_type})")
        label.setFixedWidth(120)

        emiss_input = QLineEdit()
        emiss_input.setText(f"{current_emissivity:.3f}")
        emiss_input.setFixedWidth(60)

        def update_emissivity():
            try:
                value = float(emiss_input.text())
                if not (0 < value <= 1):
                    raise ValueError
                self.image_label.shape_emissivities[shape_index] = value
                self.image_label.update_display()
            except ValueError:
                QMessageBox.warning(self, "Invalid", "Enter a valid emissivity between 0 and 1.")

        emiss_input.editingFinished.connect(update_emissivity)
        layout.addWidget(label)
        layout.addWidget(QLabel("Emissivity:"))
        layout.addWidget(emiss_input)
        self.shapes_layout.addLayout(layout)

    def build_emissivity_matrix(self, m=20, n=20):
        if self.image_data is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return

        try:
            base_emiss = float(self.emiss_input.text())
            if not (0 < base_emiss <= 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Enter a valid base emissivity.")
            return

        height, width = self.image_data.shape[:2]
        matrix = np.full((m, n), base_emiss)

        # Celdas de la rejilla
        cell_h = height / m
        cell_w = width / n

        for shape, emiss in zip(self.image_label.shapes, self.image_label.shape_emissivities):
            tipo, datos = shape
            mask = np.zeros((height, width), dtype=np.uint8)
            if tipo == 'circle':
                center, radius = datos
                cv2.circle(mask, center, radius, 1, -1)
            else:  # incluye triangle o polygon
                pts = np.array(datos, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)

            for i in range(m):
                for j in range(n):
                    y0, y1 = int(i * cell_h), int((i + 1) * cell_h)
                    x0, x1 = int(j * cell_w), int((j + 1) * cell_w)
                    cell = mask[y0:y1, x0:x1]
                    if cell.size == 0:
                        continue
                    frac = np.sum(cell) / cell.size
                    if frac > 0:
                        matrix[i, j] = base_emiss * (1 - frac) + emiss * frac

        return matrix

    def show_emissivity_matrix(self):
        matrix = self.build_emissivity_matrix()
        if matrix is None:
            return

        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Emissivity')
        plt.title('Emissivity Matrix')
        plt.tight_layout()
        plt.show()

    def apply_correction(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return

        try:
            temperature = float(self.temp_input.text())
            tau = float(self.tau_input.text()) if self.tau_input.text() else 1.0
            if not (0 < tau <= 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid temperature and tau.")
            return

        # Construir ruta del archivo de corrección
        temperature_int = int(round(temperature))
        folder_name = f"T{temperature_int}"
        correction_path = os.path.join(folder_name, f"correction_T{temperature_int}.npy")

        # Obtener matriz de emisividad
        emissivity_matrix = self.build_emissivity_matrix()
        if emissivity_matrix is None:
            return

        # Convertir imagen a Kelvin
        aligned_radiometric_data = self.image_data.astype(np.float32) + 273.15

        if os.path.exists(correction_path):
            correction_file = np.load(correction_path)
            final_image(temperature, aligned_radiometric_data, correction_file, emissivity_matrix)
            QMessageBox.information(self, "Correction Applied", "Used existing correction file.")
        else:
            correction_image(temperature, aligned_radiometric_data, emissivity_matrix)
            QMessageBox.information(self, "Correction Created", "New correction file generated.")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IRCorrectionApp()
    window.show()
    sys.exit(app.exec_())
