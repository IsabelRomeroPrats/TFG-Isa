import sys
import os
import numpy as np
import cv2
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QWidget, QHBoxLayout,
                             QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from scipy.constants import sigma
from image_processing import select_corners_jpg, select_corners_tif

class ClickableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.points = []
        self.pixmap_backup = None
        self.callback = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            x = event.pos().x()
            y = event.pos().y()

            if self.pixmap_backup is None:
                self.pixmap_backup = self.pixmap().copy()

            # Reescalar coordenadas a imagen original
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            x_ratio = self.pixmap().width() / label_size.width()
            y_ratio = self.pixmap().height() / label_size.height()
            x = int(x * x_ratio)
            y = int(y * y_ratio)

            self.points.append((x, y))
            self.update_display_with_points()

            if self.callback:
                self.callback(self.points)

    def update_display_with_points(self):
        if self.pixmap_backup:
            image = self.pixmap_backup.toImage()
            image = image.convertToFormat(QImage.Format_RGB32)
            painter_image = QImage(image)
            painter = QPainter(painter_image)
            painter.setPen(Qt.red)

            for pt in self.points:
                painter.drawEllipse(pt[0], pt[1], 6, 6)
            painter.end()

            self.setPixmap(QPixmap.fromImage(painter_image).scaled(self.size(), Qt.KeepAspectRatio))

    def clear_points(self):
        self.points = []
        if self.pixmap_backup:
            self.setPixmap(self.pixmap_backup.scaled(self.size(), Qt.KeepAspectRatio))


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

        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

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
        self.emiss_input.setPlaceholderText("Enter emissivity (0-1)")
        emis_layout.addWidget(QLabel("Emissivity:"))
        emis_layout.addWidget(self.emiss_input)
        layout.addLayout(emis_layout)

        process_button = QPushButton("Apply Correction")
        process_button.clicked.connect(self.apply_correction)
        layout.addWidget(process_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open TIF image', '', 'Image files (*.tif *.jpg *.png)')
        if fname:
            self.image_path = fname
            self.image_data = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            display_image = self.normalize_image(self.image_data)
            h, w = display_image.shape
            qimg = QImage(display_image.data, w, h, w, QImage.Format_Grayscale8)
            self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(500, 400, Qt.KeepAspectRatio))

    def normalize_image(self, image):
        image = image.astype(np.float32) + 273.15  # Convert to Kelvin
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def select_corners(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return

        if self.image_path.endswith('.tif'):
            try:
                T_real = float(self.temp_input.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter a valid temperature before selecting corners.")
                return

            _, aligned_radiometric = select_corners_tif(self.image_data, T_real)
            self.image_data = aligned_radiometric

        QMessageBox.information(self, "Corners Selected", "Corner selection and alignment completed.")

    def apply_correction(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return

        try:
            T_real = float(self.temp_input.text())
            emissivity = float(self.emiss_input.text())
            if not (0 < emissivity <= 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid temperature and emissivity.")
            return

        resized = cv2.resize(self.image_data, (500, 500))
        J_measured = sigma * (resized + 273.15) ** 4
        Eb = sigma * (T_real ** 4)
        J_corrected = emissivity * Eb

        output_folder = f"Test Results/Test_{int(T_real)}"
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, f"J_{int(T_real)}.npy"), J_measured)
        np.save(os.path.join(output_folder, f"corrected_continuous_{int(T_real)}.npy"), J_corrected)

        QMessageBox.information(self, "Saved", f"Results saved to {output_folder}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IRCorrectionApp()
    window.show()
    sys.exit(app.exec_())
