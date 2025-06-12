import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QLineEdit, QMessageBox
)
import os

class ThermalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Test Manager")

        # Layout principal
        layout = QVBoxLayout()

        # Botón para cargar imagen
        self.label_path = QLabel("No image selected")
        self.button_load = QPushButton("Load Image")
        self.button_load.clicked.connect(self.load_image)

        # Campo para temperatura
        self.label_temp = QLabel("Enter default temperature (K):")
        self.temp_input = QLineEdit()

        # Botón para ejecutar análisis
        self.button_run = QPushButton("Run Correction")
        self.button_run.clicked.connect(self.run_correction)

        # Agregar widgets al layout
        layout.addWidget(self.label_path)
        layout.addWidget(self.button_load)
        layout.addWidget(self.label_temp)
        layout.addWidget(self.temp_input)
        layout.addWidget(self.button_run)

        self.setLayout(layout)
        self.image_path = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select thermal image", "", "Images (*.tif *.jpg *.png)")
        if path:
            self.image_path = path
            self.label_path.setText(f"Loaded: {os.path.basename(path)}")

    def run_correction(self):
        try:
            T = float(self.temp_input.text())
            if not self.image_path:
                raise Exception("No image loaded.")
            # Aquí llamas a tu función apply_ir_correction(...)
            QMessageBox.information(self, "Success", f"Correction applied at T={T} K for\n{self.image_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThermalApp()
    window.resize(400, 200)
    window.show()
    sys.exit(app.exec_())
