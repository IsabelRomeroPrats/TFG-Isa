import sys
import os
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout,
    QLineEdit, QMessageBox, QGridLayout, QSlider
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.constants import sigma

from image_processing import warp_perspective_from_points, matrix_resized
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
        self.dragging_point = None

    def set_numpy_image(self, image, update_display=True, clear_points=True):
        self.image_np = image.copy()
        if clear_points:
            self.points.clear()
            self.shapes.clear()
            self.shape_emissivities.clear()
        if update_display:
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

        if len(self.image_np.shape) == 3:
            img_h, img_w = self.image_np.shape[:2]
        else:
            img_h, img_w = self.image_np.shape
        x_rel = (x_click - x_offset) / scaled_w
        y_rel = (y_click - y_offset) / scaled_h
        x_real = int(x_rel * img_w)
        y_real = int(y_rel * img_h)

        # Si estoy ajustando, comprobar si clico cerca de un punto existente
        if self.mode == 'adjust_corners' and self.points:
            for i, (px, py) in enumerate(self.points):
                if abs(px - x_real) < 15 and abs(py - y_real) < 15:
                    self.dragging_point = i
                    return

        # Si estoy en modo corners, añadir puntos normalmente
        if self.mode == 'corners':
            self.points.append((x_real, y_real))
            self.update_display()
            if len(self.points) == self.expected_points:
                if self.callback:
                    self.callback(self.points)

    def mouseMoveEvent(self, event):
        if self.dragging_point is not None and self.image_np is not None:
            pixmap = self.pixmap()
            label_size = self.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_w, scaled_h = scaled_pixmap.width(), scaled_pixmap.height()
            x_offset = (label_size.width() - scaled_w) // 2
            y_offset = (label_size.height() - scaled_h) // 2

            x_move = event.pos().x()
            y_move = event.pos().y()
            if len(self.image_np.shape) == 3:
                img_h, img_w = self.image_np.shape[:2]
            else:
                img_h, img_w = self.image_np.shape
            x_rel = (x_move - x_offset) / scaled_w
            y_rel = (y_move - y_offset) / scaled_h
            x_real = int(x_rel * img_w)
            y_real = int(y_rel * img_h)

            self.points[self.dragging_point] = (x_real, y_real)
            self.update_display()

    def mouseReleaseEvent(self, event):
        self.dragging_point = None

    def update_display(self):
        if self.image_np is None:
            return

        if len(self.image_np.shape) == 2:
            img_display = cv2.normalize(self.image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        else:
            img_display = self.image_np.copy()

        # Dibuja formas existentes
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

        # Dibuja líneas de conexión entre puntos de corners
        if len(self.points) >= 2:
            for i in range(len(self.points)):
                pt1 = self.points[i]
                pt2 = self.points[(i + 1) % len(self.points)]
                cv2.line(img_display, pt1, pt2, (255, 0, 0), 2)

        # Dibuja puntos
        for pt in self.points:
            cv2.circle(img_display, pt, 6, (0, 0, 255), -1)

        h, w = img_display.shape[:2]
        qimg = QImage(img_display.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio))



"--------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------"
"----------------------------          CÓDIGO DE LA APP          ----------------------------"
"--------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------"


class IRCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR Correction Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        #Almacenar imagen a color
        self.image_rgb = None  
        self.image_path = None
        self.image_data = None
        self.temperature = None
        self.emissivity = None
        self.tif_corners = []
        self.rgb_display_image = None  
        self.tif_display_image = None  
        self.setup_ui()

    def setup_ui(self):

        self.setWindowState(Qt.WindowMaximized)

        ### === LAYOUT PRINCIPAL ===
        main_layout = QHBoxLayout()

        # Dos columnas principales
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()

        ### === FILA SUPERIOR: RGB + TIF + SUPERPOSE ===
        images_row = QHBoxLayout()

        # === TITLES ROW ===
        titles_row = QHBoxLayout()

        label_rgb_title = QLabel("RGB")
        label_rgb_title.setAlignment(Qt.AlignCenter)
        label_rgb_title.setStyleSheet("font-weight: bold;")

        label_tif_title = QLabel("TIF")
        label_tif_title.setAlignment(Qt.AlignCenter)
        label_tif_title.setStyleSheet("font-weight: bold;")

        label_superpose_title = QLabel("Superpose")
        label_superpose_title.setAlignment(Qt.AlignCenter)
        label_superpose_title.setStyleSheet("font-weight: bold;")

        titles_row.addWidget(label_rgb_title)
        titles_row.addWidget(label_tif_title)
        titles_row.addWidget(label_superpose_title)

        # ---- RGB ----
        self.image_label_rgb = ClickableImageLabel()
        self.image_label_rgb.setMinimumSize(300, 300)
        rgb_block = QVBoxLayout()
        rgb_block.addWidget(self.image_label_rgb)



        rgb_buttons = QHBoxLayout()
        self.btn_insert_rgb = QPushButton("Insert RGB")
        self.btn_select_rgb = QPushButton("Adjust RGB")
        self.btn_apply_rgb = QPushButton("Apply RGB")
        self.btn_rotate_rgb = QPushButton("↻")
        rgb_buttons.addWidget(self.btn_insert_rgb)
        rgb_buttons.addWidget(self.btn_select_rgb)
        rgb_buttons.addWidget(self.btn_apply_rgb)
        rgb_buttons.addWidget(self.btn_rotate_rgb)
        rgb_block.addLayout(rgb_buttons)
        self.btn_insert_rgb.setFixedHeight(40)
        self.btn_select_rgb.setFixedHeight(40)
        self.btn_apply_rgb.setFixedHeight(40)
        self.btn_rotate_rgb.setFixedSize(30, 30)


        # ---- TIF ----
        self.image_label_tif = ClickableImageLabel()
        self.image_label_tif.setMinimumSize(300, 300)
        tif_block = QVBoxLayout()
    
        tif_block.addWidget(self.image_label_tif)

        tif_buttons = QHBoxLayout()
        self.btn_insert_tif = QPushButton("Insert TIF")
        self.btn_select_tif = QPushButton("Adjust TIF")
        self.btn_apply_tif = QPushButton("Apply TIF")
        self.btn_rotate_tif = QPushButton("↻")
        tif_buttons.addWidget(self.btn_insert_tif)
        tif_buttons.addWidget(self.btn_select_tif)
        tif_buttons.addWidget(self.btn_apply_tif)
        tif_buttons.addWidget(self.btn_rotate_tif)
        tif_block.addLayout(tif_buttons)
        # Para botones principales
        self.btn_insert_tif.setFixedHeight(40)
        self.btn_select_tif.setFixedHeight(40)
        self.btn_apply_tif.setFixedHeight(40)
        # Botón de rotar más pequeño
        self.btn_rotate_tif.setFixedSize(30, 30)

        # ---- SUPERPOSE ----
        self.superpose_label = QLabel()
        self.superpose_label.setMinimumSize(300, 300)
        superpose_block = QVBoxLayout()
        superpose_block.addWidget(self.superpose_label)


        # Sliders para transparencia
        self.slider_rgb = QSlider(Qt.Horizontal)
        self.slider_rgb.setRange(0, 100)
        self.slider_rgb.setValue(50)
        self.slider_tif = QSlider(Qt.Horizontal)
        self.slider_tif.setRange(0, 100)
        self.slider_tif.setValue(50)

        sliders = QVBoxLayout()
        sliders.addWidget(QLabel("RGB Transparency"))
        sliders.addWidget(self.slider_rgb)
        sliders.addWidget(QLabel("TIF Transparency"))
        sliders.addWidget(self.slider_tif)
        self.slider_rgb.setFixedWidth(300)
        self.slider_tif.setFixedWidth(300)

        superpose_block.addLayout(sliders)

        # Añadir bloques a la fila superior
        images_row.addLayout(rgb_block)
        images_row.addLayout(tif_block)
        images_row.addLayout(superpose_block)

        ### === PARÁMETROS Y DIBUJO ===
        # === TITLES arriba ===
        left_column.addLayout(titles_row)
        left_column.addLayout(images_row)

        # --- Título Parameters ---
        params_title = QLabel("Parameters")
        params_title.setAlignment(Qt.AlignCenter)
        params_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        left_column.addWidget(params_title)

        # --- Bloque de entradas ---
        params_layout = QHBoxLayout()
        self.temp_input = QLineEdit()
        self.temp_input.setPlaceholderText("Temperature")
        self.emiss_input = QLineEdit()
        self.emiss_input.setPlaceholderText("Emissivity")
        self.tau_input = QLineEdit()
        self.tau_input.setPlaceholderText("Tau")

        params_layout.addWidget(QLabel("Temperature:"))
        params_layout.addWidget(self.temp_input)
        params_layout.addWidget(QLabel("Emissivity:"))
        params_layout.addWidget(self.emiss_input)
        params_layout.addWidget(QLabel("Tau:"))
        params_layout.addWidget(self.tau_input)

        left_column.addLayout(params_layout)

        # --- Botones Draw ---
        draw_buttons = QHBoxLayout()
        self.start_button = QPushButton("Start Draw")
        self.finish_button = QPushButton("End Draw")
        draw_buttons.addWidget(self.start_button)
        draw_buttons.addWidget(self.finish_button)
        left_column.addLayout(draw_buttons)

        # --- Shapes y Emissivity Matrix ---
        shapes_and_matrix = QHBoxLayout()

        shapes_box = QVBoxLayout()
        shapes_box.addWidget(QLabel("Shapes:"))
        self.shapes_layout = QVBoxLayout()
        shapes_box.addLayout(self.shapes_layout)

        self.fig = Figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(400, 400)  # cuadrada

        shapes_and_matrix.addLayout(shapes_box, stretch=2)
        shapes_and_matrix.addWidget(self.canvas, stretch=3)

        left_column.addLayout(shapes_and_matrix)

        ### === COLUMNA DERECHA: CORRECCIÓN Y RESULTADOS ===
        self.btn_add_model = QPushButton("Add Correction Model")
        self.model_label = QLabel("Correction Model")
        self.model_label.setMinimumSize(200, 200)
        self.model_label.setAlignment(Qt.AlignCenter)

        self.btn_apply_corr = QPushButton("Apply Correction")
        self.result_fig1 = QLabel("Fig 1 Placeholder")
        self.result_fig1.setMinimumSize(200, 200)
        self.result_fig1.setAlignment(Qt.AlignCenter)

        self.result_fig2 = QLabel("Fig 2 Placeholder")
        self.result_fig2.setMinimumSize(200, 200)
        self.result_fig2.setAlignment(Qt.AlignCenter)

        right_column.addWidget(self.btn_add_model)
        right_column.addWidget(self.model_label)
        right_column.addWidget(self.btn_apply_corr)
        right_column.addWidget(self.result_fig1)
        right_column.addWidget(self.result_fig2)

        ### === MONTAR T===
        main_layout.addLayout(left_column, stretch=3)
        main_layout.addLayout(right_column, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        ### === CONECTAR SLIDERS ===
        self.slider_rgb.valueChanged.connect(self.update_superpose)
        self.slider_tif.valueChanged.connect(self.update_superpose)

        # === CONEXIONES PARA TODOS LOS BOTONES Y SLIDERS ===

        # Botones RGB
        self.btn_insert_rgb.clicked.connect(self.load_rgb_image)
        self.btn_select_rgb.clicked.connect(self.reselect_and_adjust_rgb)
        self.btn_apply_rgb.clicked.connect(self.apply_rgb_alignment)
        self.btn_rotate_rgb.clicked.connect(self.rotate_rgb_image)

        # Botones TIF
        self.btn_insert_tif.clicked.connect(self.load_image)
        self.btn_select_tif.clicked.connect(self.reselect_and_adjust_tif)
        self.btn_apply_tif.clicked.connect(self.apply_tif_alignment)
        self.btn_rotate_tif.clicked.connect(self.rotate_tif_image)

        # Sliders de transparencia
        self.slider_rgb.valueChanged.connect(self.update_superpose)
        self.slider_tif.valueChanged.connect(self.update_superpose)

        # Botones para shapes
        self.start_button.clicked.connect(self.start_shape)
        self.finish_button.clicked.connect(self.finish_shape)

        # Corrección
        self.btn_apply_corr.clicked.connect(self.apply_correction)
        # Si tienes un método para añadir el modelo de corrección:
        # self.btn_add_model.clicked.connect(self.load_correction_model)

###

### FUNCIONES

###

#"-------------------------------------RGB---------------------------------------------"

    def load_rgb_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Open RGB image', '', 'Image files (*.jpg *.png *.bmp)'
        )
        if not fname:
            return

        image = cv2.imread(fname)
        if image is None:
            QMessageBox.warning(self, "Error", "Could not load image.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_rgb_original = image_rgb.copy()
        self.image_rgb = image_rgb.copy()
        self.rgb_display_image = image_rgb.copy()  # ⚡️ Guardar SIEMPRE buffer display!
        self.image_label_rgb.set_numpy_image(self.rgb_display_image)
        self.update_emissivity_canvas()
        self.start_corner_selection()

    def start_corner_selection(self):
        self.image_label_rgb.points.clear()
        self.image_label_rgb.mode = 'corners'

        def on_points(points):
            self.rgb_corners = points.copy()   
            self.image_label_rgb.mode = 'adjust_corners'
            QMessageBox.information(self, "Adjust Mode", "Now you can drag RGB corners to adjust.")

        self.image_label_rgb.callback = on_points
        self.image_label_rgb.update_display()
        QMessageBox.information(self, "Select", "Click 4 corners on the RGB image.")

    def reselect_and_adjust_rgb(self):
        if self.image_rgb_original is None:
            QMessageBox.warning(self, "Error", "Load RGB first.")
            return

        self.image_rgb = self.image_rgb_original.copy()
        self.image_label_rgb.set_numpy_image(self.image_rgb, update_display=False, clear_points=False)

        if len(self.rgb_corners) != 4:
            QMessageBox.warning(self, "Error", "Corners must be defined first.")
            return

        self.image_label_rgb.points = self.rgb_corners.copy()
        self.image_label_rgb.mode = 'adjust_corners'
        self.image_label_rgb.update_display()
        QMessageBox.information(self, "Adjust Mode", "Drag RGB corners to adjust.")

    def apply_rgb_alignment(self):
        if len(self.image_label_rgb.points) != 4:
            QMessageBox.warning(self, "Error", "Define 4 corners first.")
            return

        src_pts = np.array(self.image_label_rgb.points, dtype='float32')
        dst_pts = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype='float32')

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aligned = cv2.warpPerspective(self.image_rgb_original, matrix, (400, 400))

        self.image_rgb = aligned
        self.rgb_display_image = aligned.copy()  # ⚡️ actualiza buffer display
        self.image_label_rgb.set_numpy_image(self.rgb_display_image)
        self.update_superpose()

        QMessageBox.information(self, "Alignment Done", "RGB image aligned with adjusted corners.")

    def select_corners_rgb(self):
        if self.image_rgb is None:
            QMessageBox.warning(self, "Error", "Load RGB image first.")
            return

        def on_4_points(points):
            from image_processing import warp_perspective_from_points
            aligned = warp_perspective_from_points(self.image_rgb, points, output_size=(400, 400))
            self.image_rgb = aligned
            self.image_label_rgb.set_numpy_image(aligned)
            QMessageBox.information(self, "RGB aligned", "RGB image has been aligned.")

        self.image_label_rgb.points.clear()
        self.image_label_rgb.expected_points = 4
        self.image_label_rgb.mode = 'corners'
        self.image_label_rgb.callback = on_4_points
        self.image_label_rgb.update_display()
        QMessageBox.information(self, "Select", "Click 4 corners on the RGB image.")

    def crop_rgb_image(self):
        if self.image_rgb is None:
            QMessageBox.warning(self, "Error", "Load RGB image first.")
            return

        # Mostrar ventana OpenCV para seleccionar ROI
        image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
        roi = cv2.selectROI("Select Region", image_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            QMessageBox.warning(self, "Warning", "No region selected.")
            return

        x, y, w, h = roi

        # Definir corners de la ROI como si fueran puntos de warp
        src_pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')

        dst_pts = np.array([
            [0, 0],
            [400, 0],
            [400, 400],
            [0, 400]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aligned = cv2.warpPerspective(self.image_rgb, matrix, (400, 400))

        self.image_rgb = aligned
        self.image_label_rgb.set_numpy_image(aligned)
        QMessageBox.information(self, "RGB Cropped", "RGB image has been cropped and aligned.")


#"-------------------------------------TIF---------------------------------------------"

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open TIF image', '', 'Image files (*.tif *.jpg *.png)')
        if fname:
            self.image_data = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if self.image_data is not None:
                self.image_tif_original = self.image_data.copy()
                # Normaliza para display
                aligned_norm = cv2.normalize(self.image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                self.tif_display_image = aligned_norm.copy()  # ⚡️ buffer display
                self.image_label_tif.set_numpy_image(self.tif_display_image)
                self.start_tif_corner_selection()

    def start_tif_corner_selection(self):
        self.image_label_tif.points.clear()
        self.image_label_tif.mode = 'corners'

        def on_points(points):
            self.image_label_tif.mode = 'adjust_corners'
            QMessageBox.information(self, "Adjust Mode", "Now you can drag TIF corners to adjust.")

        self.image_label_tif.callback = on_points
        self.image_label_tif.update_display()
        QMessageBox.information(self, "Select", "Click 4 corners on the TIF image.")

    def reselect_and_adjust_tif(self):
        if self.image_tif_original is None:
            QMessageBox.warning(self, "Error", "Load TIF first.")
            return

        self.image_data = self.image_tif_original.copy()
        self.image_label_tif.set_numpy_image(self.image_data, update_display=False, clear_points=False)

        if len(self.tif_corners) != 4:
            QMessageBox.warning(self, "Error", "Corners must be defined first.")
            return

        self.image_label_tif.points = self.tif_corners.copy()
        self.image_label_tif.mode = 'adjust_corners'
        self.image_label_tif.update_display()
        QMessageBox.information(self, "Adjust Mode", "Drag TIF corners to adjust.")

    def apply_tif_alignment(self):
        if len(self.image_label_tif.points) != 4:
            QMessageBox.warning(self, "Error", "Define 4 corners first.")
            return

        src_pts = np.array(self.image_label_tif.points, dtype='float32')
        dst_pts = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype='float32')

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aligned = cv2.warpPerspective(self.image_tif_original, matrix, (400, 400))

        self.tif_corners = self.image_label_tif.points.copy()
        self.image_data = aligned  # radiométrico original
        aligned_norm = cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.tif_display_image = aligned_norm.copy()  # ⚡️ buffer display actualizado
        self.image_label_tif.set_numpy_image(self.tif_display_image)
        self.update_superpose()

        QMessageBox.information(self, "Done", "TIF image aligned with adjusted corners.")

    def start_tif_corner_selection(self):
        self.image_label_tif.points.clear()
        self.image_label_tif.mode = 'corners'

        def on_points(points):
            self.tif_corners = points.copy()  # GUARDA!
            self.image_label_tif.mode = 'adjust_corners'
            QMessageBox.information(self, "Adjust Mode", "Now you can drag TIF corners to adjust.")

        self.image_label_tif.callback = on_points
        self.image_label_tif.update_display()
        QMessageBox.information(self, "Select", "Click 4 corners on the TIF image.")


##

### Botones de rotación

###

    def rotate_tif_image(self):
        if self.image_data is not None:
            self.image_data = cv2.rotate(self.image_data, cv2.ROTATE_90_CLOCKWISE)
            aligned_norm = cv2.normalize(self.image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.tif_display_image = aligned_norm.copy()  # ⚡️ actualiza display buffer!
            self.image_label_tif.set_numpy_image(self.tif_display_image)
            self.update_superpose()

    def rotate_rgb_image(self):
        if self.image_rgb is not None:
            self.image_rgb = cv2.rotate(self.image_rgb, cv2.ROTATE_90_CLOCKWISE)
            self.rgb_display_image = self.image_rgb.copy()  
            self.image_label_rgb.set_numpy_image(self.rgb_display_image)
            self.update_superpose()


###☺

### SUPERPONER

###

    def update_superpose(self):
        if self.rgb_display_image is None or self.tif_display_image is None:
            return

        rgb = self.rgb_display_image.copy()
        tif = self.tif_display_image.copy()

        if rgb.ndim == 2:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if tif.ndim == 2:
            tif_bgr = cv2.cvtColor(tif, cv2.COLOR_GRAY2BGR)
        else:
            tif_bgr = cv2.cvtColor(tif, cv2.COLOR_RGB2BGR)

        if tif_bgr.shape[:2] != rgb_bgr.shape[:2]:
            tif_bgr = cv2.resize(tif_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))

        alpha_rgb = self.slider_rgb.value() / 100.0
        alpha_tif = self.slider_tif.value() / 100.0

        combined_bgr = cv2.addWeighted(rgb_bgr, alpha_rgb, tif_bgr, alpha_tif, 0)
        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

        h, w, ch = combined_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(combined_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.superpose_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.superpose_label.size(), Qt.KeepAspectRatio))


###

## Funciones de la matriz de emisividad

###

    def update_emissivity_canvas(self):
        matrix = self.build_emissivity_matrix(silent=True) 
        if matrix is None:
            return

        matrix_cont, matrix_discrete = matrix_resized(matrix, continuous_shape=(500, 500), m=20, n=20)

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(matrix_discrete, cmap='hot', interpolation='nearest')
        self.fig.colorbar(im, ax=self.ax, label='Emissivity')
        self.ax.set_title("Emissivity Matrix")
        self.canvas.draw()

    def activate_shape_selection(self):
        self.image_label_rgb.points.clear()
        self.image_label_rgb.expected_points = 4  # por ahora, polígono de 4 lados
        self.image_label_rgb.mode = 'polygon'

        def on_shape_added(shape_data, emissivity):
            index = len(self.image_label_rgb.shapes)
            QMessageBox.information(self, "Shape Added", f"Shape {index} added with emissivity {emissivity:.2f}")
            # Aquí podrías actualizar una lista visual de formas si quieres

        self.image_label_rgb.callback = on_shape_added
        self.image_label_rgb.update_display()
        QMessageBox.information(self, "Select", "Click polygon points on the image.")

    def start_shape(self):
        self.image_label_rgb.points.clear()
        self.image_label_rgb.mode = 'polygon'
        self.image_label_rgb.expected_points = 9999  # número indefinido
        self.image_label_rgb.update_display()
        QMessageBox.information(self, "Shape Mode", "Click to add points. Then click 'Finish Shape'.")

    def finish_shape(self):
        points = self.image_label_rgb.points.copy()

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
        value_str, ok = QInputDialog.getText(self, "Emissivity", "Enter emissivity (0-1):")
        if not ok:
            return

        try:
            emissivity = float(value_str.replace(",", "."))
            if not (0 < emissivity <= 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Enter a valid emissivity between 0 and 1.")
            return

        if shape_type == 'circle':
            pts = np.array(points, dtype=np.float32)
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            self.image_label_rgb.shapes.append(('circle', ((int(cx), int(cy)), int(radius))))
        else:
            self.image_label_rgb.shapes.append(('polygon', points))

        self.image_label_rgb.shape_emissivities.append(emissivity)
        self.image_label_rgb.points.clear()
        self.image_label_rgb.update_display()
        shape_type_label = shape_type.capitalize()
        self.add_shape_entry(len(self.image_label_rgb.shapes) - 1, shape_type_label, emissivity)
        self.update_emissivity_canvas()

    def add_shape_entry(self, shape_index, shape_type, current_emissivity):
        layout = QHBoxLayout()
        label = QLabel(f"Shape {shape_index + 1} ({shape_type})")
        label.setFixedWidth(120)

        emiss_input = QLineEdit()
        emiss_input.setText(f"{current_emissivity:.3f}")
        emiss_input.setFixedWidth(60)

        # Función para actualizar emisividad y refrescar gráfica
        def update_emissivity():
            try:
                value = float(emiss_input.text().replace(",", "."))
                if not (0 < value <= 1):
                    raise ValueError
                self.image_label_rgb.shape_emissivities[shape_index] = value
                self.image_label_rgb.update_display()
                self.update_emissivity_canvas()
            except ValueError:
                QMessageBox.warning(self, "Invalid", "Enter a valid emissivity between 0 and 1.")

        emiss_input.editingFinished.connect(update_emissivity)

        # Botón pequeño para borrar
        delete_btn = QPushButton("🗑️")
        delete_btn.setFixedWidth(30)

        def delete_shape():
            if shape_index < len(self.image_label_rgb.shapes):
                self.image_label_rgb.shapes.pop(shape_index)
                self.image_label_rgb.shape_emissivities.pop(shape_index)
                self.image_label_rgb.update_display()
                self.update_emissivity_canvas()
                self.refresh_shape_entries()

        delete_btn.clicked.connect(delete_shape)

        layout.addWidget(label)
        layout.addWidget(QLabel("Emissivity:"))
        layout.addWidget(emiss_input)
        layout.addWidget(delete_btn)
        self.shapes_layout.addLayout(layout)

    def refresh_shape_entries(self):
        def _refresh():
            while self.shapes_layout.count():
                child = self.shapes_layout.takeAt(0)
                if child.layout():
                    while child.layout().count():
                        item = child.layout().takeAt(0)
                        if item.widget():
                            item.widget().setParent(None)
                elif child.widget():
                    child.widget().setParent(None)

            for idx, (shape, emiss) in enumerate(zip(self.image_label_rgb.shapes, self.image_label_rgb.shape_emissivities)):
                tipo = shape[0].capitalize() if shape[0] != 'triangle' else 'Triangle'
                self.add_shape_entry(idx, tipo, emiss)

        QTimer.singleShot(0, _refresh)

    def build_emissivity_matrix(self, m=20, n=20, silent=False):
        if self.image_data is None:
            if not silent:
                QMessageBox.warning(self, "Error", "Load an image first.")
            return None

        try:
            base_emiss = float(self.emiss_input.text())
            if not (0 < base_emiss <= 1):
                raise ValueError
        except ValueError:
            if not silent:
                QMessageBox.warning(self, "Error", "Enter a valid base emissivity.")
            return None
        
        height, width = self.image_rgb.shape[:2]
        matrix = np.full((m, n), base_emiss)

        # Celdas de la rejilla
        cell_h = height / m
        cell_w = width / n

        for shape, emiss in zip(self.image_label_rgb.shapes, self.image_label_rgb.shape_emissivities):
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
        matrix = self.build_emissivity_matrix(silent=False)  
        if matrix is None:
            return

        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Emissivity')
        plt.title('Emissivity Matrix')
        plt.tight_layout()

###

## Funciones Reflectancia

###

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


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = IRCorrectionApp()
    window.show()
    sys.exit(app.exec_())
