import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtMultimedia import QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5 import uic

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("GUI.ui", self)

        self.cameraPlaceholder = self.findChild(QWidget, "cameraPlaceholder")

        available_cameras = QCameraInfo.availableCameras()
        if not available_cameras:
            QMessageBox.warning(self, "Error", "No cameras found")
            sys.exit()

        # Selecting the first available camera
        self.camera = QCamera(available_cameras[0])

        self.viewfinder = QCameraViewfinder(self.cameraPlaceholder)
        self.camera.setViewfinder(self.viewfinder)

        layout = QVBoxLayout(self.cameraPlaceholder)
        layout.addWidget(self.viewfinder)
        self.cameraPlaceholder.setLayout(layout)

        self.camera.errorOccurred.connect(self.handle_camera_error)
        self.camera.start()

        self.show()

    def handle_camera_error(self, error, error_string):
        QMessageBox.warning(self, "Camera Error", f"Camera error: {error_string}")
        self.camera.stop()

def main():
    app = QApplication(sys.argv)
    window = GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
