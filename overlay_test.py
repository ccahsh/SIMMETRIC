import cv2
from PySide2 import QtWidgets, QtCore
import sksurgeryvtk.widgets.vtk_overlay_window as OW
import sksurgeryvtk.models.vtk_surface_model as SM
import sksurgeryvtk.models.vtk_surface_model_directory_loader as SMDL
from sksurgeryvtk.text import text_overlay

class OverlayApp():

    def __init__(self, video_source):
        self.vtk_overlay_window = OW.VTKOverlayWindow()
        self.video_source = cv2.VideoCapture(video_source)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

        update_frequency_ms = 30
        self.timer.start(update_frequency_ms)

        self.vtk_overlay_window.show()
        corner_annotation = text_overlay.VTKCornerAnnotation()
        corner_annotation.set_text(["1", "2", "3", "4"])
        self.vtk_overlay_window.add_vtk_actor(corner_annotation.text_actor, layer=2)

        large_text = text_overlay.VTKLargeTextCentreOfScreen("Central Text")
        large_text.set_colour(1.0, 0.0, 0.0)
        large_text.set_parent_window(self.vtk_overlay_window)
        self.vtk_overlay_window.add_vtk_actor(large_text.text_actor, layer=2)

        more_text = text_overlay.VTKText("More text", x=50, y=100)
        more_text.set_colour(0.0, 1.0, 0.0)
        self.vtk_overlay_window.add_vtk_actor(more_text.text_actor, layer=2)

        self.counter = 0

    def update(self):
        ret, img = self.video_source.read()
        # img = cv2.flip(img, 0)
        self.vtk_overlay_window.set_video_image(img)
        self.vtk_overlay_window._RenderWindow.Render()

        self.counter+=1

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

camera_source = "datasets\\JIGSAW\\Knot_Tying\\Knot_Tying\\video\\Knot_Tying_B001_capture1.avi"
overlay_app = OverlayApp(camera_source)

app.exec_()

