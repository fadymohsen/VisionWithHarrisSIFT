import numpy as np
import cv2
import pyqtgraph as pg

class SIFTCornerDetection:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.SIFTDetector()


    def SIFTDetector(self):
        pass