import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import time

class ImageMatcher(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Matcher")
        self.setGeometry(100, 100, 1000, 600)

        self.label_image1 = QLabel(self)
        self.label_image2 = QLabel(self)
        self.label_result = QLabel(self)

        self.btn_load_images = QPushButton("Load Images", self)
        self.btn_load_images.clicked.connect(self.loadImages)

        layout = QHBoxLayout()
        layout.addWidget(self.label_image1)
        layout.addWidget(self.label_image2)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_load_images)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.label_result)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def loadImages(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg)")
        if len(file_names) == 2:
            image1 = cv2.imread(file_names[0], cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(file_names[1], cv2.IMREAD_GRAYSCALE)

            if image1 is not None and image2 is not None:
                if image1.shape[0] * image1.shape[1] > image2.shape[0] * image2.shape[1]:
                    large = image1
                    small = image2
                else:
                    large = image2
                    small = image1

                small_image_height, small_image_width = small.shape[:2]
                large_image_height, large_image_width = large.shape[:2]

                ssd_scores = []
                ncc_scores = []
                for y in range(large_image_height - small_image_height + 1):
                    for x in range(large_image_width - small_image_width + 1):
                        window = large[y:y + small_image_height, x:x + small_image_width]
                        ssd_score = self.calculateSSD(window, small)
                        ncc_score = self.calculateNCC(window, small)
                        ssd_scores.append((ssd_score, (x, y)))
                        ncc_scores.append((ncc_score, (x, y)))

                if ssd_scores:
                    best_ssd_score, best_ssd_position = min(ssd_scores, key=lambda x: x[0])
                else:
                    best_ssd_score, best_ssd_position = float('inf'), None

                if ncc_scores:
                    best_ncc_score, best_ncc_position = max(ncc_scores, key=lambda x: x[0])
                else:
                    best_ncc_score, best_ncc_position = float('-inf'), None

                self.displayImage(large, self.label_image1)
                self.displayImage(small, self.label_image2)

                self.label_result.setText("Best SSD Score: {:.2f}, Position: {}\nBest NCC Score: {:.2f}, Position: {}".format(best_ssd_score, best_ssd_position, best_ncc_score, best_ncc_position))
            else:
                print("Error loading images.")



    def calculateSSD(self, image1, image2):
        return np.sum((image1 - image2) ** 2)

    def calculateNCC(self, image1, image2):
        mean1 = np.mean(image1)
        mean2 = np.mean(image2)
        ncc_score = np.sum((image1 - mean1) * (image2 - mean2)) / (np.sqrt(np.sum((image1 - mean1) ** 2)) * np.sqrt(np.sum((image2 - mean2) ** 2)))
        return ncc_score

    def displayImage(self, image, label_widget):
        h, w = image.shape
        qimage = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap)
        label_widget.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageMatcher()
    ex.show()
    sys.exit(app.exec_())
