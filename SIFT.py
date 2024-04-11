import numpy as np
import cv2
import pyqtgraph as pg
import time

class SIFTCornerDetection:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget


    def SIFTDetector(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            self.ui.graphicsLayoutWidget_beforeSIFT.clear()
            original_img_item = pg.ImageItem(imageArray)
            original_view = self.ui.graphicsLayoutWidget_beforeSIFT.addViewBox()
            original_view.addItem(original_img_item)
            self.original_image = imageArray

            startTime = time.time()

            #Creating a Base Image to start by upsampling by doupling the image in size in both direction
            initialImage = cv2.resize(imageArray, (0, 0), fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
            #Inorder to start the image with sigma = 1.6 we blurr the upsambled image by sigma diff
            #assuming original image has blurr of 0.5 
            startSigma = 1.6
            sigma_diff = np.sqrt(max((startSigma ** 2) - ((2 * 0.5) ** 2), 0.01))
            baseImage = cv2.GaussianBlur(initialImage, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff) 
            

            #Calculating the number of possible octaves that will result of minimum image of 3 * 3
            baseImageShape = baseImage.shape
            numOfOctaves = int(round(np.log(min(baseImageShape)) / np.log(2) - 1))
            print(f"number of octaves = ", numOfOctaves)


            #Create a list of the amount of blur for each image in a particular layer
            numOfIntervals = 3
            numOfImagesPerOctave  = 6
            k = 2 ** (1. / numOfIntervals)
            gaussianKernels = np.zeros(numOfImagesPerOctave)  
            gaussianKernels[0] = startSigma #Start sigma = 1.6
            for image_index in range(1, numOfImagesPerOctave ):
                sigma_previous = (k ** (image_index - 1)) * 1.6
                sigma_total = k * sigma_previous
                gaussianKernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)


            #Finding the Gaussian Images for each octave to prepare for DOG
            gaussianImages = []

            for octave_index in range(numOfOctaves):
                gaussian_images_in_octave = []
                gaussian_images_in_octave.append(baseImage)  # first image in octave already has the correct blur

                for gaussianKernel in gaussianKernels[1:]:
                    image = cv2.GaussianBlur(baseImage, (0, 0), sigmaX=gaussianKernel, sigmaY=gaussianKernel)
                    gaussian_images_in_octave.append(image)

                gaussianImages.append(gaussian_images_in_octave)
                octave_base = gaussian_images_in_octave[-3]
                #downsampling the images in each octave for half
                image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2),
                                                  int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
                
            gaussianImages = np.array(gaussianImages)


            #Now generating the DOG Images
            DOGImages = []

            for gaussian_images_in_octave in gaussianImages:
                dog_images_in_octave = []
                #Subtracting adjacent images in the gaussian images
                for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                    dog_images_in_octave.append(np.subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
                DOGImages.append(dog_images_in_octave)
            DOGImages =  np.array(DOGImages)


            #Now we Find pixel positions of all scale-space extrema in all octaves
            #Some constants from openCV implementation found in papers
            contrastThreshold=0.04
            threshold = np.floor(0.5 * contrastThreshold / numOfIntervals * 255) 

            keypoints = []

            for octaveIndex, DOGImageInOctave in enumerate(DOGImages):
                for imageIndex, (firstImage, secondImage, thirdImage) in enumerate(zip(DOGImageInOctave,
                                                                                        DOGImageInOctave[1:],
                                                                                          DOGImageInOctave[2:])):
                    #We exclude a 5px border to avoid edge effects
                    imageBorderWidthExcluded = 5
                    for i in range(imageBorderWidthExcluded, firstImage.shape[0] - imageBorderWidthExcluded):
                        for j in range(imageBorderWidthExcluded, firstImage.shape[1] - imageBorderWidthExcluded):
                            #extracting 3x3 submatrices centered at pixel (i, j)
                            firstSubmatrix = firstImage[i-1:i+2, j-1:j+2]
                            secondSubmatrix = secondImage[i-1:i+2, j-1:j+2]
                            thirdSubmatrix = thirdImage[i-1:i+2, j-1:j+2]
                            #Finding if a centred pixel is local minima or maxima
                            centredPixel = firstSubmatrix[1, 1]
                            islocalMaxima = False  
                            islocalMinima = False
                            if abs(centredPixel)>threshold:
                                #Checking if it's local maxima
                                if centredPixel > 0:
                                    if all(centredPixel >= firstSubmatrix.flatten()) and \
                                          all(centredPixel >= thirdSubmatrix.flatten()) and \
                                    all(centredPixel >= secondSubmatrix[0,: ].flatten()) and\
                                          all(centredPixel >= secondSubmatrix[2,: ].flatten()) and \
                                    all(centredPixel >= secondSubmatrix[1,0].flatten()) and \
                                          all(centredPixel >= secondSubmatrix[1,2].flatten()):
                                        islocalMaxima = True
                                #Checking if it's local minima
                                elif centredPixel < 0:
                                    if all(centredPixel <= firstSubmatrix.flatten()) and \
                                         all(centredPixel <= thirdSubmatrix.flatten()) and \
                                    all(centredPixel <= secondSubmatrix[0,: ].flatten()) and\
                                          all(centredPixel <= secondSubmatrix[2,: ].flatten()) and \
                                    all(centredPixel <= secondSubmatrix[1,0].flatten()) and \
                                          all(centredPixel <= secondSubmatrix[1,2].flatten()):
                                        islocalMinima = True
                            if (islocalMaxima or islocalMinima):
                                pass

            

            endTime = time.time()
            totalTime = endTime - startTime
            self.ui.textEdit_computationTime.setText(str(totalTime))


    
    def displayFinalImage(self, image):
        self.ui.graphicsLayoutWidget_afterSIFT.clear()
        original_img_item = pg.ImageItem(image)
        original_view = self.ui.graphicsLayoutWidget_afterSIFT.addViewBox()
        original_view.addItem(original_img_item)
        