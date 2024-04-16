import numpy as np
import cv2
import pyqtgraph as pg
import time
from numpy.linalg import det, lstsq, norm




class SIFTCornerDetection:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget

    
    def uploadImageSIFT(self):
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


    def SIFTDetector(self):
            startTime = time.time()
            # Creating a Base Image to start by over sampling by doubling the image in size in both direction
            initialImage = cv2.resize(self.original_image, (0, 0), fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
            # In order to start the image with sigma = 1.6 we blur the over sampling image by sigma diff
            # Assuming original image has blur of 0.5 
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
            for octaveIndex in range(numOfOctaves):
                gaussian_images_in_octave = []
                gaussian_images_in_octave.append(baseImage)  # first image in octave already has the correct blur
                for gaussianKernel in gaussianKernels[1:]:
                    image = cv2.GaussianBlur(baseImage, (0, 0), sigmaX=gaussianKernel, sigmaY=gaussianKernel)
                    gaussian_images_in_octave.append(image)
                gaussianImages.append(gaussian_images_in_octave)
                octave_base = gaussian_images_in_octave[-3]
                # Down Sampling the images in each octave for half
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
            keyPoints = []

            for octaveIndex, DOGImageInOctave in enumerate(DOGImages):
                for imageIndex, (firstImage, secondImage, thirdImage) in enumerate(zip(DOGImageInOctave,
                                                                                        DOGImageInOctave[1:],
                                                                                          DOGImageInOctave[2:])):
                    #We exclude a 5px border to avoid edge effects
                    imageBorderWidthExcluded = 5
                    for i in range(imageBorderWidthExcluded, firstImage.shape[0] - imageBorderWidthExcluded):
                        for j in range(imageBorderWidthExcluded, firstImage.shape[1] - imageBorderWidthExcluded):
                            #extracting 3x3 subMatrices centered at pixel (i, j)
                            firstSubMatrix = firstImage[i-1:i+2, j-1:j+2]
                            secondSubMatrix = secondImage[i-1:i+2, j-1:j+2]
                            thirdSubMatrix = thirdImage[i-1:i+2, j-1:j+2]
                            if (self.isPixelMinimaOrMaxima(firstSubMatrix, secondSubMatrix, thirdSubMatrix, threshold)):
                                localization_result = self.localizeExtremum(i, j, imageIndex + 1, octaveIndex, numOfIntervals, DOGImageInOctave, startSigma, contrastThreshold, imageBorderWidthExcluded)
                                if localization_result is not None:
                                    keyPoint, localized_image_index = localization_result
                                  

            endTime = time.time()
            totalTime = endTime - startTime
            self.ui.textEdit_computationTime.setText(str(totalTime))


    def isPixelMinimaOrMaxima(self, firstSubMatrix, secondSubMatrix, thirdSubMatrix, threshold):
        #Finding if a centred pixel is local minima or maxima
        centredPixel = secondSubMatrix[1, 1]
        if abs(centredPixel)>threshold:
            #Checking if it's local maxima
            if centredPixel > 0:
                if all(centredPixel >= firstSubMatrix.flatten()) and \
                    all(centredPixel >= thirdSubMatrix.flatten()) and \
                all(centredPixel >= secondSubMatrix[0,: ].flatten()) and\
                    all(centredPixel >= secondSubMatrix[2,: ].flatten()) and \
                all(centredPixel >= secondSubMatrix[1,0].flatten()) and \
                    all(centredPixel >= secondSubMatrix[1,2].flatten()):
                    return True
            #Checking if it's local minima
            elif centredPixel < 0:
                if all(centredPixel <= firstSubMatrix.flatten()) and \
                    all(centredPixel <= thirdSubMatrix.flatten()) and \
                all(centredPixel <= secondSubMatrix[0,: ].flatten()) and\
                    all(centredPixel <= secondSubMatrix[2,: ].flatten()) and \
                all(centredPixel <= secondSubMatrix[1,0].flatten()) and \
                    all(centredPixel <= secondSubMatrix[1,2].flatten()):
                    return True
        return False

    def localizeExtremum(self, i, j, image_index, octaveIndex, numOfIntervals, DOGImageInOctave, sigma, contrastThreshold, imageBorderWidthExcluded, eigenvalueRatio=10, num_attempts_until_convergence=5):
        #Iteratively refine pixel positions of scale-space extrema 
        extremum_is_outside_image = False
        imageShape = DOGImageInOctave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            firstImage, secondImage, thirdImage = DOGImageInOctave[image_index-1:image_index+2]
            pixel_cube = np.stack([firstImage[i-1:i+2, j-1:j+2],
                                secondImage[i-1:i+2, j-1:j+2],
                                thirdImage[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.computeGradientAtCenterPixel(pixel_cube)
            hessian = self.computeHessianAtCenterPixel(pixel_cube)
            extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < imageBorderWidthExcluded or i >= imageShape[0] - imageBorderWidthExcluded or j < imageBorderWidthExcluded or j >= imageShape[1] - imageBorderWidthExcluded or image_index < 1 or image_index > numOfIntervals:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * numOfIntervals >= contrastThreshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalueRatio * (xy_hessian_trace ** 2) < ((eigenvalueRatio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV keyPoint object
                keyPoint = cv2.keyPoint()
                keyPoint.pt = ((j + extremum_update[0]) * (2 ** octaveIndex), (i + extremum_update[1]) * (2 ** octaveIndex))
                keyPoint.octave = octaveIndex + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keyPoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(numOfIntervals))) * (2 ** (octaveIndex + 1))  # octaveIndex + 1 because the input image was doubled
                keyPoint.response = abs(functionValueAtUpdatedExtremum)
                return keyPoint, image_index
        return None

    def computeGradientAtCenterPixel(self, pixelArray):
        #Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula , where h is the step size
        # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
        dx = 0.5 * (pixelArray[1, 1, 2] - pixelArray[1, 1, 0])
        dy = 0.5 * (pixelArray[1, 2, 1] - pixelArray[1, 0, 1])
        ds = 0.5 * (pixelArray[2, 1, 1] - pixelArray[0, 1, 1])
        return np.array([dx, dy, ds])

    def computeHessianAtCenterPixel(self, pixelArray):
        center_pixel_value = pixelArray[1, 1, 1]
        dxx = pixelArray[1, 1, 2] - 2 * center_pixel_value + pixelArray[1, 1, 0]
        dyy = pixelArray[1, 2, 1] - 2 * center_pixel_value + pixelArray[1, 0, 1]
        dss = pixelArray[2, 1, 1] - 2 * center_pixel_value + pixelArray[0, 1, 1]
        dxy = 0.25 * (pixelArray[1, 2, 2] - pixelArray[1, 2, 0] - pixelArray[1, 0, 2] + pixelArray[1, 0, 0])
        dxs = 0.25 * (pixelArray[2, 1, 2] - pixelArray[2, 1, 0] - pixelArray[0, 1, 2] + pixelArray[0, 1, 0])
        dys = 0.25 * (pixelArray[2, 2, 1] - pixelArray[2, 0, 1] - pixelArray[0, 2, 1] + pixelArray[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                            [dxy, dyy, dys],
                            [dxs, dys, dss]])
    
    def displayFinalImage(self, image):
        self.ui.graphicsLayoutWidget_afterSIFT.clear()
        original_img_item = pg.ImageItem(image)
        original_view = self.ui.graphicsLayoutWidget_afterSIFT.addViewBox()
        original_view.addItem(original_img_item)
        


    def computeKeypointOrientations(self, keypoints, gaussianImage):
        # Constants for orientation computation
        radius = 3 * self.sigma
        num_bins = 36
        hist = np.zeros(num_bins)

        for keypoint in keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            weight_window = gaussianImage[y - radius:y + radius + 1, x - radius:x + radius + 1]
            exp_scale = -1 / (2.0 * (0.5 * (2 * radius) * (0.5 * (2 * radius))))

            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (0 <= y + i < gaussianImage.shape[0]) and (0 <= x + j < gaussianImage.shape[1]):
                        dx = gaussianImage[y + i, x + j + 1] - gaussianImage[y + i, x + j - 1]
                        dy = gaussianImage[y + i + 1, x + j] - gaussianImage[y + i - 1, x + j]
                        magnitude = np.sqrt(dx * dx + dy * dy)
                        orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(exp_scale * (i * i + j * j)) * magnitude
                        hist[int(orientation // (360 // num_bins))] += weight

            dominant_orientation = np.argmax(hist) * (360.0 / num_bins)
            keypoint.angle = dominant_orientation  # Adding orientation to the keypoint
