import numpy as np
import cv2
import pyqtgraph as pg
import time
from numpy.linalg import det, lstsq, norm
from typing import List, Tuple, Optional



class SIFTCornerDetection:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.sigma = 1.6

    
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



    def compute_keypoints_and_descriptors(self, sigma: float = 1.6, num_intervals: int = 3,
                                      assumed_blur: float = 0.5, image_border_width: int = 5) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Compute SIFT keypoints and descriptors for an input image.

        Parameters:
            image (np.ndarray): The input image.
            sigma (float): The sigma value used in GaussianBlur.
            num_intervals (int): Number of intervals per octave.
            assumed_blur (float): Initial image blur.
            image_border_width (int): Border width where keypoints are not detected.

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: A tuple containing keypoints and descriptors.
        """
        startTime = time.time()
        image = self.original_image.astype('float32')
        base_image = self.generate_base_image(sigma, assumed_blur)
        num_octaves = self.compute_number_of_octaves(base_image.shape)
        gaussian_kernels = self.generate_gaussian_kernels(sigma, num_intervals)
        gaussian_images = self.generate_gaussian_images(base_image, num_octaves, gaussian_kernels)
        dog_images = self.generate_dog_images(gaussian_images)
        keypoints = self.find_scale_space_extrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        keypoints = self.remove_duplicate_keypoints(keypoints)
        keypoints = self.convert_keypoints_to_input_image_size(keypoints)
        descriptors = self.generate_descriptors(keypoints, gaussian_images)
        endTime = time.time()
        totalTime = endTime - startTime
        self.ui.label_SIFTcomputationTime.setText(str(totalTime))
        
        # Draw keypoints on the image
        keypoint_image = cv2.drawKeypoints(base_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoint_image_rgb = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)
        # Display the processed image with keypoints
        self.displayFinalImage( self.keypoint_image_rgb, keypoints)
        return keypoints, descriptors




    def generate_base_image(image: np.ndarray, sigma: float, assumed_blur: float) -> np.ndarray:
        """
        Generate base image from input image by upsampling by 2 in both directions and blurring.

        Parameters:
            image (np.ndarray): The original input image.
            sigma (float): The sigma value for GaussianBlur.
            assumed_blur (float): The initial blur of the image.

        Returns:
            np.ndarray: The base image.
        """
        image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    def compute_number_of_octaves(image_shape: Tuple[int, int]) -> int:
        """
        Compute the number of octaves in the image pyramid based on the image shape.

        Parameters:
            image_shape (Tuple[int, int]): The dimensions of the base image.

        Returns:
            int: The number of octaves.
        """
        return int(round(np.log(min(image_shape)) / np.log(2) - 1))

    def generate_gaussian_kernels(sigma: float, num_intervals: int) -> np.ndarray:
        """
        Generate a list of gaussian kernels for image blurring across octaves.

        Parameters:
            sigma (float): The sigma value for the Gaussian kernel.
            num_intervals (int): Number of intervals per octave.

        Returns:
            np.ndarray: Array of Gaussian kernels.
        """
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = sigma
        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

    def generate_gaussian_images(image: np.ndarray, num_octaves: int, gaussian_kernels: np.ndarray) -> np.ndarray:
        """
        Generate a scale-space pyramid of Gaussian images.

        Parameters:
            image (np.ndarray): The base image.
            num_octaves (int): Number of octaves in the scale-space pyramid.
            gaussian_kernels (np.ndarray): Array of Gaussian kernels for blurring.

        Returns:
            np.ndarray: A 3D array containing Gaussian images for each octave.
        """
        gaussian_images = []
        for octave_index in range(num_octaves):
            gaussian_images_in_octave = [image]
            for gaussian_kernel in gaussian_kernels[1:]:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        return np.array(gaussian_images, dtype=object)

    def generate_dog_images(gaussian_images: np.ndarray) -> np.ndarray:
        """
        Generate Difference-of-Gaussians (DoG) images from Gaussian images.

        Parameters:
            gaussian_images (np.ndarray): A 3D array containing Gaussian images for each octave.

        Returns:
            np.ndarray: A 3D array containing DoG images for each octave.
        """
        dog_images = []
        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = [cv2.subtract(second_image, first_image) for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:])]
            dog_images.append(dog_images_in_octave)
        return np.array(dog_images, dtype=object)

    def find_scale_space_extrema(self, gaussian_images: np.ndarray, dog_images: np.ndarray, num_intervals: int, sigma: float, image_border_width: int,
                                contrast_threshold: float = 0.04) -> List[cv2.KeyPoint]:
        """
        Find pixel positions of all scale-space extrema in the image pyramid.

        Parameters:
            gaussian_images (np.ndarray): A 3D array of Gaussian images.
            dog_images (np.ndarray): A 3D array of Difference-of-Gaussians images.
            num_intervals (int): Number of intervals per octave.
            sigma (float): The sigma value used in Gaussian blur.
            image_border_width (int): Width of the border where keypoints are not detected.
            contrast_threshold (float): Threshold for contrast filtering.

        Returns:
            List[cv2.KeyPoint]: List of detected keypoints.
        """
        threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
        keypoints = []
        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                for i in range(image_border_width, first_image.shape[0] - image_border_width):
                    for j in range(image_border_width, first_image.shape[1] - image_border_width):
                        if self.is_pixel_an_extremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                            localization_result = self.localize_extremum_via_quadratic_fit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                            if localization_result:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = self.compute_keypoints_with_orientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                                keypoints.extend(keypoints_with_orientations)
        return keypoints

    def is_pixel_an_extremum(first_subimage: np.ndarray, second_subimage: np.ndarray, third_subimage: np.ndarray, threshold: float) -> bool:
        """
        Check if the center pixel of the 3x3x3 input array is a scale-space extremum.

        Parameters:
            first_subimage (np.ndarray): First 3x3 subimage.
            second_subimage (np.ndarray): Second 3x3 subimage (center).
            third_subimage (np.ndarray): Third 3x3 subimage.
            threshold (float): Threshold for considering a pixel as extremum.

        Returns:
            bool: True if center pixel is an extremum, False otherwise.
        """
        center_pixel_value = second_subimage[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_subimage) and \
                    np.all(center_pixel_value >= third_subimage) and \
                    np.all(center_pixel_value >= second_subimage[0, :]) and \
                    np.all(center_pixel_value >= second_subimage[2, :]) and \
                    center_pixel_value >= second_subimage[1, 0] and \
                    center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_subimage) and \
                    np.all(center_pixel_value <= third_subimage) and \
                    np.all(center_pixel_value <= second_subimage[0, :]) and \
                    np.all(center_pixel_value <= second_subimage[2, :]) and \
                    center_pixel_value <= second_subimage[1, 0] and \
                    center_pixel_value <= second_subimage[1, 2]
        return False

    def localize_extremum_via_quadratic_fit(self, i: int, j: int, image_index: int, octave_index: int, num_intervals: int, dog_images_in_octave: np.ndarray, sigma: float, contrast_threshold: float, image_border_width: int,
                                            eigenvalue_ratio: float = 10, num_attempts_until_convergence: int = 5) -> Optional[Tuple[cv2.KeyPoint, int]]:
        """
        Refine pixel position of scale-space extrema via quadratic fit around each extremum's neighbors.

        Parameters:
            i (int): Row index of the extremum.
            j (int): Column index of the extremum.
            image_index (int): Index of the image within the octave.
            octave_index (int): Index of the octave.
            num_intervals (int): Number of intervals per octave.
            dog_images_in_octave (np.ndarray): Images within the current octave.
            sigma (float): Sigma used in the Gaussian blur.
            contrast_threshold (float): Threshold for contrast filtering.
            image_border_width (int): Border width to avoid edge effects.
            eigenvalue_ratio (float): Ratio for eliminating edge responses.
            num_attempts_until_convergence (int): Maximum number of iterations for convergence.

        Returns:
            Optional[Tuple[cv2.KeyPoint, int]]: A tuple of the localized keypoint and its image index, or None if the keypoint is not valid.
        """
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
            pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.compute_gradient_at_center_pixel(pixel_cube)
            hessian = self.compute_hessian_at_center_pixel(pixel_cube)
            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if np.all(np.abs(extremum_update) < 0.5):
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image or attempt_index == num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None

    def compute_gradient_at_center_pixel(pixel_array: np.ndarray) -> np.ndarray:
        """
        Approximate gradient at center pixel [1, 1, 1] of a 3x3x3 array using central difference formula.

        Parameters:
            pixel_array (np.ndarray): A 3x3x3 array around the center pixel.

        Returns:
            np.ndarray: The gradient at the center pixel.
        """
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def compute_hessian_at_center_pixel(pixel_array: np.ndarray) -> np.ndarray:
        """
        Approximate Hessian at center pixel [1, 1, 1] of a 3x3x3 array using central difference formula.

        Parameters:
            pixel_array (np.ndarray): A 3x3x3 array around the center pixel.

        Returns:
            np.ndarray: The Hessian matrix at the center pixel.
        """
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

    def compute_keypoints_with_orientations(self, keypoint: cv2.KeyPoint, octave_index: int, gaussian_image: np.ndarray, radius_factor: int = 3, num_bins: int = 36, peak_ratio: float = 0.8, scale_factor: float = 1.5) -> List[cv2.KeyPoint]:
        """
        Compute orientations for each keypoint based on image gradients.

        Parameters:
            keypoint (cv2.KeyPoint): The keypoint for which orientations are computed.
            octave_index (int): The octave index of the keypoint.
            gaussian_image (np.ndarray): The Gaussian image corresponding to the keypoint's octave.
            radius_factor (int): The factor to determine the radius of the region considered around the keypoint.
            num_bins (int): Number of bins for the histogram of orientations.
            peak_ratio (float): The ratio to determine significant peaks in the orientation histogram.
            scale_factor (float): Scale factor to calculate effective radius based on keypoint size.

        Returns:
            List[cv2.KeyPoint]: A list of keypoints with computed orientations.
        """
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape
        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if 0 < region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if 0 < region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[(n - 1) % num_bins] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[(n - 2) % num_bins] + raw_histogram[(n + 2) % num_bins]) / 16.

        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < self.float_tolerance:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    def compare_keypoints(keypoint1: cv2.KeyPoint, keypoint2: cv2.KeyPoint) -> int:
        """
        Comparator function for keypoints to sort and remove duplicates.

        Parameters:
            keypoint1 (cv2.KeyPoint): First keypoint.
            keypoint2 (cv2.KeyPoint): Second keypoint.

        Returns:
            int: Comparison result for sorting.
        """
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def remove_duplicate_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        Sort keypoints and remove duplicates to ensure unique keypoints.

        Parameters:
            keypoints (List[cv2.KeyPoint]): List of keypoints to filter.

        Returns:
            List[cv2.KeyPoint]: List of unique keypoints.
        """
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=self.cmp_to_key(self.compare_keypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints

    def convert_keypoints_to_input_image_size(keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        Adjust keypoint coordinates and size back to the dimensions of the input image.

        Parameters:
            keypoints (List[cv2.KeyPoint]): List of keypoints detected in the processed image.

        Returns:
            List[cv2.KeyPoint]: List of keypoints adjusted to the input image size.
        """
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def unpack_octave(keypoint: cv2.KeyPoint) -> Tuple[int, int, float]:
        """
        Decompose the octave information from a keypoint.

        Parameters:
            keypoint (cv2.KeyPoint): The keypoint to unpack.

        Returns:
            Tuple[int, int, float]: The octave, layer, and scale from the keypoint.
        """
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale

    def generate_descriptors(self, keypoints: List[cv2.KeyPoint], gaussian_images: np.ndarray, window_width: int = 4, num_bins: int = 8, scale_multiplier: float = 3, descriptor_max_value: float = 0.2) -> np.ndarray:
        """
        Generate descriptors for each keypoint based on their orientation histograms.

        Parameters:
            keypoints (List[cv2.KeyPoint]): List of keypoints.
            gaussian_images (np.ndarray): The Gaussian images corresponding to each octave and layer.
            window_width (int): The width of the window used to consider the neighborhood of each keypoint.
            num_bins (int): Number of bins for the orientation histogram.
            scale_multiplier (float): Multiplier for scaling the keypoint size to determine the descriptor window.
            descriptor_max_value (float): Maximum value for clipping the descriptor elements.

        Returns:
            np.ndarray: Array of keypoint descriptors.
        """
        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            gaussian_image = gaussian_images[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if -1 < row_bin < window_width and -1 < col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_tolerance)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')
