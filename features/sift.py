from functools import cmp_to_key
import numpy as np
import time
import cv2
import pyqtgraph as pg

class SIFT:
    def __init__(self, original_image,tab_widget):
        self.ui = tab_widget
        self.start_time = time.time()
        # original_image = cv2.imread(file_path,0)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # convert image to gray scale
        self.original_image = original_image.astype('float32')
        

    def sift(self, sigma=1.6, no_of_levels=3, assumed_blur=0.5, image_border_width=5):
        # Creating the Scale Space
        image = cv2.resize(self.original_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
        num_octaves = int(np.round(np.log(min(image.shape)) / np.log(2) - 1))
        gaussian_pyramid = self.create_gaussian_pyramid(image, num_octaves, sigma, no_of_levels)
        DoG_pyramid = self.create_DoG_pyramid(gaussian_pyramid)

        # Extracting Keypoints and Descriptors
        keypoints = self.localize_keypoints(gaussian_pyramid, DoG_pyramid, no_of_levels, sigma, image_border_width)
        keypoints = self.removeDuplicateKeypoints(keypoints)
        keypoints = self.convertKeypointsToInputImageSize(keypoints)
        self.display_image(keypoints)
        descriptors = self.generate_descriptors(keypoints, gaussian_pyramid)

        print(f"SIFT Computation time: {time.time() - self.start_time}")

        return keypoints, descriptors
    

    def display_image(self,keypoints_list):
        image = np.copy(self.original_image)
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for keypoint in keypoints_list:
            y,x = int(np.round(keypoint.pt[0])), int(np.round(keypoint.pt[1]))
            image[x:x+3,y:y+3,0] = 255
            image[x:x+3,y:y+3,1] = 0
            image[x:x+3,y:y+3,2] = 0

        image= np.rot90(image, -1)
        # cv2.imwrite('keypoints2.png', image)
        self.ui.graphicsLayoutWidget_afterSIFT.clear()
        view_box = self.ui.graphicsLayoutWidget_afterSIFT.addViewBox()
        image_item = pg.ImageItem(image)
        view_box.addItem(image_item)
        view_box.autoRange()
    
    def display_discriptors(self,descriptors):
        image = np.copy(self.original_image)
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # for descriptor in descriptors:
        #     y, x = int(np.round(descriptor[0])), int(np.round(descriptor[1]))
        #     image[x:x+3, y:y+3, 0] = 255
        #     image[x:x+3, y:y+3, 1] = 0
        #     image[x:x+3, y:y+3, 2] = 0
        # Draw keypoints on the image
        descriptors = [cv2.KeyPoint(x=np.round(descriptor[0]), y=np.round(descriptor[1]), _size=1) for descriptor in descriptors]
    
        descriptor_image = cv2.drawKeypoints(image, descriptors, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Convert image color to RGB (for matplotlib display)
        descriptor_image_rgb = cv2.cvtColor(descriptor_image, cv2.COLOR_BGR2RGB)

        image= np.rot90(descriptor_image_rgb, -1)
        # cv2.imwrite('keypoints2.png', image)
        self.ui.graphicsLayoutWidget_discriptors.clear()
        view_box = self.ui.graphicsLayoutWidget_discriptors.addViewBox()
        image_item = pg.ImageItem(image)
        view_box.addItem(image_item)
        view_box.autoRange()

    def calculate_sigma_values(self, sigma, no_of_levels):
        num_images_per_octave = no_of_levels + 3
        k = 2 ** (1. / no_of_levels)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        
        return gaussian_kernels


    def create_gaussian_pyramid(self, image, num_octaves, sigma, no_of_levels):
        
        gaussian_pyramid = []
        sigmas = self.calculate_sigma_values(sigma, no_of_levels)
        
        gaussian_pyramid = []
        for octave in range(num_octaves):
            gaussian_levels = []
            
            for sigma in sigmas:
                if len(gaussian_levels) == 0:
                    gaussian_levels.append(image)
                else:
                    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    gaussian_levels.append(image)
            
            gaussian_pyramid.append(gaussian_levels)

            if octave < num_octaves - 1:
                image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        
        return np.array(gaussian_pyramid, dtype=object)


    def create_DoG_pyramid(self, gaussian_pyramid):
        
        DoG_pyramid = []

        for gaussian_levels in gaussian_pyramid:
            DoG_octave = []

            for first_image, second_image in zip(gaussian_levels, gaussian_levels[1:]):
                DoG_octave.append(np.subtract(second_image, first_image))
            
            DoG_pyramid.append(DoG_octave)

        return np.array(DoG_pyramid, dtype=object)


    def localize_keypoints(self, gaussian_pyramid, DoG_pyramid, no_of_levels, sigma, image_border_width, contrast_threshold=0.04):
        threshold = np.floor(0.5 * contrast_threshold / no_of_levels * 255)
        keypoints = []

        for octave_index, DoG_octave in enumerate(DoG_pyramid):
            for image_index, (first_image, second_image, third_image) in enumerate(zip(DoG_octave, DoG_octave[1:], DoG_octave[2:])):
                for i in range(image_border_width, first_image.shape[0] - image_border_width):
                    for j in range(image_border_width, first_image.shape[1] - image_border_width):
                        if self.is_extreme(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                            localization_result = self.localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, no_of_levels, DoG_octave, sigma, contrast_threshold, image_border_width)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = self.computeKeypointsWithOrientations(keypoint, octave_index, gaussian_pyramid[octave_index][localized_image_index])
                                for keypoint_with_orientation in keypoints_with_orientations:
                                    keypoints.append(keypoint_with_orientation)

        return keypoints

    def is_extreme(self, first_subimage, second_subimage, third_subimage, threshold):
        center_pixel_value = second_subimage[1, 1]

        # Check if all elements satisfy the condition
        if np.all(np.abs(center_pixel_value) <= threshold):
            return False

        if np.all(center_pixel_value > 0):
            return (
                np.all(center_pixel_value >= first_subimage.max()) and
                np.all(center_pixel_value >= third_subimage.max()) and
                np.all(center_pixel_value >= second_subimage.max())
            )
        else:
            return (
                np.all(center_pixel_value <= first_subimage.min()) and
                np.all(center_pixel_value <= third_subimage.min()) and
                np.all(center_pixel_value <= second_subimage.min())
            )




    def localizeExtremumViaQuadraticFit(self, i, j, image_index, octave_index, no_of_levels, DoG_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
        extremum_is_outside_image = False
        image_shape = DoG_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            first_image, second_image, third_image = DoG_octave[image_index-1:image_index+2]
            pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                                second_image[i-1:i+2, j-1:j+2],
                                third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.computeGradientAtCenterPixel(pixel_cube)
            hessian = self.computeHessianAtCenterPixel(pixel_cube)
            extremum_update = -np.linalg.lstsq(hessian.reshape(-1, 3), gradient.reshape(-1, 1), rcond=None)[0]
            if np.abs(extremum_update[0]) < 0.5 and np.abs(extremum_update[1]) < 0.5 and np.abs(extremum_update[2]) < 0.5:
                break
            j += int(np.round(extremum_update[0]))
            i += int(np.round(extremum_update[1]))
            image_index += int(np.round(extremum_update[2]))
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > no_of_levels:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if np.all(np.abs(functionValueAtUpdatedExtremum) * no_of_levels >= contrast_threshold):
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian = hessian[:2, :2]
            if xy_hessian.shape[0] != xy_hessian.shape[1]:
                return None
            xy_hessian_det = np.linalg.det(xy_hessian)

            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((j + extremum_update[0][0]) * (2 ** octave_index)), ((i + extremum_update[1][0]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = int(sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(no_of_levels))) * (2 ** (octave_index + 1)))
                keypoint.response = abs(functionValueAtUpdatedExtremum[0])
                return keypoint, image_index
        return None

    def computeGradientAtCenterPixel(self, pixel_array):
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def computeHessianAtCenterPixel(self, pixel_array):
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])

    #########################
    # Keypoint orientations #
    #########################

    def computeKeypointsWithOrientations(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        radius = int(np.round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(np.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(np.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_index = int(np.round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < 1e-7:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    ##############################
    # Duplicate keypoint removal #
    ##############################

    def compareKeypoints(self, keypoint1, keypoint2):
        """Return True if keypoint1 is less than keypoint2
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

    def removeDuplicateKeypoints(self, keypoints):
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compareKeypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints

    #############################
    # Keypoint scale conversion #
    #############################

    def convertKeypointsToInputImageSize(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints


    def unpackOctave(self, keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale

    def generate_descriptors(self, keypoints, gaussian_pyramid, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self.unpackOctave(keypoint)
            gaussian_image = gaussian_pyramid[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            np.cos_angle = np.cos(np.deg2rad(angle))
            np.sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * np.sin_angle + row * np.cos_angle
                    col_rot = col * np.cos_angle - row * np.sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(np.round(point[1] + row))
                        window_col = int(np.round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
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
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        self.display_discriptors(descriptors)
        return np.array(descriptors, dtype='float32')
