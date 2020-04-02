import os
from enum import Enum

import face_alignment
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

import face_utils


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class FaceAligner:
    def __init__(self, output_width, output_height=None, landmarks_type=LandmarksType._3D, device='cuda', flip_input='False', face_detector='sfd'):
        self.output_width = output_width
        self.output_height = self.output_width if output_height is None else output_height
        self.output_rect = np.float32([
                            [0, 0],
                            [self.output_width, 0],
                            [0, self.output_height],
                            [self.output_width, self.output_height]])
        self.landmarks_type = self._to_fa_type(landmarks_type)
        self.fa = face_alignment.FaceAlignment(self.landmarks_type, flip_input=flip_input, device=device, face_detector=face_detector)
        self.plotter = face_utils.PlotFace(landmarks_type)

    def get_aligned_landmarks_from_images(self, images):
        if(images is None):
            raise ValueError("images parameter should not be None.")
        if (isinstance(images, list)):
            raise ValueError("images parameter should be a list.")
        
        outputs = []
        for image in images:
            aligned_images = self.align_from_image(image)
            for ai in aligned_images:
                output = self.get_landmarks(ai)
                outputs.append(output[0])

        return outputs

    def get_landmarks(self, image):
        predictions = self.fa.get_landmarks_from_image(image)
        return predictions

    def align_from_file(self, path, extensions=['.jpg', '.png']):
        if(os.path.isfile(path) and os.path.splitext(path)[1] in extensions):
            img = io.imread(path)
            return self.align_from_image(img)

    def align_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True):
        outputs = []
        for root, _, f_names in os.walk(path):
            for f in f_names:
                file_path = os.path.join(root, f)
                ext = os.path.splitext(file_path)[1]
                if(ext in extensions):
                    outputs.append(self.align_from_file(file_path, extensions))

            if not recursive:
                break
        
        return outputs

    def align_from_image(self, image, landmark=False, generate_landmarks_image=True):
        predictions = self.get_landmarks(image)
        if(predictions is None):
            return None

        outputs = []
        landmarks_imgs = []
        for i, prediction in enumerate(predictions):

            # Compute the Anchor Landmarks
            # This ensures the eyes and chin will not move within the output
            right_eye_mean = np.mean(prediction[36:42], axis=0)
            left_eye_mean = np.mean(prediction[42:47], axis=0)
            middle_of_eye = (right_eye_mean + left_eye_mean) * 0.5
            chin = prediction[8]

            # Compute the output center and up/side vectors
            mean = ((middle_of_eye * 3) + chin) * 0.25
            centered = prediction - mean 
            right_vector = (left_eye_mean - right_eye_mean)
            up_vector = (chin - middle_of_eye)

            # Divide by the length ratio to ensure a square aspect ratio
            right_vector /= np.linalg.norm(right_vector) / np.linalg.norm(up_vector)

            # Compute the corners of the facial output
            image_corners = np.float32([(mean + ((-right_vector - up_vector)))[:2],
                                       (mean + (( right_vector - up_vector)))[:2],
                                       (mean + ((-right_vector + up_vector)))[:2],
                                       (mean + (( right_vector + up_vector)))[:2]])

            # Compute the Perspective Homography and Extract the output from the image
            transform = cv2.getPerspectiveTransform(image_corners, self.output_rect)
            output = cv2.warpPerspective(image, transform, (self.output_width, self.output_height))
            outputs.append(output)

            pred = self.get_landmarks(output)
            if(generate_landmarks_image):
                landmarks_img = self.plotter.plot(output, pred[0])
                landmarks_imgs.append(landmarks_img)
                # cv2.imwrite('./test_{}.png'.format(str(i)),landmarks_img)
        
        return outputs, landmarks_imgs

    def _to_fa_type(self, type):
        if(type == LandmarksType._2D):
            return face_alignment.LandmarksType._2D
        elif(type == LandmarksType._2halfD):
            return face_alignment.LandmarksType._2halfD
        elif(type == LandmarksType._3D):
            return face_alignment.LandmarksType._3D