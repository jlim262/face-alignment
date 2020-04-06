import os
from enum import Enum

import face_alignment
import numpy as np
import cv2
from skimage import io

from face_utils.face_plot import FacePlot
from face_utils.landmarks_type import LandmarksType

def emptylist2none(dlist):
    return None if not dlist else dlist

def is_list(data):
    if(isinstance(data, list)):
        return data
    else:
        raise ValueError('Input parameter is not list type.')

class AlignedResult:
    def __init__(self):
        self.aligned_faces = []
        self.aligned_LMs = []
        self.LM_imgs = []
    
    @property
    def aligned_faces(self):
        return self.__aligned_faces

    @aligned_faces.setter
    def aligned_faces(self, val):
        self.__aligned_faces = is_list(val)

    @property
    def aligned_LMs(self):
        return self.__aligned_LMs

    @aligned_LMs.setter
    def aligned_LMs(self, val):
        self.__aligned_LMs = is_list(val)

    @property
    def LM_imgs(self):
        return self.__LM_imgs

    @LM_imgs.setter
    def LM_imgs(self, val):
        self.__LM_imgs = is_list(val)


class FaceAligner:
    def __init__(self, output_width, output_height=None, landmarks_type=LandmarksType._3D, device='cuda', flip_input='False', face_detector='sfd'):
        self.output_width = output_width
        self.output_height = self.output_width if output_height is None else output_height        
        self.landmarks_type = self._to_fa_type(landmarks_type)
        self.fa = face_alignment.FaceAlignment(self.landmarks_type, flip_input=flip_input, device=device, face_detector=face_detector)

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

    def align_from_file(self, path, extensions=['.jpg', '.png'], plot=True, plot_LM_on_face=False):
        if(os.path.isfile(path) and os.path.splitext(path)[1] in extensions):
            img = io.imread(path)
            return self.align_from_image(img, plot, plot_LM_on_face)

    def align_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True, plot=True, plot_LM_on_face=False):
        aligned_results = []
        for root, _, f_names in os.walk(path):
            for f in f_names:
                file_path = os.path.join(root, f)
                ext = os.path.splitext(file_path)[1]
                if(ext in extensions):
                    aligned_result = self.align_from_file(file_path, extensions, plot, plot_LM_on_face)
                    aligned_results.append(aligned_result)

            if not recursive:
                break
        
        return aligned_results

    def align_from_image(self, image, plot=True, plot_LM_on_face=False):
        predictions = self.get_landmarks(image)
        if(predictions is None):
            return None

        aligned_result = AlignedResult()

        output_rect = np.float32([
                            [0, 0],
                            [self.output_width, 0],
                            [0, self.output_height],
                            [self.output_width, self.output_height]])

        for prediction in predictions:
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
            transform = cv2.getPerspectiveTransform(image_corners, output_rect)
            aligned_face = cv2.warpPerspective(image, transform, (self.output_width, self.output_height))
            aligned_result.aligned_faces.append(aligned_face)

            aligned_prediction = self.get_landmarks(aligned_face)[0]            
            aligned_result.aligned_LMs.append(aligned_prediction)
            
            if(plot):
                LM_img = FacePlot.plot(aligned_face, aligned_prediction, use_face=plot_LM_on_face)
                aligned_result.LM_imgs.append(LM_img)

        # return emptylist2none(aligned_faces), emptylist2none(LM_imgs)
        return aligned_result

    def _to_fa_type(self, type):
        if(type == LandmarksType._2D):
            return face_alignment.LandmarksType._2D
        elif(type == LandmarksType._2halfD):
            return face_alignment.LandmarksType._2halfD
        elif(type == LandmarksType._3D):
            return face_alignment.LandmarksType._3D