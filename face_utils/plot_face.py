import cv2
import numpy as np

import face_utils
# from .face_aligner import LandmarksType

class PlotFace:
    def __init__(self, landmarks_type=face_utils.LandmarksType._3D):
        self.landmarks_type = landmarks_type
    
    def plot(self, face_image, landmarks, is_white_bg=True):        
        output = None
        if(is_white_bg):
            output = np.ones(shape=[face_image.shape[0], face_image.shape[1], 3], dtype=np.uint8) * 255
        else:
            output = np.zeros(shape=[face_image.shape[0], face_image.shape[1], 3], dtype=np.uint8)

        # face
        self._plot_element(output, landmarks[0:17], (0,255,0))

        # eyebrows
        self._plot_element(output, landmarks[17:22], (255,0,0))
        self._plot_element(output, landmarks[22:27], (255,0,0))

        # nose
        self._plot_element(output, landmarks[27:31], (255,255,0))
        self._plot_element(output, landmarks[31:36], (255,255,0))

        # eyes
        self._plot_element(output, landmarks[36:42], (0,0,255), is_closed=True)
        self._plot_element(output, landmarks[42:48], (0,0,255), is_closed=True)

        # mouth
        self._plot_element(output, landmarks[48:60], (255,100,255), is_closed=True)
        self._plot_element(output, landmarks[60:68], (255,100,255), is_closed=True)

        return output

    def _plot_element(self, output, landmarks, color, connect_line=True, is_closed=False, marker_size=0):
        if(landmarks is None or len(landmarks) == 0):
            return

        X = landmarks[:, 0]
        Y = landmarks[:, 1]

        if(marker_size > 0):
            for i in range(0,len(X)):
                cv2.circle(output, (X[i], Y[i]), 1, color, marker_size)

        XY = np.array([np.column_stack((X, Y))], np.int32)
        XY= XY.reshape((-1, 1, 2))
        cv2.polylines(output, [XY], is_closed, color)
