import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections

from face_utils.landmarks_type import LandmarksType

class FacePlot(object):
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (255, 0, 255)),
                    'eyebrow1': pred_type(slice(17, 22), (0, 0, 255)),
                    'eyebrow2': pred_type(slice(22, 27), (0, 0, 255)),
                    'nose': pred_type(slice(27, 31), (0, 255, 0)),
                    'nostril': pred_type(slice(31, 36), (0, 255, 0)),
                    'eye1': pred_type(slice(36, 42), (0, 255, 255)),
                    'eye2': pred_type(slice(42, 48), (0, 255, 255)),
                    'lips': pred_type(slice(48, 60), (255, 0, 0)),
                    'teeth': pred_type(slice(60, 68), (255, 0, 0))
                    }
    
    @classmethod
    def plot(cls, face_frame, landmarks, use_face=False, use_white_bg=True):        
        output = None
        if(use_face):
            output = face_frame
        elif(use_white_bg):
            output = np.ones(shape=[face_frame.shape[0], face_frame.shape[1], 3], dtype=np.uint8) * 255
        else:
            output = np.zeros(shape=[face_frame.shape[0], face_frame.shape[1], 3], dtype=np.uint8)        

        for key, pred_type in cls.pred_types.items():
            if key in ['eye1, eye2, lips, teeth']:
                cls._plot_element(output, landmarks[pred_type.slice], pred_type.color, is_closed=True)
            else:
                cls._plot_element(output, landmarks[pred_type.slice], pred_type.color)

        return output

    @classmethod
    def _plot_element(cls, output, landmarks, color, connect_line=True, is_closed=False, marker_size=0):
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

    @classmethod
    def plot_show(cls, face_frame, landmarks, use_face=False):
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(cls.plot(face_frame, landmarks, use_face))
        ax.axis('off')

        # 3D-Plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(landmarks[:, 0] * 1.2,
                        landmarks[:, 1],
                        landmarks[:, 2],
                        c='cyan',
                        alpha=1.0,
                        edgecolor='b')

        for pred_type in cls.pred_types.values():
            ax.plot3D(landmarks[pred_type.slice, 0] * 1.2,
                    landmarks[pred_type.slice, 1],
                    landmarks[pred_type.slice, 2], color='blue')

        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()