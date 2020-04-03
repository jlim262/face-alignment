import cv2
import numpy as np

# import face_utils
# from .face_aligner import LandmarksType
from face_utils.landmarks_type import LandmarksType


class FacePlot:
    def __init__(self, landmarks_type=LandmarksType._3D, use_face=False, use_white_bg=True):
        self.landmarks_type = landmarks_type
        self.use_face = use_face
        self.use_white_bg = use_white_bg
    
    def plot(self, face_frame, landmarks):        
        output = None
        if(self.use_face):
            output = face_frame
        elif(self.use_white_bg):
            output = np.ones(shape=[face_frame.shape[0], face_frame.shape[1], 3], dtype=np.uint8) * 255
        else:
            output = np.zeros(shape=[face_frame.shape[0], face_frame.shape[1], 3], dtype=np.uint8)        

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

# import face_alignment
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from skimage import io
# import collections


# # Run the 3D face alignment on a test image, without CUDA.
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)

# try:
#     input_img = io.imread('./aflw-test.jpg')
# except FileNotFoundError:
#     input_img = io.imread('test/assets/aflw-test.jpg')

# preds = fa.get_landmarks(input_img)[-1]

# # 2D-Plot
# plot_style = dict(marker='o',
#                   markersize=4,
#                   linestyle='-',
#                   lw=2)

# pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
# pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
#               'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
#               'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
#               'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
#               'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
#               'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
#               'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
#               'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
#               'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
#               }

# fig = plt.figure(figsize=plt.figaspect(.5))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(input_img)

# for pred_type in pred_types.values():
#     ax.plot(preds[pred_type.slice, 0],
#             preds[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)

# ax.axis('off')

# # 3D-Plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(preds[:, 0] * 1.2,
#                   preds[:, 1],
#                   preds[:, 2],
#                   c='cyan',
#                   alpha=1.0,
#                   edgecolor='b')

# for pred_type in pred_types.values():
#     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
#               preds[pred_type.slice, 1],
#               preds[pred_type.slice, 2], color='blue')

# ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
# plt.show()