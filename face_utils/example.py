import matplotlib.pyplot as plt
from skimage import io

from face_utils.face_aligner import FaceAligner, AlignedResult
from face_utils.face_plot import FacePlot
from face_utils.landmarks_type import LandmarksType

if __name__ == '__main__':
    img = io.imread('./test/assets/sub_dir/five.jpg')
    fa = FaceAligner(256, 256, device='cpu', landmarks_type=LandmarksType._3D)
    result = fa.align_from_image(img)

    fig = plt.figure(figsize=plt.figaspect(.5))

    row = 3
    col = len(result.aligned_faces)

    for i, aligned_face in enumerate(result.aligned_faces):
        ax = fig.add_subplot(row, col, i + 1)
        ax.imshow(aligned_face)
        ax.axis('off')

    for i, LM_img in enumerate(result.LM_imgs):
        ax = fig.add_subplot(row, col, (col*1) + i + 1)
        ax.imshow(LM_img)
        ax.axis('off')

    for i, aligned_LM in enumerate(result.aligned_LMs):
        ax = fig.add_subplot(row, col, (col*2) + i + 1)
        LM_frame = FacePlot.plot(face_frame=result.aligned_faces[i], landmarks=aligned_LM, use_face=True)
        ax.imshow(LM_frame)
        ax.axis('off')                

    plt.show()