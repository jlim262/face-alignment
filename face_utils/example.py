import matplotlib.pyplot as plt
from skimage import io

from face_utils.face_aligner import FaceAligner, AlignedResult
from face_utils.face_plot import FacePlot
from face_utils.landmarks_type import LandmarksType

if __name__ == '__main__':
    img = io.imread('./test/assets/sub_dir/five.jpg')
    fa = FaceAligner(256, 256, device='cpu', landmarks_type=LandmarksType._3D)

    plotter = FacePlot(use_face=True)
    result = fa.align_from_image(img, plotter=plotter)

    fig = plt.figure(figsize=(2, 2))
    col = 10
    row = 10
    for i, landmarks_img in enumerate(result.LM_imgs):
        fig.add_subplot(row, col, i+1)
        # plt.imshow(output)
        plt.imshow(landmarks_img)

    # plt.imshow(output[0])
    plt.show()