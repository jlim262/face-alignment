import matplotlib.pyplot as plt
from skimage import io
from face_aligner import FaceAligner

if __name__ == '__main__':
    img = io.imread('face_utils/test/assets/aflw-test.jpg')
    fa = FaceAligner(200, 200, device='cpu')
    output = fa.align_from_image(img)
    plt.imshow(output[0])
    plt.show()