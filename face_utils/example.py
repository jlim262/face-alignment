import matplotlib.pyplot as plt
from skimage import io
from face_aligner import FaceAligner

if __name__ == '__main__':
    # img = io.imread('face_utils/test/assets/aflw-test.jpg')
    img = io.imread('./face_utils/test/assets/sub_dir/five.jpg')
    fa = FaceAligner(200, 200, device='cpu')
    outputs = fa.align_from_image(img)

    fig = plt.figure(figsize=(8, 8))
    col = 10
    row = 1
    for i, output in enumerate(outputs):
        fig.add_subplot(row, col, i+1)
        plt.imshow(output)

    # plt.imshow(output[0])
    plt.show()