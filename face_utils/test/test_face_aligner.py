from unittest import TestCase, main
from skimage import io
from face_utils.face_aligner import FaceAligner


class Test(TestCase):
    input_img = io.imread('face_utils/test/assets/aflw-test.jpg')

    def test_output_width_height(self):
        output_width = 200
        output_height = 300
        fa = FaceAligner(output_width, output_height, device='cpu')
        outputs = fa.align_from_image(self.input_img)
        for output in outputs:
            self.assertEqual(output_width, output.shape[1])
            self.assertEqual(output_height, output.shape[0])

if __name__ == '__main__':
    main()
 