import os
from unittest import TestCase, main
from skimage import io
from face_utils.face_aligner import FaceAligner


class Test(TestCase):
    one_face_img = io.imread('./face_utils/test/assets/one.jpg')
    four_faces_img = io.imread('./face_utils/test/assets/four.jpg')

    def test_output_width_height(self):
        output_width = 200
        output_height = 300
        fa = FaceAligner(output_width, output_height, device='cpu')
        outputs = fa.align_from_image(self.one_face_img)
        for output in outputs:
            self.assertEqual(output_width, output.shape[1])
            self.assertEqual(output_height, output.shape[0])

    def test_multiple_faces(self):        
        fa = FaceAligner(200, 200, device='cpu')
        outputs = fa.align_from_image(self.four_faces_img)
        self.assertEqual(len(outputs), 4)

    def test_align_from_directory_non_recursive(self):
        img_dir_path = './face_utils/test/assets'
        fa = FaceAligner(200, 200, device='cpu')
        outputs = fa.align_from_directory(os.path.abspath(img_dir_path), recursive=False)
        number_of_faces = 0
        for output in outputs:
            number_of_faces += len(output)
        self.assertEqual(number_of_faces, 5)

    def test_align_from_directory_recursive(self):
        img_dir_path = './face_utils/test/assets'
        fa = FaceAligner(200, 200, device='cpu')
        outputs = fa.align_from_directory(os.path.abspath(img_dir_path), recursive=True)
        self.assertEqual(len(outputs), 3)
        number_of_faces = 0
        for output in outputs:
            number_of_faces += len(output)
        self.assertEqual(number_of_faces, 10)


if __name__ == '__main__':
    main()
 