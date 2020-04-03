import os
from unittest import TestCase, main
from skimage import io
# from face_utils.face_aligner import FaceAligner
from face_utils import FaceAligner, AlignedResult


class Test(TestCase):
    one_face_img = io.imread('./test/assets/one.jpg')
    four_faces_img = io.imread('./test/assets/four.jpg')

    def test_output_width_height(self):
        output_width = 200
        output_height = 300
        fa = FaceAligner(output_width, output_height, device='cpu')
        result = fa.align_from_image(self.one_face_img)
        for output in result.aligned_faces:
            self.assertEqual(output_width, output.shape[1])
            self.assertEqual(output_height, output.shape[0])

    def test_multiple_faces(self):        
        fa = FaceAligner(200, 200, device='cpu')
        result = fa.align_from_image(self.four_faces_img)
        self.assertEqual(len(result.aligned_faces), 4)

    def test_align_from_directory_non_recursive(self):
        img_dir_path = './test/assets'
        fa = FaceAligner(200, 200, device='cpu')
        results = fa.align_from_directory(os.path.abspath(img_dir_path), recursive=False)
        number_of_faces = 0
        for result in results:
            number_of_faces += len(result.aligned_faces)
        self.assertEqual(number_of_faces, 5)

    def test_align_from_directory_recursive(self):
        img_dir_path = './test/assets'
        fa = FaceAligner(200, 200, device='cpu')
        results = fa.align_from_directory(os.path.abspath(img_dir_path), recursive=True)
        self.assertEqual(len(results), 3)
        number_of_faces = 0
        for result in results:
            number_of_faces += len(result.aligned_faces)
        self.assertEqual(number_of_faces, 10)

    def test_aligned_result(self):
        result = AlignedResult()

        result.aligned_faces = ['face1', 'face2', 'face3']
        self.assertEqual(len(result.aligned_faces), 3)
        with self.assertRaises(ValueError):
            result.aligned_faces = None

        result.aligned_LMs = ['LM1', 'LM2', 'LM3']
        self.assertEqual(len(result.aligned_LMs), 3)
        with self.assertRaises(ValueError):
            result.aligned_LMs = None

        result.LM_imgs = ['LM_img1', 'LM_img2', 'LM_img3']
        self.assertEqual(len(result.LM_imgs), 3)
        with self.assertRaises(ValueError):
            result.LM_imgs = None

if __name__ == '__main__':
    main()
 