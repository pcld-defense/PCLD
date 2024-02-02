import unittest
import test_new


class MainTest(unittest.TestCase):

    # @unittest.skip("integrative")
    # noinspection SpellCheckingInspection
    def test_run_integrative(self):
        args = test_new.Arguments(max_step=80,
                                  actor='/Users/royambar/Desktop/rmbg/attack/learning_to_paint/LearningToPaint/actor.pkl',
                                  renderer='/Users/royambar/Desktop/rmbg/attack/learning_to_paint/LearningToPaint/renderer.pkl',
                                  img='/Users/royambar/Desktop/rmbg/attack/learning_to_paint/LearningToPaint/images_temp/n02123045_871.png',
                                  imgid=0,
                                  divide=5,
                                  output_dir='/Users/royambar/Desktop/rmbg/attack/learning_to_paint/LearningToPaint/images_temp/output_temp',
                                  output_img_name='n02123045_871',
                                  save_every=700,
                                  save_strokes=False,
                                  verbose=False)

        test_new.main(args)
        self.assertTrue(True)