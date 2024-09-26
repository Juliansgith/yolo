# test_models.py

import unittest
import argparse
import sys
import cv2
from cnn import process_image as cnn_process_image
from ssd import process_image as ssd_process_image
from vision import YOLO

class TestCNN(unittest.TestCase):
    def test_cnn_process_image(self):
        print("Start test_cnn_process_image")
        image = cv2.imread('test_image.jpg')
        self.assertIsNotNone(image, "Kon testafbeelding niet laden.")
        boxes, labels, scores, inference_time = cnn_process_image(image)
        self.assertIsInstance(boxes, (list, tuple, np.ndarray), "Boxes moeten een lijst, tuple of ndarray zijn.")
        self.assertIsInstance(labels, (list, tuple, np.ndarray), "Labels moeten een lijst, tuple of ndarray zijn.")
        self.assertIsInstance(scores, (list, tuple, np.ndarray), "Scores moeten een lijst, tuple of ndarray zijn.")
        print("Einde test_cnn_process_image")

class TestSSD(unittest.TestCase):
    def test_ssd_process_image(self):
        print("Start test_ssd_process_image")
        image = cv2.imread('test_image.jpg')
        self.assertIsNotNone(image, "Kon testafbeelding niet laden.")
        boxes, labels, scores, inference_time = ssd_process_image(image)
        self.assertIsInstance(boxes, (list, tuple, np.ndarray), "Boxes moeten een lijst, tuple of ndarray zijn.")
        self.assertIsInstance(labels, (list, tuple, np.ndarray), "Labels moeten een lijst, tuple of ndarray zijn.")
        self.assertIsInstance(scores, (list, tuple, np.ndarray), "Scores moeten een lijst, tuple of ndarray zijn.")
        print("Einde test_ssd_process_image")

class TestYOLO(unittest.TestCase):
    def test_yolo_process_image(self):
        print("Start test_yolo_process_image")
        image = cv2.imread('test_image.jpg')
        self.assertIsNotNone(image, "Kon testafbeelding niet laden.")
        model = YOLO('yolov8n.pt')  
        results = model(image)
        self.assertIsNotNone(results, "Resultaten van YOLO mogen niet None zijn.")
        self.assertGreaterEqual(len(results), 1, "Er moeten minstens één resultaat zijn.")
        print("Einde test_yolo_process_image")

def main():
    parser = argparse.ArgumentParser(description='Selecteer welke modellen je wilt testen.')
    parser.add_argument('--models', nargs='+', choices=['cnn', 'ssd', 'yolo'], default=['cnn', 'ssd', 'yolo'],
                        help='Specificeer de modellen die je wilt testen: cnn, ssd, yolo')
    args = parser.parse_args()

    suite = unittest.TestSuite()

    if 'cnn' in args.models:
        suite.addTest(unittest.makeSuite(TestCNN))
    if 'ssd' in args.models:
        suite.addTest(unittest.makeSuite(TestSSD))
    if 'yolo' in args.models:
        suite.addTest(unittest.makeSuite(TestYOLO))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main()
