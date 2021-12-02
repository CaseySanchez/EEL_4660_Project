import argparse
import pathlib

import numpy as np
import cv2

import tensorflow as tf

class FaceRegistrar:
    def __init__(self, dataset_path, class_name, samples, image_width, image_height):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.samples = samples
        self.image_width = image_width
        self.image_height = image_height

    def __enter__(self):
        self.dataset_directory = pathlib.Path(self.dataset_path)

        if not self.dataset_directory.exists():
            self.dataset_directory.mkdir()

        self.class_name_directory = self.dataset_directory / self.class_name

        if not self.class_name_directory.exists():
            self.class_name_directory.mkdir()

        self.video_capture = cv2.VideoCapture(-1)
        
        self.face_cascade = cv2.CascadeClassifier("haar_cascade_face.xml")

    def __exit__(self):
        self.video_capture.release()
        
        cv2.destroyAllWindows()

    def register(self):
        samples = 0

        while samples < self.samples:
            ret, frame = self.video_capture.read()

            if ret:
                faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=10, minSize=(32, 96), flags=cv2.CASCADE_SCALE_IMAGE)

                if len(faces) > 0:
                    x, y, w, h = faces[0]

                    centroid_x = x + w * 0.5
                    centroid_y = y + h * 0.5

                    length = max(h, w)

                    image_cropped = frame[int(centroid_y - length * 0.5) : int(centroid_y + length * 0.5), int(centroid_x - length * 0.5) : int(centroid_x + length * 0.5)]

                    image_resized = cv2.resize(image_cropped, (self.image_width, self.image_height))
                    
                    cv2.imwrite(str(self.class_name_directory / (str(samples) + ".jpg")), image_resized)

                    print("[{0} of {1}] {2}.jpg saved".format(samples, self.samples, samples))

                    samples += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)

    args = parser.parse_args()
    
    with FaceRegistrar(args.dataset_path, args.class_name, args.samples, args.image_width, args.image_height) as face_registrar:
        face_registrar.register()
    