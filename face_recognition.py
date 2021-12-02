import argparse

import numpy as np
import cv2

import tensorflow as tf

class FaceRecognition:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def __enter__(self):
        self.model = tf.keras.models.load_model("model.h5")

        self.face_cascade = cv2.CascadeClassifier("haar_cascade_face.xml")

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def recognize(self, image):
        results = []

        faces = self.face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=(32, 96), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            centroid_x = x + w * 0.5
            centroid_y = y + h * 0.5

            length = max(h, w)

            image_cropped = image[int(centroid_y - length * 0.5) : int(centroid_y + length * 0.5), int(centroid_x - length * 0.5) : int(centroid_x + length * 0.5)]

            image_resized = cv2.resize(image_cropped, (self.image_width, self.image_height))
            
            image_normalized = cv2.normalize(image_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            image_array = np.asarray(image_normalized)
            image_array = tf.expand_dims(image_array, 0)
            
            predictions = self.model.predict(image_array)
            score = tf.nn.softmax(predictions[0])

            results.append((np.argmax(score), np.max(score)))

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)

    args = parser.parse_args()

    with FaceRecognition(args.image_width, args.image_height) as face_recognition:
        video_capture = cv2.VideoCapture(-1)

        while True:
            ret, frame = video_capture.read()

            if ret:
                results = face_recognition.recognize(frame)

                for result in results:
                    print("{0} detected with {1}%% accuracy".format(result[0], result[1] * 100.0))
                
                print("----------")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video_capture.release()
        
        cv2.destroyAllWindows()
