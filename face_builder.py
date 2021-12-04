import argparse
import os
import pathlib

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class FaceBuilder:
    def __init__(self, dataset_path, batch_size, epochs, image_width, image_height):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.image_width = image_width
        self.image_height = image_height

    def __enter__(self):
        self.dataset_directory = pathlib.Path(self.dataset_path)

        image_count = len(list(self.dataset_directory.glob("*/*.jpg")))

        dataset_list = tf.data.Dataset.list_files(str(self.dataset_directory / "*/*.jpg"), shuffle=False).shuffle(image_count, reshuffle_each_iteration=False)

        self.class_names = np.array(sorted([ item.name for item in self.dataset_directory.glob("*/") ]))

        print("Class names: {0}".format(self.class_names))

        self.num_classes = len(self.class_names)

        validation_split = 0.2
        validation_size = int(image_count * validation_split)

        self.training_dataset = dataset_list.skip(validation_size)
        self.validation_dataset = dataset_list.take(validation_size)

        def process_path(file_path):
            parts = tf.strings.split(file_path, os.path.sep)
            
            one_hot = parts[-2] == self.class_names

            label = tf.argmax(one_hot)

            image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)

            image = tf.image.resize(image, [ self.image_height, self.image_width ])

            return image, label

        self.training_dataset = self.training_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        '''
        def flip(image, label):
            image = tf.image.random_flip_left_right(image)

            return image, label

        training_dataset_flip = self.training_dataset.map(flip, num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset_flip = self.validation_dataset.map(flip, num_parallel_calls=tf.data.AUTOTUNE)

        self.training_dataset = self.training_dataset.concatenate(training_dataset_flip)
        self.validation_dataset = self.validation_dataset.concatenate(validation_dataset_flip)

        def color(image, label):
            image = tf.image.random_saturation(image, 0.6, 1.6)
            image = tf.image.random_brightness(image, 0.05)
            image = tf.image.random_contrast(image, 0.7, 1.3)

            return image, label

        training_dataset_color = self.training_dataset.map(color, num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset_color = self.validation_dataset.map(color, num_parallel_calls=tf.data.AUTOTUNE)

        self.training_dataset = self.training_dataset.concatenate(training_dataset_color)
        self.validation_dataset = self.validation_dataset.concatenate(validation_dataset_color)
        '''

        def scale(image, label):
            image = tf.cast(image, tf.float32) / 255.0

            return image, label

        self.training_dataset = self.training_dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE)

        self.training_dataset = self.training_dataset.cache().shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.cache().shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def build(self):
        self.model = Sequential([
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(self.image_height, self.image_width, 3)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=4096, activation="relu"),
            layers.Dense(units=4096, activation="relu"),
            layers.Dense(units=self.num_classes)
        ])
        
        self.model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

        self.model.build((self.image_height, self.image_width, 3))

        self.model.summary()

        self.history = self.model.fit(self.training_dataset, validation_data=self.validation_dataset, epochs=self.epochs)

    def save(self):
        self.model.save("model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)

    args = parser.parse_args()

    with FaceBuilder(args.dataset_path, args.batch_size, args.epochs, args.image_width, args.image_height) as face_builder:
        face_builder.build()
        face_builder.save()
    
