import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import keras
from keras import layers
from keras import backend as K
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator




class CNNModel:
    def __init__(self, input_shape=(250, 400, 3), num_classes=278):
        self.model = self.build_model(input_shape, num_classes)
        self.label_binarizer = None


    def build_model(self, input_shape, num_classes):
        K.clear_session()
        
        model = Sequential()

        # Add convolutional layers
        model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())

        # Add fully connected layers
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(num_classes, activation="softmax"))

        # Compile the model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
        model.summary()
        return model


    def train(self, X_train, y_train, epochs=20, batch_size=45):
        if self.label_binarizer is None:
            self.label_binarizer = LabelBinarizer()

        # Convert labels to one-hot encoded vectors using LabelBinarizer
        y_train_encoded = self.label_binarizer.fit_transform(y_train)
        self.model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size)
    
    
    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_labels = self.label_binarizer.inverse_transform(predictions)
        return predicted_labels


