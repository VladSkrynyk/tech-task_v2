# Importing necessary libraries from TensorFlow and the MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from .mnist_classifier_interface import MnistClassifierInterface

class CNNMnist(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1)):
        """
        Initialize the Convolutional Neural Network model for MNIST classification.
        :param input_shape: The shape of the input data (default is (28, 28, 1) for MNIST images).
        """
        # Building the CNN model using Keras Sequential API
        self.model = keras.Sequential([
            # First convolutional layer with 32 filters and 3x3 kernel size, followed by BatchNormalization and MaxPooling
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            # Second convolutional layer with 64 filters and 3x3 kernel size, followed by BatchNormalization and MaxPooling
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            # Third convolutional layer with 128 filters and 3x3 kernel size, followed by BatchNormalization and MaxPooling
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            # Flatten the output from the convolutional layers to a 1D array
            keras.layers.Flatten(),

            # Dropout layer with 50% dropout to regularize and prevent overfitting
            keras.layers.Dropout(0.5),

            # Dense layer with 128 units and ReLU activation
            keras.layers.Dense(128, activation='relu'),

            # Output layer with 10 units (for 10 classes) and softmax activation
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile the model with Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        """
        Train the CNN model on the training data.
        :param X_train: Training data features.
        :param y_train: Training data labels.
        """
        print("Training CNN...")
        # Train the model for 5 epochs with the given training data
        self.model.fit(X_train, y_train, epochs=5)

    def predict(self, X_test):
        """
        Predict labels for the test data using the trained CNN model.
        :param X_test: Test data features.
        :return: Predicted labels for the test data.
        """
        print("Making predictions using CNN...")
        # Predict the labels for the test data
        return self.model.predict(X_test)

    def test(self, X_test, y_test):
        """
        Test the performance of the CNN model.
        :param X_test: Test data features.
        :param y_test: True labels for the test data.
        """
        print("Evaluating CNN on the test data...")
        # Evaluate the model on the test data and print the results
        print(self.model.evaluate(X_test, y_test))
