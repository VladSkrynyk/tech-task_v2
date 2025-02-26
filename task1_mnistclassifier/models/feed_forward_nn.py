# Importing necessary libraries from TensorFlow and the MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from .mnist_classifier_interface import MnistClassifierInterface

class FeedForwardNNMnist(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28)):
        """
        Initialize the Feed-Forward Neural Network model for MNIST classification.
        :param input_shape: The shape of the input data (default is (28, 28) for MNIST images).
        """
        # Building the neural network model using Keras Sequential API
        self.model = keras.Sequential([
            # Flatten the input images to a 1D array
            keras.layers.Flatten(input_shape=input_shape),

            # First Dense layer with 128 units, followed by BatchNormalization, ReLU activation, and Dropout
            keras.layers.Dense(128),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.2),

            # Second Dense layer with 256 units, followed by BatchNormalization, ReLU activation, and Dropout
            keras.layers.Dense(256),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.2),

            # Output layer with 10 units (for 10 classes) and softmax activation
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile the model with Adam optimizer and sparse categorical crossentropy loss function
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        """
        Train the neural network model on the training data.
        :param X_train: Training data features.
        :param y_train: Training data labels.
        """
        print("Training neural network...")
        # Train the model for 10 epochs with the given training data
        self.model.fit(X_train, y_train, epochs=10, verbose=1)

    def predict(self, X_test):
        """
        Predict labels for the test data using the trained model.
        :param X_test: Test data features.
        :return: Predicted labels for the test data.
        """
        print("Making predictions using neural network...")
        # Predict the labels for the test data
        return self.model.predict(X_test)

    def test(self, X_test, y_test):
        """
        Test the performance of the neural network model.
        :param X_test: Test data features.
        :param y_test: True labels for the test data.
        """
        print("Evaluating neural network on the test data...")
        # Evaluate the model on the test data and print the results
        print(self.model.evaluate(X_test, y_test))
