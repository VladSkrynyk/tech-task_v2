# Importing necessary model classes
from .random_forest import RandomForestMnist
from .feed_forward_nn import FeedForwardNNMnist
from .cnn import CNNMnist
from .mnist_classifier_interface import MnistClassifierInterface

class MnistClassifier:
    def __init__(self, algorithm: str):
        # Initialize the model based on the selected algorithm
        if algorithm == 'rf':
            # If the algorithm is 'rf', use the RandomForestMnist model
            self.model = RandomForestMnist()
        elif algorithm == 'nn':
            # If the algorithm is 'nn', use the FeedForwardNNMnist model
            self.model = FeedForwardNNMnist()
        elif algorithm == 'cnn':
            # If the algorithm is 'cnn', use the CNNMnist model
            self.model = CNNMnist()
        else:
            # Raise an error if an invalid algorithm is selected
            raise ValueError("Invalid algorithm. Choose from 'rf', 'nn', or 'cnn'.")

        # Check if the selected model implements the MnistClassifierInterface
        if not isinstance(self.model, MnistClassifierInterface):
            # Raise an error if the model does not implement the required interface
            raise TypeError("The selected model does not implement MnistClassifierInterface")

    def train(self, X_train, y_train):
        # Train the selected model using the training data
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        # Predict labels for the test data using the trained model
        return self.model.predict(X_test)

    def test(self, X_test, y_test):
        # Test the model's performance on the test dataset
        self.model.test(X_test, y_test)
