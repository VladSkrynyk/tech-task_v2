# Importing ABC and abstractmethod from the abc module to define an abstract base class
from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    This is an abstract base class (ABC) for all MNIST classifier models.
    Any class that inherits from this interface must implement the `train`, `predict`, and `test` methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        This method is responsible for training the model using the provided training data.
        :param X_train: Features of the training dataset.
        :param y_train: Labels of the training dataset.
        """
        pass  # Abstract method, should be implemented in the child class

    @abstractmethod
    def predict(self, X_test):
        """
        This method makes predictions on the test data.
        :param X_test: Features of the test dataset.
        :return: Predicted labels for the test data.
        """
        pass  # Abstract method, should be implemented in the child class

    @abstractmethod
    def test(self, X_test, y_test):
        """
        This method evaluates the model on the test data.
        :param X_test: Features of the test dataset.
        :param y_test: True labels of the test dataset.
        :return: The evaluation result (e.g., accuracy, loss, etc.).
        """
        pass  # Abstract method, should be implemented in the child class
