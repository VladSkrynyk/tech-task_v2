# Importing the RandomForestClassifier from sklearn and the MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier
from .mnist_classifier_interface import MnistClassifierInterface


class RandomForestMnist(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        """
        Initialize the Random Forest model.
        :param n_estimators: Number of trees in the forest (default is 100).
        """
        # Initialize the RandomForestClassifier with the specified number of estimators
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        :param X_train: Training data features.
        :param y_train: Training data labels.
        """
        print("Training Random Forest model...")

        # Flatten the training data if it is 3D (such as images) to make it compatible with the model
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        # Fit the model to the training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict labels for the test data.
        :param X_test: Test data features.
        :return: Predicted labels.
        """
        print("Making predictions using Random Forest model...")

        # Flatten the test data if it is 3D (such as images) to make it compatible with the model
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Return the predicted labels for the test data
        return self.model.predict(X_test)

    def test(self, X_test, y_test):
        """
        Test the performance of the Random Forest model.
        :param X_test: Test data features.
        :param y_test: True labels for the test data.
        """
        print("Evaluating Random Forest model on the test data...")

        # Flatten the test data if it is 3D (such as images) to make it compatible with the model
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Calculate the accuracy of the model on the test data
        accuracy = self.model.score(X_test, y_test)

        # Print the test accuracy
        print(f"Test accuracy: {accuracy:.4f}")
