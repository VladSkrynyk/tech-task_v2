# Image Classification with Multiple Models

This repository contains the solution to the Image Classification task using the MNIST dataset. The task is divided into three different models for classifying handwritten digits using the following algorithms:

1. Random Forest
2. Feed-Forward Neural Network (FFNN)
3. Convolutional Neural Network (CNN)

Each of these models is encapsulated in its own class that implements a common interface. This approach allows for flexible switching between algorithms while maintaining a consistent input-output structure. The task also demonstrates the use of Object-Oriented Programming (OOP) principles for better code organization and scalability.

## Task Overview

In this project, the following requirements were implemented:

- **MnistClassifierInterface**: An interface that defines the common methods `train()`, `predict()`, and `test()` for all classification models.
- **Three Model Classes**: 
  - **Random Forest**: Implements a Random Forest model for MNIST classification.
  - **Feed-Forward Neural Network**: Implements a simple feed-forward neural network for the task.
  - **Convolutional Neural Network**: Implements a CNN model, optimized for image classification tasks.
- **MnistClassifier**: A class that takes an algorithm name as input (cnn, rf, nn) and provides predictions using the corresponding model. It hides the implementation details, making it easy to switch between models.

## Folder Structure

The repository is organized as follows:

- **models/**
  - **classifier.py**: General class for working with models, providing an interface for training and predictions.
  - **cnn.py**: Implementation of the Convolutional Neural Network classifier.
  - **feed_forward.py**: Implementation of the Feed-Forward Neural Network classifier.
  - **mnist_classifier_interface.py**: Interface definition that includes `train()`, `predict()`, and `test()` methods for all models.
  - **random_forest.py**: Implementation of the Random Forest classifier.
  
- **notebooks/**
  - **demo.ipynb**: Jupyter notebook demonstrating the working of the solution with examples and edge cases.

- **test_data/**
  - Contains PNG images for testing the models.

- **requirements.txt**: List of required Python libraries.

- **README.md**: This file.


## Installation

To run this project locally, clone the repository and install the dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/VladSkrynyk/tech-task_v2.git
   cd task1_mnistclassifier

## How it Works

### MnistClassifierInterface

The `MnistClassifierInterface` serves as a blueprint for all classification models. It defines three key methods that all models must implement:

- `train()`: Trains the model on the MNIST dataset.
- `predict()`: Makes predictions on new, unseen data.
- `test()`: Evaluates the performance of the model on a test dataset to estimate its accuracy.

Although the `train()` and `predict()` methods were required for the task, the `test()` method was added to provide additional evaluation functionality. This allows for assessing how well each model performs after training, providing insight into its generalization capabilities.

This interface ensures that each model adheres to the same structure, making it easy to switch between different classification algorithms without changing the overall workflow.

### Model Classes

The three model classes implement the `MnistClassifierInterface` and encapsulate the logic for each classification algorithm:

1. **Random Forest**:
   - This model leverages the Random Forest algorithm from `scikit-learn` to classify digits in the MNIST dataset. Random Forest is a powerful ensemble method that uses multiple decision trees to improve classification accuracy.

2. **Feed-Forward Neural Network (FFNN)**:
   - This model is built using Keras and TensorFlow. It features a simple fully connected neural network consisting of input, hidden, and output layers, with activation functions to introduce non-linearity and allow for complex decision boundaries.

3. **Convolutional Neural Network (CNN)**:
   - The CNN is designed for image classification tasks, utilizing convolutional layers to automatically extract features from the MNIST images. It is more effective than traditional models, like Random Forest, for image recognition due to its ability to learn spatial hierarchies.

### MnistClassifier

The `MnistClassifier` class acts as a wrapper for the three model classes. It simplifies the use of the models by allowing the user to specify the algorithm they wish to use (either `cnn`, `rf`, or `nn`) and provides a unified interface for training, prediction, and testing. The main goal of this class is to abstract away the implementation details of each model, providing a consistent and easy-to-use interface.

- **Training**: The `MnistClassifier` calls the `train()` method of the selected model to train it on the MNIST dataset.
- **Prediction**: The `predict()` method of `MnistClassifier` is used to generate predictions from the trained model.
- **Testing**: The `test()` method of `MnistClassifier` evaluates the performance of the trained model on a test dataset. It helps to estimate the model's accuracy and generalization ability. Although this method wasn't required in the task, it was added to provide additional insight into the quality of the models' predictions after training.

### Demo

The demo is provided in the `demo.ipynb` Jupyter notebook, which demonstrates the following:

- **Training**: How to train the three models (Random Forest, FFNN, CNN) using the MNIST dataset.
- **Prediction**: How to make predictions using each of the models.

This notebook showcases how the `MnistClassifier` can seamlessly switch between models while maintaining the same input-output structure.

## Evaluation

### Confusion Matrix

The primary evaluation metric used for assessing the model's performance is the **confusion matrix**. The confusion matrix is a valuable tool for understanding how well the model is performing, especially in classification problems.

#### Why is the Confusion Matrix Important?

The confusion matrix provides a detailed breakdown of the model's predictions:

- **True positives (TP)**: Correctly predicted positive instances.
- **True negatives (TN)**: Correctly predicted negative instances.
- **False positives (FP)**: Instances incorrectly predicted as positive.
- **False negatives (FN)**: Instances incorrectly predicted as negative.

By examining the confusion matrix, we can understand the number of misclassifications, and specifically, how the model is performing across different classes. This allows us to identify any class imbalances or biases that might be affecting the model's performance.

#### Why are Our Results Good?

Our results are considered good when the confusion matrix has a **diagonal** appearance. This indicates that the model is correctly predicting the majority of instances for each class. A diagonal matrix suggests that the number of false positives and false negatives is low, meaning that the model is reliably distinguishing between classes.

In particular, a high number of true positives along the diagonal of the matrix and low false positives and false negatives suggest that the model is well-calibrated and making accurate predictions across all categories. This is a strong indicator that the model is performing well and is suitable for the task at hand.

### Conclusion

By focusing on the confusion matrix, we gain a comprehensive understanding of the model's performance. A matrix with a clear diagonal pattern demonstrates that the model is reliably classifying the data, making it a good fit for the task. This makes the confusion matrix an essential evaluation tool in our project.

## Main Results

The following table presents the accuracy of each model on the test dataset. The accuracy is calculated based on the number of correct predictions divided by the total number of test samples. This metric helps in comparing the performance of the different models on the same dataset.

| **Model**               | **Accuracy on Test Data** |
|-------------------------|---------------------------|
| **Feed-Forward Neural Network (FFNN)** | `0.9734`       |
| **Convolutional Neural Network (CNN)** | `0.9858`       |
| **Random Forest**       | `0.9696`       |

### Interpretation of Results

- **Feed-Forward Neural Network (FFNN)**: A traditional neural network that works well for simple classification tasks.
- **Convolutional Neural Network (CNN)**: A deep learning model designed to work with image data. CNNs are expected to perform better than FFNNs on image-related tasks due to their ability to capture spatial hierarchies.
- **Random Forest**: An ensemble learning method that aggregates the predictions of multiple decision trees. Random Forests can perform well even when the data is noisy or not linearly separable.

The results can be interpreted in the context of the model's architecture and its suitability for the task. For example, CNNs are likely to outperform FFNNs on image datasets due to their specialized convolutional layers that capture spatial features, while Random Forests may offer competitive results, especially in cases with more structured or tabular data.

## Best Performing Model: Convolutional Neural Network (CNN)

Among all the models tested, the **Convolutional Neural Network (CNN)** achieved the best performance on the test dataset. This result is expected due to the inherent strengths of CNNs in processing image data.

### Why CNN Performed the Best

1. **Feature Extraction**: CNNs are specifically designed to automatically learn and extract spatial features from images using convolutional layers. These layers allow the model to identify patterns such as edges, textures, and shapes, which are critical for classifying images, particularly handwritten digits in the MNIST dataset.

2. **Local Receptive Fields**: By using filters (kernels) that operate over local regions of the image, CNNs are able to detect low-level features (like edges and corners) and combine them into more complex high-level features (like curves and shapes). This is particularly useful for digit recognition, where different digits share similar visual patterns.

3. **Pooling Layers**: MaxPooling layers help in reducing the spatial dimensions of the image while retaining the most important information. This not only helps in reducing the computational load but also makes the model more invariant to small translations and distortions, which is common in handwritten digits.

4. **Hierarchical Learning**: CNNs build hierarchical representations of the data. In the case of MNIST, the lower layers might learn to detect simple shapes, while higher layers capture more abstract features like the curves and strokes of digits. This ability to learn a hierarchy of features allows CNNs to generalize better and classify images more accurately than traditional models.

5. **Overfitting Prevention**: CNNs tend to generalize better due to their structure, especially when combined with techniques like pooling, dropout, and data augmentation. This helps avoid overfitting, a common issue in models with high capacity (like Feed-Forward Neural Networks).

### Conclusion

In summary, the CNN model excelled due to its specialized architecture for image data. While Feed-Forward Neural Networks and Random Forests can perform well in many scenarios, CNNs are superior when dealing with tasks like image classification where spatial hierarchies and feature extraction are key. This makes CNN the most effective model for recognizing and classifying handwritten digits in the MNIST dataset.

## Additional Section

An additional project related to neural networks is available on my GitHub:

[Neural Network without Libraries](https://github.com/VladSkrynyk/NeuralNetworkForHandwrittenDigitRecognition)

This project demonstrates the implementation of a simple neural network with both forward and backward propagation, built entirely from scratch without the use of any external libraries. This approach highlights the fundamental principles of how neural networks operate.



