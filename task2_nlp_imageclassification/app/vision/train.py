import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model # app.vision.


def train_and_evaluate(data_dir, output_model_path, batch_size=32, epochs=10):
    # Data augmentation and normalization
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Training data generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    # Validation data generator
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    # Splitting validation set into validation and test set (20% from validation)
    X_val, X_test, y_val, y_test = train_test_split(
        val_generator.filepaths,
        val_generator.classes,
        test_size=0.5,
        random_state=42)

    # Model Initialization
    model = build_model(num_classes=len(train_generator.class_indices))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model Training
    print("Training the model...")
    model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    # Save the trained model
    model.save(output_model_path)
    print(f"✅ Model trained and saved to {output_model_path}")

    # Evaluate on Test Set
    print("Evaluating on Test Set...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': X_test, 'class': y_test}),
        directory=None,
        x_col='filename',
        y_col='class',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}")

# Example Usage:
data_dir = "../../mnt/data/raw-img/"
output_model_path = "../mnt/models/vision_model.h5"
train_and_evaluate(data_dir, output_model_path, batch_size=32, epochs=10)
