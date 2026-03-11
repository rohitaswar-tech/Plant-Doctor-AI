import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def build_model(num_classes):
    """
    Builds a plant disease detection model using Transfer Learning with MobileNetV2.
    """
    # Load base model with pre-trained ImageNet weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Build the architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(data_dir, epochs=10):
    """
    Data preprocessing pipeline and training.
    """
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = build_model(num_classes=train_generator.num_classes)
    
    # Training
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the model
    model.save('plant_disease_model.h5')
    print("Model saved as plant_disease_model.h5")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('plant_disease_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted to plant_disease_model.tflite")

    return model, train_generator.class_indices

def predict_disease(image_path, model, class_indices):
    """
    Predicts the disease from an image path.
    """
    # Load and preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    
    # Map index to class name
    inv_map = {v: k for k, v in class_indices.items()}
    disease_name = inv_map[class_idx]

    return {
        "Disease Name": disease_name,
        "Confidence Score": f"{score:.2%}",
        "Prevention/Treatment": "Placeholder: Consult local agricultural guidelines for specific treatment."
    }

if __name__ == "__main__":
    # Example usage (requires a dataset directory)
    # model, indices = train_model('path/to/dataset')
    # result = predict_disease('test_leaf.jpg', model, indices)
    # print(result)
    pass
