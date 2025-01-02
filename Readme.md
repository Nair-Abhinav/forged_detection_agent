# Signature Forgery Detection

This repository contains code for detecting forged signatures using deep learning. The code utilizes Convolutional Neural Networks (CNNs) implemented in Python with TensorFlow and Keras.

## Features
- **Binary classification**: Detects whether a given signature is genuine or forged.
- **Model architecture**: Utilizes CNNs for feature extraction and classification.
- **Custom dataset loading**: Handles loading and preprocessing of signature image datasets.
- **Evaluation metrics**: Accuracy, precision, recall, and F1-score.

---

## Installation

### Prerequisites
- Python 3.8 or above
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV (optional for advanced image preprocessing)

Install the dependencies using pip:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python
```

---

## Dataset

The model expects a dataset of signature images organized into two folders:
- `genuine/`: Contains images of genuine signatures.
- `forged/`: Contains images of forged signatures.

The directory structure should look like this:
```
/dataset
    /genuine
        signature1.png
        signature2.png
        ...
    /forged
        signature1.png
        signature2.png
        ...
```

Place your dataset in a folder named `dataset` in the root directory of this project.

---

## Code Explanation

### Importing Libraries
The script starts by importing necessary libraries:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
```
- **NumPy**: Used for numerical operations.
- **Matplotlib**: For visualizing data and results.
- **Scikit-learn**: For splitting the dataset into training and testing sets.
- **Keras**: To build and train the CNN model.

### Data Preprocessing

#### Loading Images
The images from the `genuine` and `forged` folders are loaded and labeled as 0 (genuine) and 1 (forged):
```python
# Load data
# Assume you have a function `load_images()` to load and preprocess the images
```

#### Data Splitting
The data is split into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Data Augmentation
Image augmentation is applied to increase the diversity of the training set:
```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
```

### Model Architecture

A Convolutional Neural Network (CNN) is built with the following layers:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
- **Conv2D**: Extracts spatial features from images.
- **MaxPooling2D**: Reduces the spatial dimensions to prevent overfitting.
- **Flatten**: Converts 2D feature maps to 1D feature vectors.
- **Dense**: Fully connected layers for classification.

### Model Compilation

The model is compiled with a binary cross-entropy loss function and Adam optimizer:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Model Training
The model is trained on the dataset for 10 epochs:
```python
history = model.fit(train_datagen.flow(X_train, y_train), epochs=10, validation_data=(X_test, y_test))
```

### Evaluation
The trained model is evaluated on the test set:
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

### Predictions
New signature images can be classified as genuine or forged using:
```python
predictions = model.predict(new_images)
```

---

## Results

The following metrics are used to evaluate the model:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of correctly identified genuine signatures.
- **Recall**: Proportion of genuine signatures identified by the model.
- **F1-Score**: Harmonic mean of precision and recall.

---

## Usage

1. Place the dataset in the `dataset` folder.
2. Run the script:
   ```bash
   python signature_detection.py
   ```
3. View the evaluation results and metrics.
4. Use the model to classify new signature images.

---

## License
This project is licensed under the MIT License.

---

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.
