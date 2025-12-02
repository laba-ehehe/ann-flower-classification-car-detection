# EEL4930 Project 3 - Artificial Neural Networks: Flower Classification and Car Detection

An implementation of artificial neural networks for multi-class flower species classification and car object detection using TensorFlow.

## Table of Contents
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Flower Classification](#flower-classification)
  - [Car Detection](#car-detection)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## About The Project

This project implements neural networks for two distinct computer vision tasks:

1. **Flower Species Classification**: A multi-class classification model trained to identify 10 different flower species from images.
2. **Car Detection**: An object detection model that predicts bounding box coordinates for cars in satellite imagery.

The project explores various neural network architectures, implements best training practices including early stopping and adaptive learning rates, and evaluates model performance using appropriate metrics for each task.

### Built With
* Python 3.x
* TensorFlow
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

## Getting Started

### Dependencies

* tensorflow==2.13.0
* numpy==1.23.0
* pandas==1.5.0
* matplotlib==3.6.1
* opencv-python==4.8.0
* scikit-learn==1.2.2
* pillow==9.3.0
* jupyter==1.0.0

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/UF-MLforAISystems-Fall24/project-3-undergraduate-laba-ehehe.git
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebooks:
   ```bash
   jupyter notebook "training.ipynb"
   jupyter notebook "test.ipynb"
   ```

4. Download the required datasets and trained models:
   Due to file size constraints, the datasets and trained models are hosted on Google Drive. Please download the required files from the link below:

   [Download Models and Dataset](https://drive.google.com/drive/folders/1hH19cVmmtd9_XD_ZS2BVlyAhRfPDrUss?usp=sharing)

   Ensure the downloaded files are placed in the appropriate directories:
   - Model `car_detector.keras` and `flower_classifier.keras`: root directory (`project-3-undergraduate-laba-ehehe`)  
   - Flower Classification Dataset: `flower_species_classification/`
   - Car Detection Dataset: `car_detection_dataset/`

## Dataset Structure

#### Flower Species Dataset
- Located in `flower_species_classification/`
- Training data: `data_train.npy` (1678 images, 300x300x3)
- Training labels: `labels_train.npy`
- Test data: `data_test.npy`
- Test labels: `labels_test.npy`
- 10 flower species classes:
  - Roses (0)
  - Magnolias (1)
  - Lilies (2)
  - Sunflowers (3)
  - Orchids (4)
  - Marigold (5)
  - Hibiscus (6)
  - Firebush (7)
  - Pentas (8)
  - Bougainvillea (9)

#### Car Detection Dataset
- Located in `car_detection_dataset/`
- Training images: `training_images/`
- Training annotations: `train_bounding_boxes.csv`
- Test images: `testing_images/`
- Annotation format: [xmin, ymin, xmax, ymax]

## Usage

### Flower Classification

1. Training:
   - Open `training.ipynb`
   - Configure hyperparameters in the training section
   - Run all cells to train the model
   - Model checkpoints saved as `flower_classifier.keras`

2. Testing:
   - Open `test.ipynb`
   - Load trained model and test data
   - Run evaluation cells to get performance metrics

### Car Detection

1. Training:
   - Open `training.ipynb`
   - Configure detection model parameters
   - Run training cells to train the model
   - Model saved as `car_detector.keras`

2. Testing:
   - Open `test.ipynb`
   - Load model and test images
   - Run cells to visualize bounding box predictions

## Model Architectures

### Flower Classification Model
```python
model = Sequential([
    Input(shape=(300, 300, 3)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

### Car Detection Model
```python
model = Sequential([
    Input(shape=(380, 676, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='linear')
])
```

## Results

### Flower Classification
- Training accuracy: 12.5%
- Validation accuracy: 12.0%
- Test accuracy: 11.1%

### Car Detection
- Final training loss: 9796.68
- Final validation loss: 18260.56
- Final MAE: 55.32 (training), 87.78 (validation)

Training histories and detailed performance metrics are available in the test notebook.

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

Lan Anh Do - [LinkedIn](https://www.linkedin.com/in/lananhnguyendo/)

Project Link: [GitHub](https://github.com/laba-ehehe/ann-flower-classification-car-detection)

## Acknowledgements

* [Dr. Catia Silva](https://faculty.eng.ufl.edu/catia-silva/) - Instructor of EEL4930 - Applied Machine Learning System (Fall 2024)
* TensorFlow documentation and tutorials
* Deep learning resources and research papers
