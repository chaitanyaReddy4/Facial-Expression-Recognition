# Facial Expression Recognition Using CNN

A deep learning-based facial expression classification system built using **Convolutional Neural Networks (CNN)** and the **FER-2013 dataset**. The application detects a facial region from an uploaded image and predicts one of seven facial expression categories.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Face%20Detection-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Dataset](https://img.shields.io/badge/Dataset-FER--2013-purple)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Expression Categories](#expression-categories)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [CNN Model Architecture](#cnn-model-architecture)
- [Model Training and Improvements](#model-training-and-improvements)
- [Performance Results](#performance-results)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Overview

Facial Expression Recognition (FER) is a computer vision task that identifies visible facial expression patterns from images.

This project implements a **seven-class facial expression recognition system** using a Convolutional Neural Network trained on the FER-2013 dataset.

The model processes 48 × 48 grayscale facial images and predicts one of the following categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

A Streamlit-based web application provides an interface for uploading facial images and viewing the predicted expression along with class-wise prediction scores.

The project covers the complete workflow, including dataset preparation, preprocessing, CNN development, model training, fine-tuning, evaluation, and application integration.

## Features

- Facial image classification into seven categories.
- Face detection using OpenCV Haar Cascade.
- Image preprocessing and resizing to 48 × 48 pixels.
- Grayscale conversion and pixel normalization.
- CNN-based automatic feature extraction.
- Data augmentation during training.
- Class-wise prediction scores.
- Interactive image-upload interface using Streamlit.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.

## Expression Categories

The model supports the following seven classes:

| Class Index | Expression |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

## Dataset

The project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset.

Dataset characteristics:

| Property | Value |
|---|---|
| Dataset | FER-2013 |
| Image dimensions | 48 × 48 pixels |
| Image format | Grayscale |
| Number of classes | 7 |
| Training images | 22,968 |
| Validation images | 5,741 |
| Test images | 7,178 |
| Total images | 35,887 |

### Class Distribution

**Training set**

| Expression | Images |
|---|---:|
| Angry | 3,218 |
| Disgust | 363 |
| Fear | 3,285 |
| Happy | 5,813 |
| Neutral | 3,965 |
| Sad | 3,834 |
| Surprise | 2,490 |
| **Total** | **22,968** |

**Test set**

| Expression | Images |
|---|---:|
| Angry | 958 |
| Disgust | 111 |
| Fear | 1,024 |
| Happy | 1,774 |
| Neutral | 1,233 |
| Sad | 1,247 |
| Surprise | 831 |
| **Total** | **7,178** |

The dataset has an imbalanced class distribution, particularly for the Disgust category. This is considered during model training and result analysis.

Dataset source: [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## System Architecture

The project follows this workflow:

```text
             FER-2013 Dataset
                     |
                     v
             Image Preprocessing
                     |
          -----------------------
          |         |           |
       Grayscale  Resize     Normalize
          |       48x48       /255
          -----------------------
                     |
                     v
              CNN Model Training
                     |
                     v
             Model Evaluation
                     |
                     v
             Fine-Tuning Model
                     |
                     v
              Saved CNN Model
                     |
                     v
             Streamlit Application
                     |
              Upload Image
                     |
                     v
              OpenCV Face Detection
                     |
                     v
             Face Preprocessing
                     |
                     v
              CNN Prediction
                     |
                     v
          Predicted Expression
          + Class-wise Scores
```

## CNN Model Architecture

The final model uses three convolutional feature-extraction blocks followed by Global Average Pooling and fully connected classification layers.

| Layer | Configuration |
|---|---|
| Input | 48 × 48 × 1 |
| Convolution Block 1 | 32 filters + Batch Normalization + ReLU |
| Max Pooling | 2 × 2 |
| Dropout | 0.20 |
| Convolution Block 2 | 64 filters + Batch Normalization + ReLU |
| Max Pooling | 2 × 2 |
| Dropout | 0.25 |
| Convolution Block 3 | 128 filters + Batch Normalization + ReLU |
| Max Pooling | 2 × 2 |
| Dropout | 0.30 |
| Global Average Pooling | Feature aggregation |
| Dense Layer | 128 neurons, ReLU |
| Dropout | 0.50 |
| Output Layer | 7 neurons, Softmax |

### Model Parameters

| Parameter | Value |
|---|---:|
| Total parameters | 110,759 |
| Trainable parameters | 110,311 |
| Non-trainable parameters | 448 |
| Input shape | 48 × 48 × 1 |
| Output classes | 7 |

### Why These Layers?

- **Convolution:** Extracts important spatial features from facial images.
- **Batch Normalization:** Helps stabilize training.
- **ReLU:** Introduces non-linearity.
- **Max Pooling:** Reduces spatial dimensions.
- **Dropout:** Helps reduce overfitting.
- **Global Average Pooling:** Produces a compact feature representation.
- **Softmax:** Produces class-wise prediction scores.

## Model Training and Improvements

Three major experiments were performed during development.

### Experiment 1: Initial CNN

The initial model used a larger convolutional architecture with repeated convolution layers and a flattening-based classification head.

| Metric | Result |
|---|---:|
| Total parameters | 878,631 |
| Best validation accuracy | 36.09% |
| Test accuracy | 36.26% |

The model showed weak class separation, with predictions concentrated in a limited number of categories.

### Experiment 2: Compact CNN

A redesigned architecture was introduced using:

- Three convolutional blocks.
- Batch normalization.
- Dropout regularization.
- Global Average Pooling.
- Square-root-based class weighting.

Training configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.0003 |
| Loss function | Sparse Categorical Cross-Entropy |
| Maximum epochs | 40 |
| Batch size | 64 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Early stopping | Enabled |

Results:

| Metric | Result |
|---|---:|
| Best validation accuracy | 45.55% |
| Best validation loss | 1.4365 |
| Test accuracy | 46.95% |

### Experiment 3: Fine-Tuning

The project's own saved CNN checkpoint was fine-tuned without using another researcher's pretrained weights.

Major changes:

- Square-root-based class weights normalized to mean 1.
- Cosine learning-rate decay.
- Initial learning rate of 0.0001.
- Label smoothing of 0.05.
- Early stopping.
- Best-model checkpoint saving.

Training configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning-rate schedule | Cosine Decay |
| Initial learning rate | 0.0001 |
| Loss function | Categorical Cross-Entropy |
| Label smoothing | 0.05 |
| Maximum fine-tuning epochs | 12 |
| Best checkpoint epoch | 8 |

Results:

| Metric | Result |
|---|---:|
| Best validation accuracy | 46.44% |
| Validation loss | 1.4770 |
| Test accuracy | **47.35%** |

## Performance Results

The final fine-tuned model was evaluated on 7,178 test images.

### Overall Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **47.35%** |
| Macro Precision | 0.38 |
| Macro Recall | 0.39 |
| Macro F1-score | 0.38 |
| Weighted Precision | 0.44 |
| Weighted Recall | 0.47 |
| Weighted F1-score | 0.45 |

### Class-wise Performance

| Expression | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Angry | 0.32 | 0.27 | 0.29 |
| Disgust | 0.00 | 0.00 | 0.00 |
| Fear | 0.27 | 0.13 | 0.18 |
| Happy | 0.61 | 0.76 | 0.68 |
| Neutral | 0.41 | 0.49 | 0.45 |
| Sad | 0.38 | 0.41 | 0.39 |
| Surprise | 0.64 | 0.66 | 0.65 |

### Experiment Comparison

| Model | Test Accuracy |
|---|---:|
| Initial CNN | 36.26% |
| Compact CNN | 46.95% |
| Fine-tuned CNN | **47.35%** |

The fine-tuned model improved test accuracy by 11.09 percentage points compared with the initial CNN.

The fine-tuning improvement over the compact baseline was 0.40 percentage points.

## Confusion Matrix Analysis

The confusion matrix helps identify which expressions are classified correctly and which expressions are confused with other categories.

Important observations:

- Happy has a recall of 0.76.
- Surprise has an F1-score of 0.65.
- Neutral and Sad show moderate classification performance.
- Fear has low recall of 0.13.
- Disgust has zero correctly classified examples in the reported test evaluation.

These results indicate that the model learns some expression patterns more effectively than others.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN implementation |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Model evaluation |
| OpenCV | Face detection and image processing |
| Streamlit | Web application |

## Project Structure

The following represents the main logical organization of the project. Adapt the names if your local folders differ.

```text
Facial_Expression_Recognition/
│
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
├── models/
│   ├── best_fer_final.keras
│   └── best_fer_improved.keras
│
├── app.py
├── README.md
└── requirements.txt
```

Training notebooks, evaluation outputs and generated plots may be stored in separate folders if included in the repository.

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Facial_Expression_Recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not yet available, the main dependencies are:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python streamlit pillow
```

Use compatible package versions for the Python and TensorFlow environment used by the project.

## Running the Application

After activating the virtual environment and installing the dependencies, run:

```bash
streamlit run app.py
```

The Streamlit application will start locally. Open the local URL displayed in your terminal.

Upload a facial image to view the detected facial region, predicted expression and class-wise scores.

## How It Works

1. The user uploads a facial image.
2. OpenCV detects the facial region.
3. The detected face is cropped.
4. The image is converted to grayscale.
5. The face is resized to 48 × 48 pixels.
6. Pixel values are normalized to the [0, 1] range.
7. The processed image is passed to the trained CNN.
8. The model generates scores for all seven categories.
9. The category with the highest score is displayed as the predicted expression.

## Limitations

- The overall test accuracy is moderate at 47.35%.
- Performance varies significantly across expression categories.
- Disgust is particularly difficult because it is under-represented in the training data.
- Fear has low recall.
- Lighting, pose, occlusion, face detection and image quality can affect predictions.
- The model is trained on FER-2013 and may not generalize equally well to all real-world photographs.
- Facial expression classification does not establish a person's actual emotional state.

## Future Improvements

- Improve minority-class representation.
- Experiment with transfer learning architectures such as MobileNet and ResNet.
- Investigate attention-based CNN architectures.
- Improve face alignment and preprocessing.
- Evaluate performance on additional independent datasets.
- Explore real-time webcam-based expression classification.
- Improve model calibration and robustness.

## References

1. Goodfellow, I. J., et al. (2013). *Challenges in Representation Learning: A Report on Three Machine Learning Contests.*
2. FER-2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
3. TensorFlow Documentation: https://www.tensorflow.org/
4. Keras Documentation: https://keras.io/
5. OpenCV Documentation: https://docs.opencv.org/
6. Streamlit Documentation: https://docs.streamlit.io/
7. Scikit-learn Documentation: https://scikit-learn.org/

---

## Author

Developed as an individual academic capstone project.

**Project:** Facial Expression Recognition Using CNN  
**Domain:** Deep Learning / Computer Vision  
**Dataset:** FER-2013

---

**Note:** This project is intended for educational and experimental purposes. Predictions represent model classifications of visible facial patterns and should not be treated as psychological assessments or definitive emotion detection.
