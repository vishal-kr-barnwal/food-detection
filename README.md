# Food Detection and Classification using CNN

This project implements a food detection and classification system using Convolutional Neural Networks (CNN) on the Food_101 dataset with bounding box annotations.

## Overview

The project consists of the following components:
1. Data import and exploration
2. Mapping training and testing images to their classes
3. Creating and processing annotations for training and testing images
4. Displaying images with bounding boxes
5. Designing, training, and testing CNN models for food classification

## Dataset

The project uses a subset of the Food_101 dataset, which contains images of 17 food categories:
- apple_pie
- chocolate_cake
- donuts
- falafel
- french_fries
- hot_dog
- ice_cream
- nachos
- onion_rings
- pancakes
- pizza
- ravioli
- samosa
- spring_rolls
- strawberry_shortcake
- tacos
- waffles

The dataset includes XML annotations for bounding boxes around food items in the images.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/CapstoneFoodDetection.git
cd CapstoneFoodDetection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Open and run the Jupyter notebook:
```
jupyter notebook FoodDetection.ipynb
```

2. Follow the steps in the notebook to:
   - Import the Food_101 dataset and annotations
   - Create dataframes for images and bounding box annotations
   - Split the data into training and testing sets (70:30 ratio)
   - Display images with their bounding boxes
   - Design, train, and test CNN models for food classification

## Model Architecture

The CNN model uses transfer learning with MobileNetV2 with fine-tuning:
- Pre-trained MobileNetV2 base model (with ImageNet weights)
- Fine-tuning of the top 30 layers of MobileNetV2
- Global Average Pooling layer
- Batch Normalization layer
- Dense layer with 256 units and ReLU activation
- Dropout layer (0.4) for regularization
- Dense layer with 128 units and ReLU activation
- Dropout layer (0.3) for regularization
- Final dense layer with softmax activation for classification

## Results

The model achieves over 80% accuracy on the test set. The notebook includes:
- Images with their bounding box annotations
- Distribution of food classes in the dataset
- Training and validation accuracy/loss curves
- Per-class accuracy analysis to identify strengths and weaknesses
- Predictions on test images with bounding boxes

The improved performance is achieved through:
- Data augmentation (random flips, rotations, zoom, contrast, and brightness adjustments)
- Fine-tuning of the top layers of MobileNetV2
- Enhanced model architecture with additional layers
- Learning rate scheduling and early stopping
- Extended training with up to 30 epochs

## Future Improvements

Potential enhancements include:
- Experimenting with more advanced architectures (EfficientNetV2, ConvNeXt, etc.)
- Implementing more sophisticated data augmentation techniques like MixUp or CutMix
- Using semi-supervised learning to leverage unlabeled data
- Implementing object detection (YOLO, SSD, Faster R-CNN) to locate and classify food items simultaneously
- Exploring ensemble methods to combine predictions from multiple models
- Adding more food categories and annotations
- Implementing cross-validation for more robust evaluation
- Using techniques like knowledge distillation to create smaller, faster models
