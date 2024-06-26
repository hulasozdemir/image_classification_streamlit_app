# Image Classification with Convolutional Neural Networks (CNN) using TensorFlow

## Overview

This project demonstrates how to build, train, evaluate, and deploy a Convolutional Neural Network (CNN) model for image classification using TensorFlow. The dataset used for this project is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Project Structure

```
image_classification/
├── app/
│   └── streamlit_app.py
├── data/
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── model_architecture.py
│   ├── test_data_preprocessing.py
│   ├── test_model_architecture.py
│   └── train_model.py
├── best_model.keras
├── README.md
└── requirements.txt
```
## Streamlit App

[Link](https://imageclassificationappapp-3r6eseu4harmqgk2vu8mjo.streamlit.app)

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hulasozdemir/image_classification.git
    cd image_classification
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

### 1. Data Exploration

Use the Jupyter notebook to explore the dataset:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 2. Test Data Preprocessing

Run the following script to ensure data preprocessing works correctly:
```bash
python src/test_data_preprocessing.py
```

### 3. Test Model Architecture

Run the following script to ensure the model architecture is defined correctly:
```bash
python src/test_model_architecture.py
```

### 4. Train the Model

Train the CNN model by running the training script:
```bash
export PYTHONPATH=$(pwd)
python src/train_model.py
```
This will save the best model as `best_model.keras` in the root directory.

### 5. Evaluate the Model

Evaluate the trained model by running the evaluation script:
```bash
export PYTHONPATH=$(pwd)
python src/evaluate_model.py
```

### 6. Deploy the Model with Streamlit

Run the Streamlit app to deploy the model:
```bash
streamlit run app/streamlit_app.py
```

This will launch a web interface where you can upload images and get classification predictions from the trained model.

## Project Components

- **data_preprocessing.py**: Contains functions to load and preprocess the CIFAR-10 dataset.
- **model_architecture.py**: Defines the architecture of the CNN model.
- **train_model.py**: Script to train the CNN model with the CIFAR-10 dataset.
- **evaluate_model.py**: Script to evaluate the trained model and generate evaluation metrics.
- **streamlit_app.py**: Streamlit application to deploy the trained model and classify uploaded images.

## Requirements

- tensorflow
- streamlit
- matplotlib
- seaborn
- Pillow
- scikit-learn

Install all the requirements using:
```bash
pip install -r requirements.txt
```

## Acknowledgements

- This project uses the CIFAR-10 dataset provided by the [UoT](https://www.cs.toronto.edu/~kriz/cifar.html).
- TensorFlow and Keras libraries are used for building and training the neural network model.
