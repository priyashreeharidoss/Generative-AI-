# Deep Learning-Based Age and Gender Prediction from Facial Images

## Overview
This project aims to develop machine learning models for predicting the age and gender of individuals based on their facial images. Leveraging deep learning techniques, the models are trained on annotated datasets to accurately classify facial images into different age groups and gender categories.

## Features
- Predicts age and gender from facial images
- Utilizes convolutional neural networks (CNNs) for feature extraction
- Supports preprocessing tasks such as resizing and normalization
- Evaluates model performance using various metrics
- Includes sample code for training and testing the prediction models

## Requirements
- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- pandas
- numpy
- OpenCV

## Dataset
The project utilizes a dataset of facial images annotated with age and gender labels. Follow these steps to download the dataset:
1. Visit the Kaggle dataset page: [Age, Gender, and Ethnicity Face Data CSV](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv).
2. Log in to your Kaggle account. If you don't have one, sign up for a free account.
3. Once logged in, click the "Download" button on the dataset page.
4. Save the downloaded dataset file to your local machine.

## Usage
To use the code provided in this project, follow these steps:
1. Ensure you have Python installed on your system along with the required libraries.
2. Download the dataset from the provided Kaggle link and place it in the specified directory ("/content/drive/MyDrive/age_gender.csv").
3. Open the Python script containing the code and run it in your preferred Python environment (e.g., Jupyter Notebook, Google Colab).
4. The script will preprocess the data, train the age and gender prediction models, and evaluate their performance.
5. After training, the models will be saved as "age_predictor.h5" and "gen_predictor.h5" for future use.

## Results
The results of the age and gender prediction models can be observed in the following ways:
1. **Training Progress**: The script will display the training progress, including training and validation accuracy, loss, and other metrics, using matplotlib visualizations.
2. **Model Evaluation**: The trained models will be evaluated using validation data, and the evaluation results will be displayed, including accuracy and other relevant metrics.
3. **Prediction**: The models will make predictions on a sample of unseen data, displaying the predicted age group and gender alongside the actual labels for comparison.
4. **Visualizations**: Visualizations of predicted facial images alongside their actual labels can be viewed to assess the performance of the models visually.

By following these steps, you can utilize the code provided to train, evaluate, and make predictions using the age and gender prediction models developed in this project.
