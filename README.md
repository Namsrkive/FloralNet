# Flower-Species-Classification (Oxford Flowers Dataset)

## Abstract

This project focuses on classifying different species of flowers using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The dataset used for this project is sourced from Kaggle's [PyTorch Challenge Flower Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/code). The goal of this project is to build an effective model that can classify flower species based on images, offering practical utility for plant identification in agriculture and botany.

The model was trained and tested on the dataset using Keras and TensorFlow. It achieved high accuracy in recognizing different flower species. The project is deployed using Streamlit, providing an interactive web interface for real-time classification of flower images.

## Project Link

- [Kaggle Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/code)
- [Streamlit Web App](https://flower-species-classification-twdmufeepjzmmsyqjzzkzy.streamlit.app/)

## Features

- **Deep Learning Model**: A CNN model trained using TensorFlow and Keras for flower classification.
- **Streamlit Web App**: A deployed interface where users can upload images of flowers and receive the predicted species.
- **Real-time Prediction**: The web app classifies flowers in real time as users interact with it.

## Repository Structure

- `Model/`: Contains the trained model and other related files.
- `app.py`: The main script for the Streamlit app.
- `cat_to_name.json`: A JSON file that maps the flower species to human-readable names.
- `.gitignore`: Specifies files and directories that Git should ignore.
- `requirements.txt`: Contains the list of dependencies required to run the project.

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Namsrkive/Flower-Species-Classification.git
