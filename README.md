# Real vs. Fake Image Detector

## Project Overview
This project is a deep learning-based application that detects whether an uploaded image is real or fake. It uses a pre-trained ResNet-18 model with transfer learning to classify images, trained on a dataset of real and fake images. The application is deployed as a Streamlit web app.

## Features
- **Deep Learning Model**: Uses ResNet-18 for image classification.
- **Dataset Handling**: Processes images using PyTorch’s DataLoader.
- **Training & Evaluation**: Trains the model and evaluates accuracy, precision, recall, and F1-score.
- **User Interface**: A Streamlit-based web app for easy image uploads and predictions.

## Project Structure
```
├── resnet_model.py       # Defines the ResNet-18 model
├── resnet_trainer.py     # Trains the model and saves it
├── app.py                # Streamlit app for inference
├── model.pth             # Trained model weights (generated after training)
└── requirements.txt      # Dependencies
```
Dataset(url)- https://www.kaggle.com/datasets/itamargr/dfdc-faces-of-the-train-sample

## Setup Instructions
### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the Model (Optional, if model.pth is not available)
```bash
python resnet_trainer.py
```
### 4. Run the Web App
```bash
streamlit run app.py
```

## How It Works
1. The user uploads an image through the web app.
2. The image is preprocessed and passed through the trained ResNet-18 model.
3. The model outputs a prediction (Real or Fake) with a confidence score.
4. The result is displayed on the web app.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- Streamlit
- NumPy
- PIL
- scikit-learn
- matplotlib

## Future Improvements
- Expand dataset for better accuracy.
- Deploy the model on a cloud service.
- Improve UI design.



