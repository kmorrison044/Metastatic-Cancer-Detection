# MetastaticCancerDetection
In this project, I create a CNN model to identify metastatic cancer in CT scans using PyTorch. This project was a submission in a Kaggle competition found here:  https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview. Feel free to contribute to the project to obtain better results!

## Project Overview
The goal of this project is to develop a machine learning algorithm that can accurately identify metastatic cancer from pathology scans. We will evaluate the performance of the model using the area under the ROC curve (AUC), a metric that measures how well the model distinguishes between cancerous and non-cancerous images. The larger the AUC, the better the model performs.

This project includes the following steps:

- Preprocessing and transforming pathology image data.
- Training a convolutional neural network (CNN) to classify images as cancerous or non-cancerous.
- Evaluating the model using AUC to assess performance.
## Dataset
We use a dataset of pathology images, with each image labeled as either cancerous (1) or non-cancerous (0). The dataset consists of 220,025 images, each 96x96 pixels in size.

### Data Description
- Labels: A CSV file that contains the IDs of the images and their corresponding labels (1 for cancer, 0 for no cancer).
- Images: Two sets of pathology scans (train and test), stored in .tif format.
## Model
We use a Convolutional Neural Network (CNN) built using PyTorch. The key steps involved are:

- Preprocessing the images: resizing, normalizing, and converting them to tensors.
- Defining the dataset and data loader for efficient data handling.
- Training the model with an optimization loop, including backpropagation and gradient descent.
- Evaluating the model's performance using the ROC AUC score.
### Model Architecture
The model uses transfer learning with a pre-trained CNN from PyTorch’s model library, which is fine-tuned on our dataset.

### Hyperparameters
- Learning Rate: 0.0025
- Epochs: 10
- Batch Size: 32

## Exploratory Data Analysis
Before training the model, the dataset is explored to understand its distribution. Here’s a visualization of the number of cancerous vs. non-cancerous images in the dataset:


## Results
The final trained model is evaluated based on the AUC score, which measures its ability to distinguish between cancerous and non-cancerous pathology images.

## Conclusion
This project demonstrates how deep learning can be applied to pathology scans for cancer detection. The trained CNN achieves a satisfactory AUC score, proving its potential for assisting in medical diagnoses.

## License
This project is licensed under the MIT License.
