# Face Recognition Project

This project aims to create and train face recognition models using the Labeled Faces in the Wild (LFW) dataset. The primary goals include data preprocessing, model training, and performance evaluation. The project involves the use of two different models: a Simple Convolutional Neural Network (CNN) and a Fine-tuned VGG16 model.

## Purpose

The purpose of this project is to explore and compare the performance of different face recognition models on the LFW dataset. By employing various techniques and models, the project seeks to achieve accurate and efficient face recognition capabilities.

## Technologies Used

- Python
- TensorFlow and Keras (for model development)
- OpenCV (for image processing)
- Numpy (for numerical operations)
- Matplotlib (for data visualization)

## Dependencies

1. Install required Python packages:
   pip install tensorflow opencv-python numpy matplotlib

## Data Preprocessing
1. Loaded the LFW dataset containing images of various individuals.
2. Resized images to (224, 224) pixels and normalized pixel values to [0, 1].
3. Split the dataset into training and testing sets.

## Data Visualization and Insights
1. Visualized sample images from the dataset to understand its content.
2. Plotted the distribution of classes to identify class imbalances.
3. Calculated the mean and standard deviation of pixel values to gain insights into data variations.

## Model Training
### Model 1: Simple Convolutional Neural Network (CNN)
1. Created a CNN model with three convolutional layers followed by max pooling.
2. Trained the CNN model with 10 epochs and observed training/validation accuracy and loss curves.
3. Evaluated the model's performance on the test dataset using test accuracy and loss.
4. Visualized the training/validation accuracy and loss curves.

### Model 2: Fine-tuned VGG16 Model
1. Loaded the VGG16 model without top layers and added custom layers for face recognition.
2. Frozen base VGG16 layers and compiled the model with Adam optimizer.
3. Trained the model for 10 epochs and evaluated its performance using test accuracy and loss.
4. Made predictions on the test dataset using the fine-tuned VGG16 model.

## Conclusion
In this project, we successfully conducted data preprocessing, data visualization, and training of two different face recognition models: a simple CNN model and a fine-tuned VGG16 model. After evaluating the models, we found that the simple CNN model achieved higher test accuracy compared to the fine-tuned VGG16 model. This observation suggests that for this specific face recognition task and dataset, the simpler architecture of the CNN model was more effective in capturing relevant features and patterns. The project highlights the significance of model selection and experimentation in achieving optimal results for face recognition tasks.