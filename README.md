
![Uploading load.jpg…]()


# SBT Prediction Model for CPT Data

This project is implemented as a Google Colab notebook, allowing you to easily run and experiment with the SBT Prediction Model in a cloud-based environment. Google Colab provides a convenient interface for running Python code, managing dependencies, and leveraging GPU resources for accelerated model training.

To get started with the SBT Prediction Model on Google Colab:

1. Open the notebook file (`SBT_Prediction_Model.ipynb`) in Google Colab.
2. Follow the instructions provided within the notebook to upload your training and new data files, adjust hyperparameters, train the model, and make predictions.

This repository contains code for training and using a deep learning model to predict Soil Behavior Type (SBT) values based on Cone Penetrometer Test (CPT) data. The model is implemented using TensorFlow and Keras. 

## Introduction

The SBT Prediction Model is designed to learn the relationship between input features derived from CPT data and SBT values, enabling accurate predictions on new data. The model architecture consists of multiple fully connected layers with dropout regularization to prevent overfitting. The training data is normalized using a Min-Max scaler before training the model. The model is trained using the Adam optimizer and mean squared error loss.

**SBT Prediction**

The SBT Prediction Model utilizes a deep learning approach to estimate SBT values directly from CPT data, without the need for explicit calculations of cone resistance (qt) and friction ratio (Rf). By learning the underlying patterns and correlations between the input features extracted from CPT data and SBT values, the model makes direct predictions, simplifying the prediction process and saving computational resources. It enables efficient estimation of SBT values based on readily available input parameters, making it a valuable tool in geotechnical engineering and related fields.

**SBT Prediction Model**

The SBT Prediction Model is a machine learning model that predicts the Soil Behavior Type (SBT) using input features derived from CPT data. The model eliminates the need for calculating cone resistance (qt) and friction ratio (Rf) by directly predicting the SBT value.

**Features**

- Utilizes a neural network-based model for accurate SBT prediction from CPT data.
- Supports customization of model architecture through adjustable hyperparameters.
- Allows training on user-provided CPT data and prediction on new data.

**Dependencies**

The SBT Prediction Model requires the following dependencies:

- numpy
- pandas
- scikit-learn (sklearn)
- tensorflow
- matplotlib
- ipywidgets

## Usage

1. Upload the training data file:
   - The training data should be structured in a TXT file with columns for Depth, Speed, Tip, Friction, Pressure, and SBT.
   - Ensure that the units for Tip, Friction, and Pressure match the expected units.
   - Use the provided notebook interface to upload the training data file.

2. Adjust the model hyperparameters:
   - The model allows customization of the number of layers and neurons through sliders.
   - Optimize these hyperparameters based on your specific dataset.

3. Click the 'Train' button:
   - The model will be trained using the uploaded training data and the selected hyperparameters.
   - The training process includes normalization, neural network creation, compilation, and fitting.

4. Upload the new data file for prediction:
   - The new data file should follow the same structure as the training data file.
   - Use the provided notebook interface to upload the new data file.

5. Click the 'Predict' button:
   - The trained model will make predictions on the new data.
   - The predicted SBT values will be rounded and displayed.
   - A plot comparing the predicted SBT values with the depth will be generated.

**Included Data**

For convenience, we have included two sample data files in this repository:

- sbttrainingdata.txt: This file contains example training data formatted as a TXT file. It can be used to train the SBT Prediction Model.
- sbttestdata.txt: This file contains example data for prediction formatted as a TXT file. It can be used to test the SBT Prediction Model and observe its performance on unseen data.

Feel free to use these files as a reference or to try out the model initially.

**Example: Setting Up and Testing with HogenGoler Sounding Data**

To demonstrate the usage of the SBT prediction model, we provide an example using HogenGoler sounding data. The example assumes that the units for Tip, Friction, and Pressure are as follows:

- Tip: TSF (Ton per Square Foot)
- Friction: TSF (Ton per Square Foot)
- Pressure: PSI (Pound per Square Inch)

Here's how you can set up and test the model using the HogenGoler sounding data:

1. Prepare your HogenGoler sounding data in the TXT file format with the following columns:

Depth Speed Tip Friction Pressure SBT

0.066 2 0.3716 0.21973 -0.09607 5

...


The columns represent the following information:

- Depth: Depth value for each data point.
- Speed: Speed value for each data point.
- Tip: Tip value for each data point.
- Friction: Friction value for each data point.
- Pressure: Pressure value for each data point.
- SBT: SBT (Soil Behavior Type) value for each data point.

Ensure that the columns are separated by either spaces or tabs consistently throughout the file.

2. The new data for prediction should be a TXT file with the same column structure as the training data, but without the SBT column. For example:

Depth Speed Tip Friction Pressure SBT

0.02 2.29 0.37404 0.21034 -0.09698

...

3. Use the provided sliders and buttons to adjust the model hyperparameters and initiate training or prediction.

4. Click on the 'Train' button to train the model using the uploaded HogenGoler sounding data.

5. Upload a new data file for prediction, following the same structure as the training data.

6. Click on the 'Predict' button to make predictions using the trained model on the new data.

7. The predicted SBT values will be displayed, and a plot comparing the predicted SBT values with the depth will be generated.

Feel free to explore and experiment with different configurations and datasets to utilize the SBT Prediction Model effectively.
