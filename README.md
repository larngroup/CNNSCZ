# CNNSCZ
# CNN Models in Prediction of Feno Type from Genotype
This study explores the use of deep learning to analyze genetic data and predict phenotypic traits associated with schizophrenia, a complex psychiatric disorder with a strong hereditary component yet incomplete genetic characterization. We applied Convolutional Neural Networks models to a large-scale case-control exome sequencing dataset from the Swedish population to identify genetic patterns linked to schizophrenia. To enhance model performance and reduce overfitting, we employed advanced optimization techniques, including dropout layers, learning rate scheduling, batch normalization, and early stopping. Following systematic refinements in data preprocessing, model architecture, and hyperparameter tuning, the final model achieved an accuracy of 80 %. These results demonstrate the potential of deep learning approaches to uncover intricate genotype-phenotype relationships and support their future integration into precision medicine and genetic diagnostics for psychiatric disorders such as schizophrenia.


# Model Architecture:
![Model Architecture New](https://github.com/user-attachments/assets/7c8e8f7a-8336-4a02-b8e9-93cf48c0a74b)


# How To Run:
To successfully run this code on your own computer, it's important to ensure that all required Python libraries are properly installed and compatible with your system. The core libraries include:
  - **pandas**, used for data manipulation and reading CSV files
  - **numpy**, provides numerical operations and array handling
  - **h5py**, necessary for reading .h5 files that contain large-scale data (genotype dataset)
  - **matplotlib.pyplot**, visualizing training performance

The machine learning functionality relies heavily on **TensorFlow** and **Keras**. Specifically, you need to install a version of tensorflow that includes the integrated Keras API (**tensorflow>=2.10** works well). This script makes use of both high level Sequential and functional Keras APIs to define the CNN model. 

Layers, such as the following, are all part of the **tensorflow.keras.layers** module:
  - Conv1D
  - MaxPooling1D
  - Dense
  - BatchNormalization
  - Dropout
  - Flatten

Callbacks, such as the following, are all part of **tensorflow.keras.callbacks** module and are used to monitor training and control learning rate and overfitting: 
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau

Ensure that your environment has a compatible version of Python (ideally **Python >=3.8**) and that there are no conflicts between TensorFlow and other installed packages. Using a virtual environment is highly recommended to isolate dependencies and avoid version mismatches. Once all libraries are installed, and your environment is set up correctly, you should be able to execute the script.
