# models.py

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from PIL import Image

# Path to the model file
modelSavePath = "lung_colon_model1.h5"
model_1 = load_model(modelSavePath)



def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def pred_and_plot(filename, img_shape=224):
    class_names = ['colon_adenocarcinoma', 'colon_benign_tissue',
        'lung_adenocarcinoma', 'lung_benign_tissue',
        'lung_squamous_cell_carcinoma']

    # Import the image and preprocess it
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.

    pred = model_1.predict(tf.expand_dims(img, axis=0))

    # Convert to percentages and round to two decimal places
    pred_percentages = np.round(pred * 100, 2)

    # Get the predicted class (the class with the highest probability)
    pred_class = class_names[pred_percentages.argmax()]

    return pred_class, pred_percentages.tolist(), class_names


model2_save_path = "brain-lung-breast.h5"
model_2 = load_model(model2_save_path)


def predict_cancer_for_ctmri(filename, img_shape=224):
    threshold = 0.9

    class_names = [
        'brain_glioma', 
        'brain_meningioma', 
        'brain_notumor', 
        'brain_pituitary',
        'breast_cancer', 
        'breast_non_cancer', 
        'lung_benign', 
        'lung_malignant',
        'lung_normal'
    ]
    
    # Import the image and preprocess it
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.

    # Get predictions
    pred = model_2.predict(tf.expand_dims(img, axis=0))

    preds = softmax(pred)

    # Convert to percentages and round to two decimal places
    pred_percentages = np.round(preds * 100, 2).flatten()

    # Get the predicted class (the class with the highest probability)
    pred_class = class_names[pred_percentages.argmax()]

    return pred_class, pred_percentages.tolist(), class_names, pred

# Example usage
# pred_class, _, _ = predict_cancer_for_ctmri("C:/Users/Siddharth/Downloads/download (1).jpeg")
# print(pred_class)
