import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt

# Load the VGG19 model and set up the custom layers
base_model = VGG19(include_top=False, input_shape=(128,128,3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4068, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('model_2_unfrozen.keras')

st.title("Pneumonia Detection App")
st.write("Upload an X-ray image to predict if it is Normal or Pneumonia.")

def get_class_name(class_no):
    if class_no == 0:
        return "Normal"
    elif class_no == 1:
        return "Pneumonia"

def get_result(img):
    image = Image.open(img).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0  # Scale as in the notebook
    input_img = np.expand_dims(image_array, axis=0)

    # Predict
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)[0]
    return result01

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Run prediction and get result
    prediction = get_result(uploaded_file)
    class_name = get_class_name(prediction)

    # Display result
    st.write(f"Prediction: **{class_name}**")


