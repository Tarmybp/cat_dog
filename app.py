import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import joblib

# Load model & encoder
clf = joblib.load("cat_dog_clf.joblib")
le = joblib.load("label_encoder.joblib")
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    x = image.img_to_array(img.resize((224,224)))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    emb = resnet.predict(x)
    pred = clf.predict(emb)
    label = le.inverse_transform(pred)[0]
    
    st.write(f"Prediction: **{label}**")
