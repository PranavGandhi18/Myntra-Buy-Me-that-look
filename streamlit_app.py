import streamlit as st
import copy
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from io import BytesIO
import cv2
from PIL import Image
# Import your classes and methods here
# from your_code import FIND_SIMILAR_PRODUCTS, PreprocessImage, GenderClassification, ExtractObjects, FindSimilarObjects, load_CSVs, load_models
from Find_similar_items import FIND_SIMILAR_PRODUCTS, PreprocessImage, GenderClassification, ExtractObjects, FindSimilarObjects


#Write functions to load our trained models and csv files which contains embeddings of our scrapped data
from tensorflow.keras.models import load_model
import pandas as pd

#Lets initialize the models and csv's storing dictionaries
LOADED_MODELS = dict()
LOADED_CSVS = dict()


@st.cache
def load_models():
    """
    Load Various Models for Recommendation Engine Pipeline
    """
    if len(LOADED_MODELS) == 6:
        return LOADED_MODELS
    else:
        print("Loading Models...")
        GENDER_CLASSIFIER = load_model("./gender_classifier_EfficientNet.h5")
        TOPWEAR_EMBEDDING = load_model("./embedding generating models/embedding_topwear_InceptionV3.h5")
        BOTTOMWEAR_EMBEDDING = load_model("./embedding generating models/embedding_bottomwear_InceptionV3.h5")
        FOOTWEAR_EMBEDDING = load_model("./embedding generating models/embedding_footwear_InceptionV3.h5")
        EYEWEAR_EMBEDDING = load_model("./embedding generating models/embedding_eyewear_InceptionV3.h5")
        HANDBAGS_EMBEDDING = load_model("./embedding generating models/embedding_handbags_InceptionV3.h5")

        LOADED_MODELS["gender_classifier"] = GENDER_CLASSIFIER
        LOADED_MODELS["topwear"] = TOPWEAR_EMBEDDING
        LOADED_MODELS["bottomwear"] = BOTTOMWEAR_EMBEDDING
        LOADED_MODELS["footwear"] = FOOTWEAR_EMBEDDING
        LOADED_MODELS["eyewear"] = EYEWEAR_EMBEDDING
        LOADED_MODELS["handbag"] = HANDBAGS_EMBEDDING
        print("Models Loaded!")
        return LOADED_MODELS


@st.cache
def load_CSVs():
    """
    Load Various CSV files for Embedding Generation Pipeline
    """
    if len(LOADED_CSVS) == 9:
        return LOADED_CSVS
    else:
        print("Loading CSVs...")
        MENS_TOPWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/mens_topwear_embeddings.csv")
        MENS_BOTTOMWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/mens_bottomwear_embeddings.csv")
        MENS_FOOTWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/mens_footwear_embeddings.csv")
        MENS_EYEWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/mens_eyewear_embeddings.csv")
        WOMENS_TOPWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/womens_topwear_embeddings.csv")
        WOMENS_BOTTOMWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/womens_bottomwear_embeddings.csv")
        WOMENS_FOOTWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/womens_footwear_embeddings.csv")
        WOMENS_EYEWEAR_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/womens_eyewear_embeddings.csv")
        HANDBAGS_CSV = pd.read_csv("./downloaded_scrapped_data_embeddings/handbags_embeddings.csv")

        
        LOADED_CSVS["mens_topwear"] = MENS_TOPWEAR_CSV
        LOADED_CSVS["mens_bottomwear"] = MENS_BOTTOMWEAR_CSV
        LOADED_CSVS["mens_footwear"] = MENS_FOOTWEAR_CSV
        LOADED_CSVS["mens_eyewear"] = MENS_EYEWEAR_CSV
        LOADED_CSVS["womens_topwear"] = WOMENS_TOPWEAR_CSV
        LOADED_CSVS["womens_bottomwear"] = WOMENS_BOTTOMWEAR_CSV
        LOADED_CSVS["womens_footwear"] = WOMENS_FOOTWEAR_CSV
        LOADED_CSVS["womens_eyewear"] = WOMENS_EYEWEAR_CSV
        LOADED_CSVS["handbags"] = HANDBAGS_CSV

        print("CSVs Loaded!")
        return LOADED_CSVS






# Streamlit dashboard code
def main():

     # Displaying the main heading for the project
    st.markdown("<h1 style='text-align: center; color: purple;'>Myntra: Buy Me That Look</h1>", unsafe_allow_html=True)
    
    # Subheading and description
    st.subheader("Find Similar Products")
    st.write("Upload an image, and we'll find similar products for you.")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load models and CSVs if they are not already loaded
        if 'LOADED_MODELS' not in st.session_state or 'LOADED_CSVS' not in st.session_state:
            st.write("Loading the Models....")
            st.session_state.LOADED_MODELS = load_models()
            st.write("Models Loaded!")
            st.write("Loading the CSVs....")
            st.session_state.LOADED_CSVS = load_CSVs()
            st.write("Data Embedding CSVs loaded!")

        elif 'LOADED_MODELS' in st.session_state and 'LOADED_CSVS' in st.session_state:
            st.write("Models and Embedding CSV's are already loaded!")
            

        # Run the FIND_SIMILAR_PRODUCTS pipeline
        weights_path = "./runs/detect/yolo_small_train/weights/best.pt"
        finder = FIND_SIMILAR_PRODUCTS("temp_image.png", weights_path,st.session_state.LOADED_MODELS,st.session_state.LOADED_CSVS)
        finder.run()

if __name__ == "__main__":
    main()
