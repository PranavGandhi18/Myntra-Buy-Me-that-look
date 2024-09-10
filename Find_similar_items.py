#Here we will copy paste all the classes and methods we wrote in UsingAllTheTrainedModelsTogether.ipynb file. 
#We will do this because we can import these classes and methods from this file to the streamlit_app.py

import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm
import copy
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import streamlit as st

class PreprocessImage:
    def __init__(self):
        pass

    def preprocess_image_forGender(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it, and
        resize it to the target shape for Gender Classification.
        """
        TARGET_SHAPE = (256,256,3)
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, TARGET_SHAPE[:2])
        return tf.expand_dims(image, axis=0)
        


    def preprocess_image_forEmbeddings(self,image_np: np.ndarray):

        TARGET_SHAPE = (224, 224, 3)

        # Convert numpy array to JPEG-encoded bytes
        _, buffer = cv2.imencode('.jpg', image_np)
        image_data = buffer.tobytes()  # Convert the encoded image to bytes

        # Now preprocess the image 
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE[:2])

        return tf.expand_dims(image, axis=0)



#Now lets write a class that can do gender classification for us using our trained model

class GenderClassification:
    def __init__(self, image, loaded_models):
        """
        Initializes the GenderClassification class.
        
        :param image: Preprocessed image for gender classification.
        :param loaded_models: Dictionary containing loaded models with "gender_classifier" as a key.
        """
        self.image = image
        self.model = loaded_models.get("gender_classifier")

    def predict_gender(self):
        """
        Predicts the gender (men or female) from the image using the gender classification model.
        
        :return: String indicating the predicted gender: "men" or "female".
        """
        if self.model is None:
            raise ValueError("The 'gender_classifier' model is not found in the loaded_models dictionary.")
        
        predictions = self.model.predict(self.image)
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

        return "male" if predicted_class == 0 else "female"
    

class ExtractObjects:
    def __init__(self):
        pass
    
    def load_model(self, weights_path):
        model = YOLO(weights_path)
        return model

    def get_predictions(self, model, image_path):
        image = cv2.imread(image_path)
        results = model.predict(image, imgsz=512)
        return results

    def generate_bbox(self, image_path, results, model):
        outputs = {}
        detected_objects = {}
        image = cv2.imread(image_path)


        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls = int(classes[i])
                class_name = model.names[cls]

                if class_name not in detected_objects or detected_objects[class_name][0] < conf:
                    detected_objects[class_name] = (conf, (x1, y1, x2, y2))

        for class_name, (conf, (x1, y1, x2, y2)) in detected_objects.items():
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            object_slice = image[y1:y2, x1:x2]
            outputs[class_name] = object_slice

        outputs["original_image"] = image
        return outputs

    def extract_objects(self, weights_path, image_path):
        model = self.load_model(weights_path)
        results = self.get_predictions(model, image_path)
        outputs = self.generate_bbox(image_path, results, model)
        return outputs


#Now lets write class that will find the similar products based on our extracted objects image slices

class FindSimilarObjects:
    def __init__(self):
        pass
    
    def generate_embeddings(self, output_dict, Loaded_Models):
        """
        Generates embeddings for each detected object in the output_dict using the corresponding models in Loaded_Models.
        
        Args:
        - output_dict: Dictionary with class names as keys and preprocessed image slices as values.
        - Loaded_Models: Dictionary with model names as keys and model instances as values.

        Returns:
        - embeddings_dict: Dictionary with class names as keys and embeddings as values.
        """
        embeddings_dict = {}

        for class_name, image_slice in output_dict.items():
            if class_name != "original_image":  # Skip the original image key
                class_name = class_name.lower()  # Convert key to lowercase
                model_key = class_name
                model = Loaded_Models.get(model_key)

                if model is not None:
                    # Assume the model takes the image slice directly and outputs embeddings
                    
                    embedding = model(image_slice)
                    embeddings_dict[class_name] = embedding[0].numpy().astype(np.float32).tolist()  # Convert to numpy array if needed

        return embeddings_dict
    
    def find_similar_embeddings(self,embeddings_dict,detected_gender,LOADED_CSVS):
        
        # Initialize an empty dictionary where we will store dataframe consisting of top 8 similar images url & product url for each class_name
        similar_items_urls = {}
        # Enable tqdm for pandas. This is to show percentage of completion while generating the embeddings. Due to this only we are able to use progress_apply in below code in find_similar_embeddings function
        tqdm.pandas()

        for class_name,embedding_of_object in embeddings_dict.items():
            if detected_gender == "male":

                if class_name=="topwear":
                    df = LOADED_CSVS["mens_topwear"]
                elif class_name=="bottomwear":
                    df = LOADED_CSVS["mens_bottomwear"]
                elif class_name=="eyewear":
                    df = LOADED_CSVS["mens_eyewear"]
                elif class_name=="footwear":
                    df = LOADED_CSVS["mens_footwear"]
                elif class_name=="handbag":
                    df = LOADED_CSVS["handbags"]
                    
            else:
                if class_name=="topwear":
                    df = LOADED_CSVS["womens_topwear"]
                elif class_name=="bottomwear":
                    df = LOADED_CSVS["womens_bottomwear"]
                elif class_name=="eyewear":
                    df = LOADED_CSVS["womens_eyewear"]
                elif class_name=="footwear":
                    df = LOADED_CSVS["womens_footwear"]
                elif class_name=="handbag":
                    df = LOADED_CSVS["handbags"]

            #Now lets calculate the distance of our embedding of extracted object with the embeddings of our scrapped data
            df['distance'] = df['embedding'].progress_apply(lambda x: np.linalg.norm(np.asarray(eval(x), dtype=np.float32) - np.asarray(embedding_of_object, dtype=np.float32)))

            # Sort the DataFrame by the 'distance' column in ascending order
            df_sorted = df.sort_values(by='distance', ascending=True).reset_index(drop=True)

            # Create a new DataFrame with the first 10 rows and specific columns: 'image_url' and 'product_url'
            top_10_similar_df = df_sorted[['image_url', 'product_url','distance']].head(10)
        
            #store this top_10_similar_df as a value for class_name as a key in similar_items_urls dictionary
            similar_items_urls[class_name] = top_10_similar_df
           

        return similar_items_urls
    

#Now lets write the main class that will use all the classes we wrote above to execute the full process
class FIND_SIMILAR_PRODUCTS:
    def __init__(self, img_path, weights_path,LOADED_MODELS,LOADED_CSVS):
        self.img_path = img_path
        self.weights_path = weights_path
        self.LOADED_MODELS = LOADED_MODELS
        self.LOADED_CSVS = LOADED_CSVS
        self.preprocess = PreprocessImage()
        self.find_similar_objects = FindSimilarObjects()


    def predict_gender(self):
        # Preprocess the image and predict gender
        preprocessed_img = self.preprocess.preprocess_image_forGender(self.img_path)
        gender_classifier = GenderClassification(preprocessed_img, self.LOADED_MODELS)
        return gender_classifier.predict_gender()

    def extract_objects(self):
        # Extract objects using YOLO model
        extract_obj_inst = ExtractObjects()
        return extract_obj_inst.extract_objects(self.weights_path, self.img_path)

    def preprocess_all_extracted_objects(self, output_dict):
        """
        Preprocess all image slices in the dictionary except for the 'original_image' key.
        """
        output_dict_copy = copy.deepcopy(output_dict)  # We are creating a deep copy here because we will later need these unprocessed image slices from output_dict to display the extracted objects
        for class_name, image_slice in output_dict_copy.items():
            if class_name != "original_image":
                output_dict_copy[class_name] = self.preprocess.preprocess_image_forEmbeddings(image_slice)
        return output_dict_copy

    def generate_embeddings(self, preprocessed_outputs):
        # Generate embeddings for the extracted objects
        return self.find_similar_objects.generate_embeddings(preprocessed_outputs, self.LOADED_MODELS)


    def find_similar_embeddings(self, embeddings_extractedobjects, predicted_gender):
        # Find similar objects based on the embeddings and gender
        return self.find_similar_objects.find_similar_embeddings(embeddings_extractedobjects, predicted_gender, self.LOADED_CSVS)

    def display_image(self, image, title="Image"):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=title, width=200)



    def show_similar_items(self, output, similar_items_urls):
        """Display the original image, detected objects, and recommended similar products."""

        # Set up how many images per row you want
        images_per_row = 4

        for class_name, img_slice in output.items():
            if class_name == "original_image":
                pass
            else:
                self.display_image(img_slice, title=f"Detected {class_name}")
                class_name = class_name.lower()  # Convert key to lowercase
                image_product_urls = similar_items_urls[class_name]

                # Now display the images in rows with 4 images per row
                num_images = image_product_urls.shape[0]
                for i in range(0, num_images, images_per_row):
                    cols = st.columns(images_per_row)  # Create columns for the row
                    for j, row in enumerate(image_product_urls.iloc[i:i + images_per_row].iterrows()):
                        idx, row_data = row
                        image_url = row_data["image_url"]
                        product_url = row_data["product_url"]

                        with cols[j]:
                            # Display clickable images using HTML
                            st.markdown(f'<a href="{product_url}" target="_blank"><img src="{image_url}" style="width:100%;"></a>', unsafe_allow_html=True)

    def run(self):

        # Step 1: Predict gender
        predicted_gender = self.predict_gender()
        print(f"Detected Gender: {predicted_gender}")

        # Step 2: Extract objects from the image using YOLO model
        outputs = self.extract_objects()

        # Step 3: Preprocess extracted objects for embedding generation
        preprocessed_outputs = self.preprocess_all_extracted_objects(outputs)

        # Step 4: Generate embeddings for the extracted objects
        embeddings_extractedobjects = self.generate_embeddings(preprocessed_outputs)


        # Step 5: Find similar objects based on embeddings and gender
        similar_objects_urls = self.find_similar_embeddings(embeddings_extractedobjects, predicted_gender)

        # Step 6: Display the results
        self.show_similar_items(outputs, similar_objects_urls)




