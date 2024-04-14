

import keras
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from numpy.linalg import norm 
import streamlit as st 
import os  
from PIL import Image
import pickle


feature_list = pickle.load(open('embeddings.pkl', 'rb'))

filesname = pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False 


model = keras.Sequential([
        model,
        GlobalMaxPooling2D(),
    ])




st.title("Fashion Recommender System")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())

        return 1 
    except:
        return 0 


def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)
    expanded_features =np.expand_dims(img,axis=0) 

    preprocessed_img=preprocess_input(expanded_features)  # Extracting Features
    result = model.predict(preprocessed_img).flatten()  # flattening image
    normalized_result =result / norm(result)

    return normalized_result


def recommend(features,Feature_list):
    neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    distances,indices   =  neighbors.kneighbors([features])
    return indices





uploaded_file = st.file_uploader("Choose an Image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
     
        display_image =Image.open(uploaded_file)
        st.image(display_image)
        features =feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        st.text(features)
        st.header("Here are some more Recommendations Based on your Inputs :")
        indices = recommend(features,feature_list)

        col1, col2, col3, col4, col5 = st.columns(5)

    # Set the desired width for the images
        image_width = 200  # Adjust this value as needed

        with col1:
            st.image(filesname[indices[0][0]], width=image_width)
        with col2:
            st.image(filesname[indices[0][1]], width=image_width)
        with col3:
            st.image(filesname[indices[0][2]], width=image_width)
        with col4:
            st.image(filesname[indices[0][3]], width=image_width)
        with col5:
            st.image(filesname[indices[0][4]], width=image_width)





    else:
        st.header("Some Error occured !")
 



