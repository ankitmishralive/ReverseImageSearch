
#---------------------- Feature Extraction -------------------------
import keras 
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D 
from keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm 
import numpy as np
import os 
# from PIL import Image
from tqdm import tqdm 

import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False 


model = keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])

# print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)
    expanded_features =np.expand_dims(img,axis=0) 
    #image ka dimension badhaya jata hai taaki usse model ke liye pass kiya ja sake. 
    preprocessed_img=preprocess_input(expanded_features)  # Extracting Features
    result = model.predict(preprocessed_img).flatten()  # flattening image
    #function is used to convert the multi-dimensional 
    # array of features into a one-dimensional array.

    normalized_result =result / norm(result)
    # scaling down the values to bring them within a similar range.
    # E.g data can be 1 also & 99999999 also their is a huge difference so
    #  scaling it down in specific range
    #While logarithmic scaling can be useful in certain contexts, 
    # such as dealing with extremely large values or creating features 
    # that exhibit multiplicative relationships, 
    # in this case, where the goal is to normalize the feature vectors for similarity calculations, using norm for scaling down to a 
    # specific range is more appropriate a
    # nd aligns with common practices in machine learning.

    return normalized_result



filesname = []


for file in os.listdir('images'):
    filesname.append(os.path.join('images', file))

# print(len(filesname))
# print(filesname[0:5])

feature_list = []
for file in tqdm(filesname):
    feature_list.append(extract_features(file,model))

# print(np.array(feature_list))


pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filesname,open('filenames.pkl','wb'))
    