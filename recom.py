import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Page configuration
st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    .title {
        color: #f0f2f6;
        font-weight: bold;
        text-align: center;
    }
    .header {
        background-color: #333333;
        color: white;
        padding: 10px;
        text-align: center;
    }
    .upload-section {
        background-color: #2a2a2a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
    }
    .recommendations {
        padding: 20px;
        margin-top: 20px;
        background-color: #2a2a2a;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4b6cb7;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #3a5aa6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list, num_recommendations=10):
    neighbors = NearestNeighbors(n_neighbors=num_recommendations+1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:num_recommendations+1]  # Exclude the first result which is the input image

# Load filenames and feature_list
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Header
st.markdown('<div class="header"><h1>Fashion Recommender System</h1></div>', unsafe_allow_html=True)
import streamlit as st
from PIL import Image

image = Image.open(r'C:\Users\ANKITHA_REDDY\OneDrive\Pictures\a.jpg')
new_image = image.resize((1000, 450))
st.image(new_image)


st.markdown("## Select a Dress Option to Get Recommendations")

# Dress options
dress_options = {
    "Kurti": r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\56.jpg",
    "Saree": r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\59.jpg",
    "lehanga": r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\197.jpg",
    "T-shirt": r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\83.jpg",
    "Jeans": r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\47.jpg",
    "kids wear":r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\28.jpg",
    "trousers":r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\69.jpg",
    "Dresses":r"C:\Users\ANKITHA_REDDY\OneDrive\Pictures\15.jpg"
}

# Dropdown menu for dress selection
selected_dress = st.selectbox("Choose a dress type:", list(dress_options.keys()))

# File uploader for custom image upload
uploaded_file = st.file_uploader("Or upload your own image")

# Display the selected dress image or uploaded image
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        img_path = os.path.join('uploads', uploaded_file.name)
        display_image = Image.open(img_path)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)
    else:
        st.error("Error in uploading file")
else:
    dress_image_path = os.path.join(os.path.dirname(__file__), dress_options[selected_dress])
    display_image = Image.open(dress_image_path)
    st.image(display_image, caption=f'Selected: {selected_dress}', use_column_width=True)
    img_path = dress_image_path

# Button to get recommendations
if st.button("Recommend Similar Items"):
    with st.spinner('Processing...'):
        features = feature_extraction(img_path, model)
        indices = recommend(features, feature_list)
        st.markdown("### Recommended Items:")
        cols = st.columns(5)

        def load_image(filename):
            return Image.open(filename)

        num_recommendations = 5  # Number of recommendations to display
        for i in range(num_recommendations):
            if i < len(indices):
                with cols[i]:
                    st.image(load_image(filenames[indices[i]]), use_column_width=True)
            else:
                st.write("No more recommendations.")
