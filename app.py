import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
import os
import io
import base64

# Set up the page config with a custom icon
st.set_page_config(page_title="Med Plant Snap", page_icon="leaf.png", layout="centered")

# Ensure your model and CSV files are in the correct directory
model_path = 'MED_PLANT_SNAP(16.08.2024).h5'
csv_path = 'details.csv'
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the CSV file with plant information
df = pd.read_csv(csv_path)

class_names = ["Arive-Dantu", "Basale", "Betel", "Crape_Jasmine", "Curry", "Drumstick", "Fenugreek", "Guava", "Hibiscus", "Indian_Beech",
               "Indian_Mustard", "Jackfruit", "Jamaica_Cherry-Gasagase", "Jamun", "Jasmine", "Karanda", "Lemon",
               "Mango", "Mexican_Mint", "Mint", "Neem", "Oleander", "Parijata", "Peepal", "Pomegranate", "Rasna", "Rose_apple",
               "Roxburgh_fig", "Sandalwood", "Tulsi"]

# Function to preprocess the image
def preprocess_image(image):
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# Function to get plant information from the CSV
def get_plant_info(class_name):
    row = df[df['Class'] == class_name].iloc[0]
    return row['Scientific Name'], row['Disease Treated'], row['Preparation Method'], row['Administration']

# Function to convert image to base64
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Streamlit app
st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');

    .title {
        text-align: center;
        color: #2ECC71;  /* Green color */
        font-size: 36px;
        font-weight: bold;
    }
    .title i {
        margin-right: 10px;
        color: #2ECC71; /* Icon color */
    }
    .upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    .upload-container .bold-label {
        font-weight: bold;
        color: #2C3E50; /* Optional: Change color if needed */
        font-size: 18px; /* Adjust size if needed */
        margin-bottom: 10px; /* Space between label and uploader */
    }
    .info-container {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: #23252F;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: #ECF0F1;
        text-align: center; /* Center the text */
    }
    .sidebar {
        background-color: #2C3E50;
        color: #ECF0F1;
    }
    .sidebar h1 {
        color: #ECF0F1;
    }
    .sidebar .social-icon {
        color: #ECF0F1;
        font-size: 24px;
        margin-right: 10px;
    }
    .sidebar a {
        text-decoration: none; /* Remove underline */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title"><i class="fas fa-leaf"></i> MED PLANT SNAP</h1>', unsafe_allow_html=True)
st.write('Many of us don’t know about the medical plants and their uses in curing a disease. Upload a snap of a plant leaf to know about the plant and their medical uses.')

# File uploader with warning message
st.markdown('<p style="color: red; font-weight: bold;">Warning: Please upload a clear image of a plant leaf. Uploading other types of images may result in incorrect predictions, showing the highest probability class.</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your leaf snap...", type="jpg")

if uploaded_file is not None:
    # Check file size
    max_size_mb = 10
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = uploaded_file.size

    if file_size > max_size_bytes:
        st.error(f"File size exceeds {max_size_mb} MB limit. Please upload a smaller file.")
    else:
        # Load and display the image with a fixed size
        image = Image.open(uploaded_file)  # Adjust width as needed

        with st.spinner('Processing...'):
            try:
                # Preprocess the image
                preprocessed_image = preprocess_image(image)
                
                # Predict using the model
                prediction = model.predict(preprocessed_image)
                pred_label = np.argmax(prediction, axis=1)
                class_prediction = class_names[pred_label[0]]

                # Get plant information
                scientific_name, disease_treated, preparation_method, administration = get_plant_info(class_prediction)
                
                # Convert the uploaded image to base64
                image_base64 = image_to_base64(image)
                
                st.markdown(f"""
                    <div class="info-container">
                        <h3>Plant Information</h3>
                        <img src="data:image/png;base64,{image_base64}" width="300" /><br></br>
                        <p><strong>Predicted Class:</strong> {class_prediction}</p>
                        <p><strong>Scientific Name:</strong> {scientific_name}</p>
                        <p><strong>Disease Treated:</strong> {disease_treated}</p>
                        <p><strong>Preparation Method:</strong> {preparation_method}</p>
                        <p><strong>Administration:</strong> {administration}</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing the image: {e}")

# Sidebar content
st.sidebar.title("About")
st.sidebar.info("Many don't have the knowledge about the medicinal leaves. So, this project aims to identify the medicinal leaf of your snap and displays about their characteristic and uses.")

# Add GitHub and LinkedIn links to the sidebar
st.sidebar.markdown("""
    <h3>To explore projects like this visit:</h3>
<a href="https://github.com/Bilal1729" target="_blank" style="color: #ECF0F1; text-decoration: none;">
    <i class="fab fa-github social-icon"></i> GitHub
</a><br><br>
<h3>Feel free to get in touch:</h3>
<a href="https://www.linkedin.com/in/bilal1729/" target="_blank" style="color: #ECF0F1; text-decoration: none;">
    <i class="fab fa-linkedin social-icon"></i> LinkedIn
</a>
""", unsafe_allow_html=True)
