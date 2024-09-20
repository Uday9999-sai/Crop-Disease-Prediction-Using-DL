from flask import (Flask, render_template, request,url_for, flash,redirect)
from flask_wtf import FlaskForm
from flask import send_from_directory
import pickle
import numpy as np
import tensorflow_hub as hub
import warnings
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from joblib import dump, load
import os
from flask_login import LoginManager,UserMixin,login_user,login_required,logout_user,current_user
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
model_path = 'mobnet.h5'
model = keras.models.load_model(model_path)

import pickle

with open('treatment_recommendation_model.pkl', 'rb') as f:
    treatment_model = pickle.load(f)
with open('label_encoder_disease.pkl', 'rb') as f:
    le_disease = pickle.load(f)

with open('label_encoder_treatment.pkl', 'rb') as f:
    le_treatment = pickle.load(f)



class_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]
def get_treatment_recommendation(disease_name):
    # Encode the disease name
    disease_encoded = le_disease.transform([disease_name])[0]

    # Reshape for prediction
    disease_encoded = np.array(disease_encoded).reshape(-1, 1)

    # Predict the treatment
    treatment_encoded = treatment_model.predict(disease_encoded)[0]

    # Decode the predicted treatment
    treatment_name = le_treatment.inverse_transform([treatment_encoded])[0]

    # Mock fertilizer image path and buy link (can be set later)
    fertilizer_image_path = None
    buy_link = None

    return treatment_name, fertilizer_image_path, buy_link


# Function to predict image class and get the treatment
def predict_image_class(img_path):
    # Preprocess the image for the image classification model
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict using the image classification model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_name = class_labels[predicted_class]

    # Get the treatment recommendation
    recommendation, fertilizer_image_path, buy_link = get_treatment_recommendation(class_name)

    return class_name, recommendation, fertilizer_image_path, buy_link





app = Flask(__name__)
app.secret_key="4949asdklfjasdklflaksdf"
app.config['UPLOAD_FOLDER'] = 'uploads'



from flask import send_from_directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/detect_brain')
def detect_brain():
    return render_template('detect.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', 'error')
            return render_template('detect.html')

        f = request.files['file']
        filename = secure_filename(f.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            f.save(img_path)
            predicted_disease, recommendation, fertilizer_image_path, buy_link = predict_image_class(img_path)
            return render_template('result.html', disease=predicted_disease, recommendation=recommendation, img_path=img_path, fertilizer_image_path=fertilizer_image_path, buy_link=buy_link)
        except Exception as e:
            print(f"Error during processing: {e}")
            flash('An error occurred. Please try again later.', 'error')

    return render_template('detect.html')



if __name__ == '__main__':
    app.run(debug=True)