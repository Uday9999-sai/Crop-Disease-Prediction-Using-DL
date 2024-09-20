import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('plant_disease_model1.h5')

# Class labels
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Function to preprocess and predict the image
def predict_image(image_path):
    # Load the image
    img = load_img(image_path, target_size=(128, 128))  # Use the target size you trained your model with
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match input shape for the model
    img_array = img_array / 255.0  # Normalize the image to match the training phase

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the index of the highest probability
    confidence_score = np.max(predictions)  # Get the confidence score

    # Output the result
    print(f"Predicted class: {class_labels[predicted_class]}")
    print(f"Confidence score: {confidence_score:.2f}")

    return class_labels[predicted_class], confidence_score

# Example usage
image_path = 'test/test/CornCommonRust1.JPG'  # Replace with the path to the image you want to predict
predicted_class, confidence_score = predict_image(image_path)

print(f"The image is classified as: {predicted_class} with a confidence score of {confidence_score:.2f}")
