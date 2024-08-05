from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import base64
import cv2
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('best_model.keras')
import pickle
import logging 
# logging.basicConfig(level=logging.DEBUG)
import os
# # Load your trained model
# model_path = 'model2.pkl'
# if os.path.exists(model_path):
#     logging.debug(f"Model file {model_path} found.")
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     logging.debug("Model loaded successfully.")
# else:
#     logging.error(f"Model file {model_path} not found.")
#     model = None

# with open('model2.pkl', 'rb') as file:
#     model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

def process_transparent_image(image_data):
    # Step 1: Decode base64 if necessary
    if isinstance(image_data, str):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data

    # print(f"Step 1: Length of image bytes: {len(image_bytes)}")

    # Step 2: Open image with PIL
    image = Image.open(io.BytesIO(image_bytes))
    # print(f"Step 2: Opened image size: {image.size}, mode: {image.mode}")

    # Step 3: Create a white background
    white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))

    # Step 4: Paste the image on the white background
    white_background.paste(image, (0, 0), image)
    # white_background.save('debug_white_background.png')
    # print(f"Step 4: Image with white background saved as 'debug_white_background.png'")

    # Step 5: Convert to RGB
    image_rgb = white_background.convert('RGB')
    # image_rgb.save('debug_rgb.png')
    # print(f"Step 5: RGB image saved as 'debug_rgb.png'")

    # Step 6: Convert to grayscale
    gray_image = image_rgb.convert('L')
    # gray_image.save('debug_grayscale.png')
    # print(f"Step 6: Grayscale image saved as 'debug_grayscale.png'")

    # Step 7: Resize
    resized_image = gray_image.resize((28, 28), Image.LANCZOS)
    # resized_image.save('debug_resized.png')
    # print(f"Step 7: Resized image saved as 'debug_resized.png'")

    # Step 8: Convert to numpy array
    image_array = np.array(resized_image)
    # print(f"Step 8: Numpy array shape: {image_array.shape}, dtype: {image_array.dtype}")
    # print(f"Array min: {image_array.min()}, max: {image_array.max()}, mean: {image_array.mean()}")

    # Step 9: Normalize
    normalized_array = image_array.astype('float32') / 255.0
    # print(f"Step 9: Normalized array min: {normalized_array.min()}, max: {normalized_array.max()}, mean: {normalized_array.mean()}")
    # plt.imsave('debug_normalized.png', normalized_array, cmap='gray')
    # print("Normalized image saved as 'debug_normalized.png'")
    # Step 10: Reshape for model
    # print("shape of norm array is", normalized_array.shape)
    normalized_array = np.expand_dims(normalized_array, axis=-1)

    # Add the batch dimension to make it (1, 28, 28, 1)
    final_image = np.expand_dims(normalized_array, axis=0)
    # final_image = normalized_array.reshape(1, 28, 28, 1)
    # print(f"Step 10: Final image shape: {final_image.shape}")

    return final_image


# In your predict route:
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("data received is",data)
    image_data = data['image']
    processed_image = process_transparent_image(image_data)
    print("processed img is : ", processed_image)
    if model is None:
        print("ew")
        return jsonify({'error': 'Model not loaded'}), 500
    prediction = model.predict(processed_image)
    print("prediction is",prediction)
    classes = ['banana', 'basketball', 'ladder']
    predicted_class = np.argmax(prediction, axis=1)[0]
    probabilities = (prediction[0] * 100).round(2).tolist()  # Convert to percentages and round to 2 decimals
    return jsonify({'prediction': classes[predicted_class], 'probabilities': probabilities})


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
port = os.getenv('PORT')

if __name__ == '__main__':
    app.run(host=os.getenv('HOST'),port=port)
