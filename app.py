import os
from flask import Flask, request, render_template, send_file
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the generator model
generator = tf.keras.models.load_model(r'D:\Study Metarials\Fgan\gan64\models\generator_saved_model')

# Function to fix grayscale images using the generator
def fix_image(grayscale_image):
    if grayscale_image is None:
        return None

    # Ensure the grayscale image is of the correct size (256x256) and data type (float32)
    grayscale_image = cv2.resize(grayscale_image, (256, 256))
    grayscale_image = grayscale_image.astype('float32') / 255.0

    # Expand the dimensions to match the input shape of the generator
    grayscale_image = np.expand_dims(grayscale_image, axis=0)
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)  # Add an extra channel dimension

    # Generate the colorized version using the generator model
    colorized_image = generator.predict(grayscale_image)

    # Convert the output to a valid image format (e.g., uint8)
    colorized_image = (colorized_image[0] * 255).astype('uint8')

    return colorized_image

@app.route('/', methods=['GET', 'POST'])
def upload_and_colorize():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user submits an empty file
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # If the file is valid
        if file:
            # Save the uploaded file
            file_path = 'static/uploaded_image.jpg'  # Save the image in the "static" folder
            file.save(file_path)

            # Load the grayscale image and colorize it
            grayscale_image = cv2.imread(file_path, -1)
            colorized_image = fix_image(grayscale_image)

            if colorized_image is None:
                return render_template('index.html', message='Failed to process the uploaded image')

            # Save the colorized image
            colorized_path = 'static/restored_image.jpg'
            cv2.imwrite(colorized_path, colorized_image)

            return render_template('restore.html', image_path=file_path, restored_image_path=colorized_path)

    return render_template('index.html', message='Upload your image to pixel restoration')

if __name__ == '__main__':
    app.run(debug=True)
