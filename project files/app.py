from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('CleanTech.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."
    
    # Save the file
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    result = "Healthy" if prediction[0][0] > 0.5 else "Rotten"

    return f"The waste is: {result}"

if __name__ == '__main__':
    app.run(debug=True)
