import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from flask import Flask, request, render_template, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Categories for healthy vs rotten classification
WASTE_CATEGORIES = ['healthy', 'rotten']

class WasteClassificationModel:
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.num_classes = len(WASTE_CATEGORIES)
        
    def create_model(self):
        """Create VGG16 transfer learning model"""
        # Load pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', 
                          include_top=False, 
                          input_shape=self.input_shape)
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        return self.model
    
    def prepare_data_generators(self):
        """Create data generators for training"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        return train_datagen, val_datagen
    
    def create_synthetic_data(self):
        """Create synthetic healthy vs rotten data for demonstration"""
        print("Creating synthetic healthy vs rotten dataset...")
        
        # Create directories
        data_dir = 'waste_dataset'
        for category in WASTE_CATEGORIES:
            os.makedirs(f'{data_dir}/{category}', exist_ok=True)
        
        # Generate synthetic images (colored patterns representing different categories)
        np.random.seed(42)
        for i, category in enumerate(WASTE_CATEGORIES):
            for j in range(100):  # 100 images per category
                # Create synthetic image with category-specific patterns
                if category == 'healthy':
                    img = np.random.normal(0.4, 0.1, (224, 224, 3))  # Fresh green tones
                    img[:, :, 1] *= 1.8  # More green
                    img[:, :, 0] *= 0.7  # Less red
                else:  # rotten
                    img = np.random.normal(0.3, 0.1, (224, 224, 3))  # Brown/dark tones
                    img[:, :, 0] *= 1.3  # More red
                    img[:, :, 1] *= 0.6  # Less green
                
                img = np.clip(img, 0, 1) * 255
                img = img.astype(np.uint8)
                
                # Save image
                image = Image.fromarray(img)
                image.save(f'{data_dir}/{category}/item_{j}.jpg')
        
        print("Synthetic dataset created successfully!")
        return data_dir
    
    def train_model(self, data_dir):
        """Train the model"""
        if not os.path.exists(data_dir):
            data_dir = self.create_synthetic_data()
        
        train_datagen, val_datagen = self.prepare_data_generators()
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            verbose=1
        )
        
        # Save model
        self.model.save('healthy_vs_rotten.h5')
        
        # Save training history
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        print("Model training completed and saved!")
        return history
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists('healthy_vs_rotten.h5'):
            self.model = tf.keras.models.load_model('healthy_vs_rotten.h5')
            print("Model loaded successfully!")
        else:
            print("No trained model found. Creating and training new model...")
            self.create_model()
            self.train_model('waste_dataset')
    
    def predict_waste_type(self, image_path):
        """Predict waste type from image"""
        if self.model is None:
            self.load_model()
        
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return WASTE_CATEGORIES[predicted_class], confidence, predictions[0]

# Initialize model
waste_model = WasteClassificationModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/blog-single')
def blog_single():
    return render_template('blog-single.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save uploaded file
        upload_path = 'temp_upload.jpg'
        file.save(upload_path)
        
        # Make prediction
        predicted_class, confidence, all_predictions = waste_model.predict_waste_type(upload_path)
        
        # Clean up
        os.remove(upload_path)
        
        # Prepare response
        predictions_dict = {
            WASTE_CATEGORIES[i]: float(all_predictions[i]) 
            for i in range(len(WASTE_CATEGORIES))
        }
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': predictions_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        waste_model.create_model()
        history = waste_model.train_model('waste_dataset')
        return jsonify({'message': 'Model training completed successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create model on startup
    print("Initializing CleanTech Waste Management System...")
    waste_model.load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)