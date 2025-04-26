import sqlite3
from sqlite3 import OperationalError
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from leprosynet import create_model

app = Flask(__name__)

# Database setup
DATABASE = 'users.db'

# Load the model
model = create_model()
# Compile the model (needed for loading)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
try:
    model.load_weights('model_weights.h5')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT
            )
        ''')
        db.commit()

def add_user(username, password):
    with get_db() as db:
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        db.commit()
def check_user(username, password):
    with get_db() as db:
        cursor = db.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        return cursor.fetchone() is not None

init_db()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    if check_user(username, password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data['username']
    password = data['password']
    try:
        add_user(username, password)
        return jsonify({'message': 'User registered successfully'}), 201
    except ValueError as e:
        return jsonify({'message': str(e)}), 400
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Username already exists'}), 400
    except OperationalError:
        return jsonify({'message': 'Database is locked'}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        try:
            # Preprocess the image
            img = image.load_img(file, target_size=(76, 102))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize pixel values

            # Make prediction
            prediction = model.predict(img_array)
            
            # Convert prediction to a simple format
            prediction_result = float(prediction[0][0])

            return jsonify({'message': 'File successfully processed', 'prediction': prediction_result}), 200
        except Exception as e:
            return jsonify({'message': f'Error processing file: {str(e)}'}), 500
    return jsonify({'message': 'File upload error'}), 500