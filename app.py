import os
import sqlite3
from sqlite3 import OperationalError
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model import CNN

app = Flask(__name__)

# Database setup
DATABASE = 'users.db'

# Load model
model_weights_path = r'C:\Users\hpatil\source\repos\santu5496\leacheddetectionImageProcessing\model_weights.pth'

try:
    model = CNN()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model weights loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# Database functions
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

# Initialize DB
init_db()

# Routes

@app.route('/', methods=['GET'])
def index():
    return render_template('login.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

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

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']
    if check_user(username, password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if model is None:
        return jsonify({'message': 'Model not loaded'}), 500

    try:
        # Preprocess image
        img = Image.open(file).convert('RGB').resize((224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)

        # Debug print statements
        print("Tensor shape:", tensor.shape)

        # Model prediction
        with torch.no_grad():
            prediction = model(tensor)
            print("Raw prediction output:", prediction)

        prediction_result = float(prediction[0][0].item())  # Extracting the scalar value

        # Classify based on the prediction
        if prediction_result >= 0.5:
            prediction_label = 'Leprosy Detected'
        else:
            prediction_label = 'No Leprosy'

        return jsonify({'message': 'File successfully processed', 'prediction': prediction_label, 'confidence': prediction_result}), 200

    except Exception as e:
        print("❌ Upload error:", e)
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
