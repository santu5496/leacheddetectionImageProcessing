import os
import sqlite3
from sqlite3 import OperationalError
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import CNN

app = Flask(__name__)

# Database setup
DATABASE = 'users.db'

# Define path for the model weights using relative path
model_weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')

# Global model variable
model = None

# Function to initialize model
def initialize_model():
    global model
    try:
        # Check if weights file exists and is not empty
        if os.path.exists(model_weights_path) and os.path.getsize(model_weights_path) > 0:
            print(f"Found model weights at {model_weights_path}")
            
            # Initialize model
            model = CNN()
            
            try:
                # Attempt to load weights
                model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
                model.eval()  # Set model to evaluation mode
                print("✅ Model weights loaded successfully.")
                return True
            except EOFError:
                print("❌ Error: Model weights file is corrupted or empty.")
            except Exception as e:
                print(f"❌ Error loading model weights: {str(e)}")
                import traceback
                print(f"Error details:\n{traceback.format_exc()}")
        else:
            print(f"❌ Error: Model weights file not found or empty at {model_weights_path}")
            
        # Create new model if loading failed
        print("⚠️ Initializing model without pre-trained weights.")
        model = CNN()
        model.eval()
        return True
        
    except Exception as e:
        print(f"❌ Critical error initializing model: {str(e)}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False

# Initialize model
model_initialized = initialize_model()

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
        
        import os
import sqlite3
from sqlite3 import OperationalError
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import CNN

app = Flask(__name__)

# Database setup
DATABASE = 'users.db'

# Define path for the model weights using relative path
model_weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')

# Global model variable
model = None

# Function to initialize model
def initialize_model():
    global model
    try:
        if os.path.exists(model_weights_path) and os.path.getsize(model_weights_path) > 0:
            print(f"Found model weights at {model_weights_path}")
            model = CNN()
            try:
                model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
                model.eval()
                print("✅ Model weights loaded successfully.")
                return True
            except EOFError:
                print("❌ Error: Model weights file is corrupted or empty.")
            except Exception as e:
                print(f"❌ Error loading model weights: {str(e)}")
                import traceback
                print(f"Error details:\n{traceback.format_exc()}")
        else:
            print(f"❌ Error: Model weights file not found or empty at {model_weights_path}")

        # Create new model if loading failed
        print("⚠️ Initializing model without pre-trained weights.")
        model = CNN()
        model.eval()
        return True

    except Exception as e:
        print(f"❌ Critical error initializing model: {str(e)}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False

# Initialize model
model_initialized = initialize_model()

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
    if not model_initialized:
        return jsonify({'message': 'Model initialization failed. Service unavailable.'}), 503

    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    try:
        # Open and preprocess the image
        img = Image.open(file).convert('RGB').resize((224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)  # Add batch dimension

        print("Tensor shape:", tensor.shape)

        with torch.no_grad():
            prediction = model(tensor)
            print("Raw prediction output:", prediction)

        prediction_result = float(prediction[0][0].item())

        # Prediction threshold
        threshold = 0.5
        is_leached = prediction_result > threshold

        # Confidence calculation
        confidence = abs(prediction_result - threshold) * 2
        confidence_percentage = round(confidence * 100, 2)

        # Final prediction label
        prediction_label = "Leprosy detected" if is_leached else "Healthy skin (No Leprosy)"

        print(f"Prediction Result: {prediction_result}, Confidence: {confidence_percentage}%, Final Label: {prediction_label}")

        return jsonify({
            'message': 'File successfully processed',
            'prediction': prediction_label,
            'confidence_percentage': confidence_percentage,
            'raw_prediction': round(prediction_result, 4)
        }), 200

    except Exception as e:
        print("❌ Upload error:", str(e))
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'ok' if model_initialized else 'degraded',
        'model_initialized': model_initialized,
        'model_weights_loaded': os.path.exists(model_weights_path) and os.path.getsize(model_weights_path) > 0
    }
    return jsonify(status)

@app.route('/train', methods=['POST'])
def train_model():
    # Placeholder for future training functionality
    return jsonify({'message': 'Training endpoint not yet implemented'}), 501

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


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
    if not model_initialized:
        return jsonify({'message': 'Model initialization failed. Service unavailable.'}), 503

    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    try:
        # Open and preprocess the image
        img = Image.open(file).convert('RGB').resize((224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)  # Add batch dimension

        print("Tensor shape:", tensor.shape)

        with torch.no_grad():
            prediction = model(tensor)
            print("Raw prediction output:", prediction)

        prediction_result = float(prediction[0][0].item())

        # Prediction threshold
        threshold = 0.5
        is_leached = prediction_result > threshold

        # Confidence calculation
        confidence = abs(prediction_result - threshold) * 2  # Scale 0 to 1
        confidence_percentage = round(confidence * 100, 2)  # Convert to %

        # Final prediction label
        prediction_label = "Leprosy detected" if is_leached else "Healthy skin (No Leprosy)"

        print(f"Prediction Result: {prediction_result}, Confidence: {confidence_percentage}%, Final Label: {prediction_label}")

        return jsonify({
            'message': 'File successfully processed',
            'prediction': prediction_label,
            'confidence_percentage': confidence_percentage,
            'raw_prediction': round(prediction_result, 4)
        }), 200

    except Exception as e:
        print("❌ Upload error:", str(e))
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

    if not model_initialized:
        return jsonify({'message': 'Model initialization failed. Service unavailable.'}), 503
        
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    try:
        # Open the image file and apply transformations
        img = Image.open(file).convert('RGB').resize((224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)

        print("Tensor shape:", tensor.shape)

        with torch.no_grad():
            # Make the prediction
            prediction = model(tensor)
            print("Raw prediction output:", prediction)

        prediction_result = float(prediction[0][0].item())
        
        # Determine if image is leached based on prediction
        # Adjust threshold as needed based on your model's training
        threshold = 0.5
        is_leached = prediction_result > threshold
        
        result_message = "Leached detected" if is_leached else "No leaching detected"
        
        return jsonify({
            'message': 'File successfully processed', 
            'prediction': prediction_result,
            'is_leached': is_leached,
            'result': result_message
        }), 200

    except Exception as e:
        print("❌ Upload error:", str(e))
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'ok' if model_initialized else 'degraded',
        'model_initialized': model_initialized,
        'model_weights_loaded': os.path.exists(model_weights_path) and os.path.getsize(model_weights_path) > 0
    }
    return jsonify(status)

# Training endpoint (placeholder for future implementation)
@app.route('/train', methods=['POST'])
def train_model():
    # This would be where you could implement model training functionality
    return jsonify({'message': 'Training endpoint not yet implemented'}), 501

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)