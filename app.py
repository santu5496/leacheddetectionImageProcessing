import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import CNN

app = Flask(__name__)

# Define dataset folder path
dataset_folder = r"C:\Users\hpatil\source\repos\santu5496\leacheddetectionImageProcessing\Dataset1\CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients\imgs"

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

# Function to check if the image exists in the dataset folder
def find_image_in_dataset(image_name):
    image_path = os.path.join(dataset_folder, image_name)
    if os.path.exists(image_path):
        return image_path
    return None

@app.route('/', methods=['GET'])
def index():
    return render_template('login.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

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

        # Get the filename and search in the dataset
        image_name = file.filename
        found_image = find_image_in_dataset(image_name)

        if not found_image:
            return jsonify({'message': f'Image "{image_name}" not found in dataset folder.'}), 404

        # If image found, process it with your model as needed
        with torch.no_grad():
            prediction = model(tensor)

        prediction_result = float(prediction[0][0].item())
        threshold = 0.5
        is_leached = prediction_result > threshold
        confidence = abs(prediction_result - threshold) * 2
        confidence_percentage = round(confidence * 100, 2)
        prediction_label = "Leprosy detected" if is_leached else "Healthy skin (No Leprosy)"

        return jsonify({
            'message': 'File successfully processed',
            'prediction': prediction_label,
            'confidence_percentage': confidence_percentage,
            'raw_prediction': round(prediction_result, 4),
            'found_image': found_image  # Returning the found image path
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
