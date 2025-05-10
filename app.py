import torch
from flask import Flask, render_template, request, url_for
from PIL import Image
from torchvision import transforms, models
import os
import uuid
from io import BytesIO
import torch.nn.functional as F
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Model setup
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
MODEL_PATH = 'labubu_classifier.pth'
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Upload directory
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction logic
def predict_image(image):
    image = transform(image).unsqueeze(0)
    output = model(image)
    probs = F.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)
    prediction = 'Labubu' if predicted_class.item() == 0 else 'Not Labubu'
    return prediction, confidence.item() * 100

# Utility to clear old images
def clear_static_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']

        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
            image = Image.open(file.stream)

            # Make prediction
            prediction, confidence = predict_image(image)

            # Clear existing images
            clear_static_folder()

            # Generate unique filename to avoid caching
            ext = file.filename.rsplit('.', 1)[-1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            image_path = os.path.join(UPLOAD_FOLDER, secure_filename(unique_filename))
            image.save(image_path)

            # Pass relative path to template
            image_url = url_for('static', filename=unique_filename)

    return render_template('index.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    # app.run(debug=True)
    # app.run(host="0.0.0.0", port=8080, debug=False)
    app.run(host='0.0.0.0', port=10000)  # Render listens on port 10000
