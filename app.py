import os
import uuid

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Allowed upload types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

# Upload directory
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model setup
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

MODEL_PATH = "labubu_classifier.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_image(image):
    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)
    probs = F.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

    prediction = "Labubu" if predicted_class.item() == 0 else "Not Labubu"
    return prediction, confidence.item() * 100


def clear_upload_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    error_message = None

    if request.method == "POST":
        if "image" not in request.files:
            error_message = "No image file was uploaded."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
                error_message=error_message,
            )

        file = request.files["image"]

        if file.filename == "":
            error_message = "Please choose an image file."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
                error_message=error_message,
            )

        if not allowed_file(file.filename):
            error_message = "Unsupported file type. Please upload PNG, JPG, JPEG, GIF, or BMP."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
                error_message=error_message,
            )

        try:
            # Convert all images to RGB so ResNet always receives 3 channels
            image = Image.open(file.stream).convert("RGB")

            # Make prediction
            prediction, confidence = predict_image(image)

            # Clear old uploaded images only
            clear_upload_folder()

            # Save uploaded image with unique name
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
            image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            image.save(image_path)

            # Build image URL
            image_url = url_for("static", filename=f"uploads/{unique_filename}")

        except UnidentifiedImageError:
            error_message = "The uploaded file is not a valid image."
        except Exception as e:
            print(f"Prediction error: {e}")
            error_message = "An error occurred while processing the image. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
