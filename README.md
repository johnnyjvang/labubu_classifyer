# Labubu Classifier

A lightweight web-based image classification application that determines whether an uploaded image contains a Labubu character. The system leverages a fine-tuned ResNet18 convolutional neural network implemented in PyTorch and is deployed using Flask on Render.

**Live Application:** https://labubu-classifyer.onrender.com/

---

## Overview

This application provides a simple interface for users to upload an image and receive a classification result in real time. The backend processes the image using a trained deep learning model and returns both a predicted label and a confidence score.

---

## Features

- Upload and classify images through a web interface
- Binary classification: **Labubu** vs **Not Labubu**
- Confidence score output for model predictions
- Automatic handling and cleanup of uploaded images
- Lightweight deployment using Flask and Render

---

## Model Details

The classification model is based on a modified ResNet18 architecture:

- Pretrained backbone replaced with custom-trained weights
- Final fully connected layer adapted for binary classification
- Softmax applied to generate class probabilities

### Dataset Composition

| Class         | Real Images | Augmented Images | Total |
|--------------|------------|------------------|-------|
| Labubu        | 500        | 2,500            | 3,000 |
| Not Labubu    | 1,500      | 3,000            | 4,500 |

Data was collected using a Google Images scraping approach and augmented to improve model generalization and robustness.

---

## System Architecture

1. User uploads an image via the web interface
2. Image is resized and normalized using torchvision transforms
3. The processed image is passed into the ResNet18 model
4. The model outputs class probabilities
5. The application returns:
   - Predicted label (Labubu / Not Labubu)
   - Confidence score (%)
6. The uploaded image and results are displayed to the user

---

## Technology Stack

- Python 3.10+
- Flask (web framework)
- PyTorch (model inference)
- torchvision (image preprocessing and model architecture)
- Pillow (image handling)
- Render (deployment platform)

---

## Project Structure

```
labubu_classifyer/
├── app.py                  # Main Flask application
├── labubu_classifier.pth  # Trained model weights
├── static/                # Uploaded images (temporary storage)
├── templates/
│   └── index.html         # Web interface
├── requirements.txt
└── README.md
```

---

## Running Locally

### 1. Clone the Repository
```
git clone https://github.com/johnnyjvang/labubu_classifyer.git
cd labubu_classifyer
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Application
```
python app.py
```

The application will be available at:
```
http://localhost:10000
```

---

## Deployment

The application is deployed using Render. The Flask server is configured to run on port 10000 to comply with Render's hosting requirements.

---

## Notes and Limitations

- Model performance is dependent on dataset quality and diversity
- Misclassifications may occur on low-quality or ambiguous images
- Uploaded images are temporarily stored and cleared between requests

---

## Future Improvements

- Expand dataset for improved accuracy
- Add multi-class classification support
- Implement batch image uploads
- Enhance UI/UX for better user interaction

---

## License

This project is open source and available under the MIT License.
