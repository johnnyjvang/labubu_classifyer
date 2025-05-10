Hosted on Render: https://labubu-classifyer.onrender.com/

# 🧠 Labubu Classifier

This is a simple image classification web app that uses a trained PyTorch model (ResNet18) to identify whether an uploaded image is **Labubu** or **Not Labubu**. The app is built using **Flask** and deployed using **Render**.

---

## 🚀 How It Works

1. **Upload an image** using the web interface.
2. The image is processed and passed through a fine-tuned ResNet18 model.
3. The model returns:
   - The predicted class: `Labubu` or `Not Labubu`
   - A confidence score (percentage)
4. The uploaded image and results are displayed on the page.

---

## 🧠 Training Details

The model was trained on a custom dataset using PyTorch:

- **Labubu images**:  
  - 500 real images  
  - 2,500 augmented images  
  - **Total**: 3,000

- **Not Labubu images**:  
  - 1,500 real images  
  - 3,000 augmented images  
  - **Total**: 4,500

Images were gathered using a Google Images scraper. Data augmentation was applied to improve generalization and reduce overfitting.

The model architecture is based on **ResNet18**, with a modified final layer for binary classification.

---

## 🧰 Technologies Used

- Python 3.10+
- Flask
- PyTorch (ResNet18)
- PIL (Pillow)
- torchvision
- Render (for deployment)

---
