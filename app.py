from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Define the number of classes (benign and malignant)
num_classes = 2
class_labels = [ "healthy","cancer"]

# Load the PyTorch model state dict
model_path = './model/efficientNetB0_state_dict.pth'
model = load_model(model_path, num_classes)
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 for EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for EfficientNet
])

# Function to check if file is a valid image (.jpg or .png)
def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

# Predict function that you provided
def predict_image(image_path, model=model):
    image = Image.open(image_path).convert('RGB')  # Open and convert the image to RGB
    image_tensor = transform(image)  # Apply transformations
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    print(f"Image tensor shape: {image_tensor.shape}, Image tensor dtype: {image_tensor.dtype}")

    with torch.no_grad():
        outputs = model(image_tensor)  # Get model output
        print(f"Model output: {outputs}")

        probabilities = torch.sigmoid(outputs).squeeze()  # Apply sigmoid for probabilities
        predicted = probabilities > 0.5  # Binarize the result

        print(f"Predicted index: {predicted.item()}")
        predicted_class = class_labels[int(predicted.item())]  # Convert the index to class label

    return predicted_class

# Flask route for the main page
@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            if is_image_file(file.filename):
                img_path = os.path.join("uploads", file.filename)
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")
                file.save(img_path)
                prediction = predict_image(img_path)  # Use the updated prediction function
                return render_template("index.html", prediction=prediction)
            else:
                return "Please upload a JPG or PNG file."
    return render_template("index.html")

# Flask route for API-based prediction
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and is_image_file(file.filename):
            img_path = os.path.join("uploads", file.filename)
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            file.save(img_path)
            prediction = predict_image(img_path)  # Use the updated prediction function
            return prediction
        else:
            return "Please upload a JPG or PNG file."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
