import torch
from torchvision import transforms
from PIL import Image
import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
model = None  # Declare model as a global variable

def load_model():
    global model
    if model is None:
        try:
            from ThesisModeCNN import SimpleCNN  # Ensure SimpleCNN is imported inside the function
            model = SimpleCNN(num_classes=4)
            pth_path = os.path.join(script_directory, '4class_model New Algo New Dataset 4.pth')
            model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")

def applyindivmodel(image_path):
    load_model()  # Load model if not already loaded

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    input_image = transform(image).unsqueeze(0)

    try:
        with torch.no_grad():
            model_output = model(input_image)
            print(f'Debug - Raw Model Output: {model_output}')
            probabilities = torch.nn.functional.softmax(model_output, dim=1)
            print(f'Debug - Softmax Probabilities: {probabilities}')
            _, predicted_class = torch.max(probabilities, 1)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None, None

    predicted_class = predicted_class.item()
    probabilities = probabilities.squeeze().tolist()

    # Ensure that probabilities are printed as expected
    print(f'Debug - Final Model Output: {model_output}')
    print(f'Debug - Final Probabilities: {probabilities}')
    print(f'Debug - Final Predicted Class: {predicted_class}')

    return predicted_class, probabilities

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predicted_class, class_probabilities = applyindivmodel(image_path)
        if predicted_class is not None:
            print(f'Predicted Class: {predicted_class}')
            print(f'Class Probabilities: {class_probabilities}')
