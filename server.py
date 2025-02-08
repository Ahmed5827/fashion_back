from flask import Flask, request, send_file
import cv2
import numpy as np
import torch
import os
import base64
from io import BytesIO
from deepface import DeepFace
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
import google.generativeai as genai
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Load YOLOv8 Model for person detection
yolo_model = YOLO("yolov8n.pt")

# Load MobileNetV2 Model for clothing classification
cnn_model = MobileNetV2(weights="imagenet")

# Load Stable Diffusion Model for image generation
TOKEN = "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"
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float32, use_auth_token=TOKEN
)
pipe.to(device)

# Configure Google Gemini API
GOOGLE_API_KEY = "AIzaSyCqTpA_yFBxOAEuETe92yHvs5g1UVX0728"
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

def image_to_base64(image_path):
    """Convert an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def detect_person(image_path):
    results = yolo_model(image_path)  
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            if label == "person":
                return box.xyxy[0].tolist()
    return None  

def crop_person(image_path, bbox):
    image = cv2.imread(image_path)
    if image is None:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    cropped_person = image[y1:y2, x1:x2]
    if cropped_person.size == 0:
        return None
    cv2.imwrite("cropped_person.jpg", cropped_person)
    return cropped_person

def classify_clothing(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = cnn_model.predict(img_array)
    return decode_predictions(predictions, top=3)[0]

def analyze_facial_features(image_path):
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'])
        return float(analysis[0]['age']), analysis[0]['gender'], analysis[0]['emotion']
    except Exception:
        return None, None, None

def gemini_generate_outfit(theme, age, gender, emotion, current_clothing):
    model = genai.GenerativeModel("gemini-1.5-pro", generation_config=generation_config)
    convo = model.start_chat(history=[])

    prompt = (
        f"Generate a complete outfit for a researcher based on:\n"
        f"- Theme: {theme}\n"
        f"- Age: {age}\n"
        f"- Gender: {gender}\n"
        f"- Emotion: {emotion}\n"
        f"- Current Clothing: {current_clothing}\n"
        f"Describe the top, bottom, and shoes, including colors, styles, and accessories."
    )

    convo.send_message(prompt)
    return convo.last.text.strip()

def generate_image(prompt, name):
    result = pipe(prompt, guidance_scale=8.5)
    image = result["images"][0]  
    filename = f"{name}.png"
    image.save(filename)
    return filename  

def safe_json(value):
    """Convert all non-serializable types to strings."""
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    elif isinstance(value, np.generic):  # Covers float32, int64, etc.
        return str(value.item())
    elif isinstance(value, (int, float, str, bool, list, dict)):
        return str(value)
    else:
        return str(value)

@app.route('/generate-outfit', methods=['POST'])
def generate_outfit():
    if 'image' not in request.files:
        return str("Error: No image file provided"), 400

    image = request.files['image']
    theme = request.form.get('theme', 'default')  
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    person_bbox = detect_person(image_path)
    if not person_bbox:
        return str("Error: No person detected in the image"), 400

    person_image = crop_person(image_path, person_bbox)
    if person_image is None:
        return str("Error: Failed to crop the person"), 400

    clothing_predictions = classify_clothing(person_image)
    current_clothing = ", ".join([pred[1] for pred in clothing_predictions])

    age, gender, emotion = analyze_facial_features("cropped_person.jpg")
    if age is None:
        return str("Error: Facial analysis failed"), 400

    outfit_suggestion = gemini_generate_outfit(theme, age, gender, emotion, current_clothing)

    outfit_image_path = generate_image(outfit_suggestion, "outfit")
    pants_image_path = generate_image(outfit_suggestion + " pants", "pants")
    top_image_path = generate_image(outfit_suggestion + " top", "top")
    shoes_image_path = generate_image(outfit_suggestion + " shoes", "shoes")

    return str(
        f"Age: {safe_json(age)}\n"
        f"Gender: {safe_json(gender)}\n"
        f"Emotion: {safe_json(emotion)}\n"
        f"Current Clothing: {safe_json(current_clothing)}\n"
        f"Outfit Suggestion: {safe_json(outfit_suggestion)}\n"
        f"Images: \n"
        f"Outfit: {image_to_base64(outfit_image_path)}\n"
        f"Pants: {image_to_base64(pants_image_path)}\n"
        f"Top: {image_to_base64(top_image_path)}\n"
        f"Shoes: {image_to_base64(shoes_image_path)}\n"
    )

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
