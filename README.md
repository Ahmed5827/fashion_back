
# Outfit Recommendation API  

## Overview  

This project is a Flask-based API that analyzes a person's image to suggest a customized outfit based on detected clothing, facial features, and an input theme. It utilizes multiple AI models for person detection, clothing classification, facial analysis, and outfit generation.

## Features  

- **Person Detection**: Uses YOLOv8 to detect and crop the person from an image.  
- **Clothing Classification**: Uses MobileNetV2 to classify the detected person's clothing.  
- **Facial Analysis**: Uses DeepFace to extract age, gender, and emotion.  
- **Outfit Generation**: Uses Google Gemini API to generate outfit descriptions.  
- **AI Image Generation**: Uses Stable Diffusion to generate images of the recommended outfit, including separate images for tops, pants, and shoes.  

## Technologies Used  

- **Flask**: Python web framework  
- **YOLOv8**: Object detection for identifying persons in an image  
- **MobileNetV2**: CNN model for clothing classification  
- **DeepFace**: Facial analysis for age, gender, and emotion detection  
- **Google Gemini API**: Text-based outfit recommendation generation  
- **Stable Diffusion**: AI image generation for outfits  

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Ahmed5827/fashion_back.git
   cd outfit-recommendation-api

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv  
   source venv/bin/activate

3. Install dependencies:
     ``` bash
    pip install -r requirements.txt

4. Set up API keys:
	•	Replace TOKEN with your Hugging Face token for Stable Diffusion.
	•	Replace GOOGLE_API_KEY with your Google Gemini API key.

## Usage

Start the API

Run the Flask app:

python app.py

## Endpoints

1. Generate Outfit Recommendation

# Endpoint:

POST /generate-outfit

## Parameters:
	•	image (file): Image containing a person.
	•	theme (optional, string): Theme for the outfit (e.g., casual, business, sporty).

 ## Response:
Returns:
	•	Detected age, gender, emotion
	•	Current clothing classification
	•	AI-generated outfit description
	•	Base64-encoded images of the outfit, pants, top, and shoes

## Example Usage (Python Request):

import requests  

url = "http://localhost:5000/generate-outfit"  
files = {'image': open('person.jpg', 'rb')}  
data = {'theme': 'business casual'}  
response = requests.post(url, files=files, data=data)  
print(response.text)  

## Download Generated Outfit Image

# Endpoint:
'''bash
GET /download/<filename>

## Downloads a generated outfit image.

# Example Usage:

GET 'http://localhost:5000/download/outfit.png';

# Notes
	•	The model runs on CUDA (GPU) if available; otherwise, it defaults to CPU.
	•	Ensure the Hugging Face token and Google API key are set correctly before running the application.

# License

This project is licensed under the MIT License.

Save this content as `README.md` in your project folder. Let me know if you need any modifications!
