import torch
import os
import cv2
import torch
from training.detectors import DETECTOR
import yaml
import numpy as np
import requests
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from fastapi import FastAPI, Request, Form, File, UploadFile, status
from fastapi.responses import JSONResponse
from twilio.rest import Client
from dotenv import load_dotenv
from resnet import resnet50
from fastapi.middleware.cors import CORSMiddleware
import io
load_dotenv()
twilio_ssid = os.getenv("TWILIO_SSID")
TWILIO_WHATSAPP_URL = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_ssid}/Messages.json"
TWILIO_AUTH = (os.getenv("TWILIO_SSID"), os.getenv("TWILIO_SECRET_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_model = resnet50(pretrained=False)
num_features = image_model.fc.in_features
image_model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 1)
)
image_model.load_state_dict(torch.load('best_model.pth', map_location=device))
image_model.to(device)
image_model.eval()

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process incoming media
async def process_image(media_url,account_sid, auth_token):
    response = requests.get(media_url, auth=(account_sid, auth_token), timeout=30)
    if response.status_code != 200:
            return f"Failed to download media. Status code: {response.status_code}, Error: {response.text}"
    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    image = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = image_model(image)
        prediction = torch.sigmoid(output).item()
    result = "Real" if prediction > 0.8 else "Fake"
    confidence = prediction if result == "Real" else 1 - prediction
    return f"Image classified as: {result} with confidence {confidence:.4f}"


async def process_audio(media_url):
    return "Audio processing"

# Load the UCF model from Hugging Face
def load_ucf_model():
    repo_id = "ArissBandoss/deepfake-video-classifier"
    model_name = "ucf"

    config_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}.yaml")
    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}_best.pth")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model_name'] = model_name
    config['pretrained'] = weights_path
    model_class = DETECTOR[model_name]
    ucf_model = model_class(config).to(device)
    ucf_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    ucf_model.eval()

    return ucf_model

def preprocess_video(video_path, frame_num=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    return frames

async def process_video(media_url, account_sid, auth_token):
    response = requests.get(media_url, auth=(account_sid, auth_token), timeout=30)
    if response.status_code != 200:
        return f"Failed to download media. Status code: {response.status_code}, Error: {response.text}"

    temp_video_path = "/tmp/temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(response.content)

    frames = preprocess_video(temp_video_path)
    if not frames:
        return "Failed to process video."

    # Apply transformations to frames
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    frames_tensor = torch.stack([transform(Image.fromarray(frame)) for frame in frames]).to(device)

    ucf_model = load_ucf_model()


    data_dict = {
        "image": frames_tensor,
        "label": torch.tensor([0]).to(device),  # Dummy label
        "label_spe": torch.tensor([0]).to(device)  # Dummy specific label
    }

    # Perform inference
    with torch.no_grad():
        pred_dict = ucf_model(data_dict, inference=True)

    logits = pred_dict["cls"]
    prob = torch.softmax(logits, dim=1)[:, 1].mean().item()

    result = "Fake" if prob > 0.5 else "Real"
    return f"Video classified as: {result} with confidence {prob:.4f}"
# Transform frames to tensors
def transform_frames(frames):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    frames_tensor = torch.stack([transform(Image.fromarray(frame)) for frame in frames]).to(device)
    return frames_tensor

# Prediction endpoint
@app.post("/predict-image/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Apply transformations
        image = image_transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = image_model(image)
            prediction = torch.sigmoid(output).item()

        # Determine prediction based on threshold
        threshold = 0.8
        if prediction > threshold:
            result = "Real"
        else:
            result = "Fake"

        # Return the prediction and confidence
        print(prediction)
        confidence = prediction if result == "Real" else 1 - prediction
        return {"prediction": result, "confidence": round(confidence, 2)}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video to a temporary file
        temp_video_path = "/tmp/temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())

        # Preprocess the video into frames
        frames = preprocess_video(temp_video_path)
        if not frames:
            return {"error": "Failed to process video."}

        # Transform frames into tensors
        frames_tensor = transform_frames(frames)

        # Load the UCF model
        ucf_model = load_ucf_model()

        # Prepare input data
        data_dict = {
            "image": frames_tensor,
            "label": torch.tensor([0]).to(device),  # Dummy label
            "label_spe": torch.tensor([0]).to(device)  # Dummy specific label
        }

        # Perform inference
        with torch.no_grad():
            pred_dict = ucf_model(data_dict, inference=True)

        logits = pred_dict["cls"]
        prob = torch.softmax(logits, dim=1)[:, 1].mean().item()

        # Return the result
        result = "Fake" if prob > 0.5 else "Real"
        return {"prediction": result, "confidence": round(prob, 4)}
    except Exception as e:
        return {"error": str(e)}

# WhatsApp webhook
@app.post("/whatsapp-webhook/")
async def whatsapp_webhook(request: Request):
    data = await request.form()
    media_url = data.get("MediaUrl0", None)
    media_type = data.get("MediaContentType0", "").split("/")[0]
    from_number = data.get("From", "")

    if media_url:
        if media_type == "image":
            result = await process_image(media_url, os.getenv("TWILIO_SSID"),os.getenv("TWILIO_SECRET_KEY"))
        elif media_type == "audio":
            result = await process_audio(media_url)
        elif media_type == "video":
            result = await process_video(media_url, os.getenv("TWILIO_SSID"),os.getenv("TWILIO_SECRET_KEY"))
        else:
            result = "Unsupported media type."
    else:
        result = "No media received."
    response_data = {
            "From": "whatsapp:+212660862067",
            "To": from_number,
            "Body": result,
        }
    response = requests.post(TWILIO_WHATSAPP_URL, data=response_data, auth=TWILIO_AUTH)
    print(response)
    return JSONResponse({"status": "success"})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return health_check_handler()

@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def health_check_handler():
    return "200 OK"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Use PORT environment variable or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)