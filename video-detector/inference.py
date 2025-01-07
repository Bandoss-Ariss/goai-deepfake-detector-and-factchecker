import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from training.detectors import DETECTOR
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(model_name, config_path, weights_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the model name in the config
    config['model_name'] = model_name
    
    # Load the model
    model_class = DETECTOR[model_name]
    model = model_class(config).to(device)
    
    # Load the pre-trained weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model

# Preprocess a single video
def preprocess_video(video_path, output_dir, frame_num=32):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
    
    cap.release()
    return frames

# Perform inference on a single video
def infer_video(video_path, model, device):
    # Preprocess the video
    output_dir = "temp_video_frames"
    frames = preprocess_video(video_path, output_dir)
    
    # Load and preprocess frames
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Perform inference
    probs = []
    for frame_path in frames:
        frame = Image.open(frame_path).convert("RGB")
        frame = transform(frame).unsqueeze(0).to(device)
        
        # Create a data_dict with dummy labels for UCFDetector
        data_dict = {
            "image": frame,
            "label": torch.tensor([0]).to(device),  # Dummy label
            "label_spe": torch.tensor([0]).to(device),  # Dummy specific label
        }
        
        with torch.no_grad():
            pred_dict = model(data_dict, inference=True)
            
            # Compute probability from logits (cls output)
            logits = pred_dict["cls"]  # Shape: [batch_size, num_classes]
            prob = torch.softmax(logits, dim=1)[:, 1].item()  # Probability of being "fake"
            probs.append(prob)
    
    # Aggregate predictions (e.g., average probability)
    avg_prob = np.mean(probs)
    prediction = "Fake" if avg_prob > 0.5 else "Real"
    return prediction, avg_prob

# Main function for terminal-based inference
def main(video_filename, model_name):
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define paths for config and weights
    config_path = f"/teamspace/studios/this_studio/DeepfakeBench/training/config/detector/{model_name}.yaml"
    weights_path = f"/teamspace/studios/this_studio/DeepfakeBench/training/weights/{model_name}_best.pth"
    
    # Check if paths exist
    if not os.path.exists(config_path):
        print(f"Error: Config file for model '{model_name}' not found at {config_path}.")
        return
    if not os.path.exists(weights_path):
        print(f"Error: Weights file for model '{model_name}' not found at {weights_path}.")
        return
    
    # Load the model
    model = load_model(model_name, config_path, weights_path)
    
    # Perform inference
    video_path = os.path.join(os.getcwd(), video_filename)
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_filename}' not found in the current directory.")
        return
    
    prediction, confidence = infer_video(video_path, model, device)
    print(f"Model: {model_name}")
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")

# Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python inference_script.py <video_filename> <model_name>")
        print("Available models: xception, meso4, meso4Inception, efficientnetb4, ucf, etc.")
    else:
        video_filename = sys.argv[1]
        model_name = sys.argv[2]
        main(video_filename, model_name)