import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from training.detectors import DETECTOR
import yaml
import gradio as gr
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# available models in the repository
AVAILABLE_MODELS = [
    "xception",
    "ucf",
]

def load_model(model_name, config_path, weights_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['model_name'] = model_name
    
    # download the pretrained model from Hugging Face
    if 'pretrained' in config:
        pretrained_filename = f"{model_name}_best.pth"
        repo_id = "ArissBandoss/deepfake-video-classifier"
        pretrained_path = hf_hub_download(repo_id=repo_id, filename=pretrained_filename)
        config['pretrained'] = pretrained_path
    
    model_class = DETECTOR[model_name]
    model = model_class(config).to(device)
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


# preprocess a single video
def preprocess_video(video_path, output_dir, frame_num=32):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
    
    # extract frames
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

# inference on a single video
def infer_video(video_path, model, device):
    output_dir = "temp_video_frames"
    frames = preprocess_video(video_path, output_dir)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    probs = []
    for frame_path in frames:
        frame = Image.open(frame_path).convert("RGB")
        frame = transform(frame).unsqueeze(0).to(device)
        
        data_dict = {
            "image": frame,
            "label": torch.tensor([0]).to(device),  # Dummy label
            "label_spe": torch.tensor([0]).to(device),  # Dummy specific label
        }
        
        with torch.no_grad():
            pred_dict = model(data_dict, inference=True)
            
            logits = pred_dict["cls"]  # Shape: [batch_size, num_classes]
            prob = torch.softmax(logits, dim=1)[:, 1].item()  # Probability of being "fake"
            probs.append(prob)
    
    # aggregate predictions (e.g., average probability)
    avg_prob = np.mean(probs)
    prediction = "Fake" if avg_prob > 0.5 else "Real"
    return prediction, avg_prob

# Gradio inference function
def gradio_inference(video, model_name):
    # Download config and weights from Hugging Face Model Registry
    repo_id = "ArissBandoss/deepfake-video-classifier"
    config_filename = f"{model_name}.yaml"
    weights_filename = f"{model_name}_best.pth"
    
    # download files
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    weights_path = hf_hub_download(repo_id=repo_id, filename=weights_filename)
    
    # load the model
    model = load_model(model_name, config_path, weights_path)
    
    # inference
    prediction, confidence = infer_video(video, model, device)
    return f"Model: {model_name}\nPrediction: {prediction} (Confidence: {confidence:.4f})"

# Gradio App
def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Deepfake Detection Demo")
        gr.Markdown("Upload a video and select a model to detect if it's real or fake.")
        
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            model_dropdown = gr.Dropdown(choices=AVAILABLE_MODELS, label="Select Model", value="xception")
        
        output_text = gr.Textbox(label="Prediction Result")
        
        submit_button = gr.Button("Run Inference")
        submit_button.click(
            fn=gradio_inference,
            inputs=[video_input, model_dropdown],
            outputs=output_text,
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(share=True)