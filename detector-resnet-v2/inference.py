import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 1)
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    if prediction > 0.8:
        print(f"Prediction: Real ({prediction:.4f})")  # Class 1 = Real
    else:
        print(f"Prediction: Fake ({prediction:.4f})")  # Class 0 = Fake

predict('bb.jpeg')
