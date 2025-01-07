import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from resnet import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = resnet50()
num_features = model.fc.in_features
#Remplacement du Réseau de neurones pour un classifier
model.fc = nn.Sequential(
    nn.Linear(num_features, 1)  
)
model.to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_dataset = torch.load('test_dataset.pt')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_predictions = []
all_labels = []


with torch.no_grad():
    for images, labels in test_loader:
   
        images = images.to(device)
        labels = labels.to(device)
        
        
        outputs = model(images)
        
        probabilities = torch.sigmoid(outputs)
        predictions = torch.round(probabilities).squeeze()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate evaluation metrics (e.g., accuracy, precision, recall, F1 score)
accuracy = (all_predictions == all_labels).mean()
print("Accuracy:", accuracy)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)