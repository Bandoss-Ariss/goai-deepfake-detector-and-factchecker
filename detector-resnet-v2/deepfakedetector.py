import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from resnet import resnet18
from resnet import resnet50

import sys
sys.path.append(".") 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = torch.load('train_dataset.pt')
val_dataset = torch.load('test_dataset.pt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = resnet50(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1) 
)

model = model.to(device)

# Load the best saved model

model.load_state_dict(torch.load('best_model.pth'))

print("Loaded best model weights.")

# Extract labels from the dataset
train_labels = [label for _, label in dataset]
val_labels = [label for _, label in val_dataset]

# Print class distribution
from collections import Counter
print("Class Distribution in Training Set:", Counter(train_labels))
print("Class Distribution in Validation Set:", Counter(val_labels))

num_fake = train_labels.count(0)
num_real = train_labels.count(1)
total = num_fake + num_real

# Calculate weights
weight_fake = total / num_fake
weight_real = total / num_real

# Use the weighted loss function
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_real / weight_fake], device=device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience = 5
patience_counter = 0
num_epochs = 30

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_images, val_labels in val_dataloader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels.float())
            val_running_loss += val_loss.item()

            # Calculate accuracy
            val_predictions = torch.round(torch.sigmoid(val_outputs))
            correct += (val_predictions == val_labels).sum().item()
            total += val_labels.size(0)

    val_loss = val_running_loss / len(val_dataloader)
    val_accuracy = correct / total

    # Print metrics
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print("Early stopping triggered!")
        break


print("Training completed.")

torch.save(model.state_dict(), 'deepfake_image_classifier_final.pth')