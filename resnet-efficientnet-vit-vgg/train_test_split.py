import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_path = '/teamspace/studios/deepfake-image-detection-resnet/dataset'
# Load the dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Define the sizes of the train and test subsets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing

# Split the dataset into train and test subsets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Save the train and test datasets
torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')