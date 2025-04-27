import torch
import torch.optim as optim
import torch.nn as nn
from model import CNN  # Import your CNN model class
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 10

# Directory containing images
dataset_dir = r'C:\Users\hpatil\Downloads\archive\CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients\split\train'

# Custom dataset class to handle loading images without class-based folder structure
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        # Get all image files and manually assign labels
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.img_paths.append(img_path)
                # Assuming you have labels, you can manually assign them or use filenames
                # For now, assigning dummy labels (you will have to update this part)
                label = 0  # Dummy class 0, update this logic
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Open image
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset using the custom dataset
train_dataset = CustomImageDataset(img_dir=dataset_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Adjust if necessary based on your model output
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())  # labels should be float for BCELoss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels.unsqueeze(1).float()).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Save the trained model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("✅ Model trained and weights saved successfully.")
