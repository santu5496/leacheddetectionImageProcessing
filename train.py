import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from torch.utils.data import DataLoader, TensorDataset

# Define hyperparameters
learning_rate = 0.001
num_epochs = 1
batch_size = 32

# Generate dummy data
num_samples = 100
input_shape = (3, 76, 102)
dummy_data = torch.randn(num_samples, *input_shape)
dummy_labels = torch.randint(0, 2, (num_samples, 1)).float()

# Create dataset and dataloader
dataset = TensorDataset(dummy_data, dummy_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model, loss function, and optimizer
model = CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved to model_weights.pth")