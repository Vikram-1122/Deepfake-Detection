import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a Simplified M2TR Model (or replace with a smaller model)
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # Use ResNet18 for faster execution
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Model, Loss, and Optimizer
model = SimpleResNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data Transformations (Resize for Speed)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Lower resolution for faster training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
])

# Load datasets and reduce size for quick testing
train_dataset = datasets.ImageFolder("C:/Users/Vikram/DFDC/data/final/train", transform=transform)
val_dataset = datasets.ImageFolder("C:/Users/Vikram/DFDC/data/final/val", transform=transform)

# Use a subset of data for faster testing
train_dataset = Subset(train_dataset, range(500))  # Use first 500 samples
val_dataset = Subset(val_dataset, range(100))      # Use first 100 samples

# Data Loaders with reduced batch size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss, train_acc = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels).item()

        # Validation Phase
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels).item()

        # Print Metrics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc/len(train_loader.dataset):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc/len(val_loader.dataset):.4f}")

        # Save Model Checkpoint
        torch.save(model.state_dict(), f"m2tr_epoch_{epoch+1}.pth")
        print(f"Model checkpoint saved: m2tr_epoch_{epoch+1}.pth")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)
