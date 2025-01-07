import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models


# Define the M2TR Model
class M2TR(nn.Module):
    def __init__(self, num_classes=2):
        super(M2TR, self).__init__()
        # Pretrained ResNet as a backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the fully connected layer
        
        # Transformer block
        self.transformer = nn.Transformer(d_model=2048, nhead=8, num_encoder_layers=4)
        
        # Classification head
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Extract features using ResNet
        features = self.backbone(x)
        features = features.unsqueeze(1)  # Add sequence dimension for the transformer

        # Pass through transformer
        transformed = self.transformer(features, features)

        # Global average pooling
        pooled = transformed.mean(dim=1)

        # Classification head
        output = self.fc(pooled)
        return output


# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define Data Loading and Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure size compatibility
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
])

# Load datasets
train_dataset = datasets.ImageFolder("C:/Users/Vikram/DFDC/data/final/train", transform=transform)
val_dataset = datasets.ImageFolder("C:/Users/Vikram/DFDC/data/final/val", transform=transform)
test_dataset = datasets.ImageFolder("C:/Users/Vikram/DFDC/data/final/test", transform=transform)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
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

            # Track accuracy
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

        # Print Stats
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc/len(train_loader.dataset):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc/len(val_loader.dataset):.4f}")

    # Save Model
    print("Saving model...")
    torch.save(model.state_dict(), "C:/Users/Vikram/DFDC/m2tr_model.pth")

    print("Model saved as 'm2tr_model.pth'.")


# Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels).item()

    print(f"Test Accuracy: {test_acc / len(test_loader.dataset):.4f}")


# Main Function
if __name__ == "__main__":
    # Initialize Model, Loss, and Optimizer
    model = M2TR(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the Model
    evaluate_model(model, test_loader)
