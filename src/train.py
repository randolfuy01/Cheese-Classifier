import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
import os
from torchvision import transforms
from model import CheeseNet
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        logger.info(f"Loading data from {data_folder}...")
        self.data_folder = data_folder
        self.transform = transform
        # List subdirectories for each class
        self.classes = os.listdir(data_folder)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all images with their class labels
        self.image_files = []
        for cls in self.classes:
            class_folder = os.path.join(data_folder, cls)
            if os.path.isdir(class_folder):
                image_files = [os.path.join(class_folder, f) 
                            for f in os.listdir(class_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                self.image_files.extend([(image, cls) for image in image_files])

        logger.info(f"Loaded {len(self.image_files)} images from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path, label = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # Convert label to an index
        label_idx = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)
        
        return image, label_idx


# Main script
if __name__ == "__main__":
    logger.info("Starting training script...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_folder = "./cheese"  # Folder containing subdirectories of different classes

    # Create dataset and dataloaders
    logger.info("Creating train and validation datasets and dataloaders...")
    train_dataset = CustomImageDataset(data_folder, transform=transform)
    val_dataset = CustomImageDataset(data_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, criterion, and optimizer
    logger.info("Initializing model...")
    model = CheeseNet(num_classes=len(train_dataset.classes), input_shape=(3, 224, 224))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training loop
        model.train() 
        train_loss = 0.0
        train_preds, train_labels = [], []

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Starting training...")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels) 

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if (i + 1) % 100 == 0:  # Log every 100 batches
                logger.info(f"Batch {i+1}/{len(train_loader)} - Training Loss: {loss.item():.4f}")

        train_accuracy = accuracy_score(train_labels, train_preds)
        
        # Validation loop
        model.eval()  
        val_loss = 0.0
        val_preds, val_labels = [], []

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Starting validation...")

        with torch.no_grad(): 
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(inputs)  
                loss = criterion(outputs, labels) 

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                if (i + 1) % 50 == 0:  # Log every 50 batches
                    logger.info(f"Batch {i+1}/{len(val_loader)} - Validation Loss: {loss.item():.4f}")

        val_accuracy = accuracy_score(val_labels, val_preds)

        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s")
        logger.info(f"Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
    # Save the trained model
    torch.save(model.state_dict(), 'cheese_net.pth')
    logger.info("Model saved to 'cheese_net.pth'")
    