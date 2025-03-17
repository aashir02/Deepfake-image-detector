import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from resnet_model import CustomResNet18

# Configuration
lr_rate = 0.01
batch_size = 64
num_epochs = 30
optimizer_name = 'rmsprop'
target_size = (224, 224)

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(224, 224)):
        self.transform = transform
        self.target_size = target_size
        self.image_paths, self.labels = self._load_dataset(data_dir)

    def _load_dataset(self, data_dir):
        image_paths, labels = [], []
        for label, category in enumerate(["Real", "Fake"]):
            category_dir = os.path.join(data_dir, category)
            images = [os.path.join(category_dir, img) for img in os.listdir(category_dir)]
            image_paths.extend(images)
            labels.extend([label] * len(images))
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).resize(self.target_size)
        image = transforms.ToTensor()(image)
        return image, self.labels[idx]

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)

        num_train = int(train_ratio * len(self))
        num_val = int(val_ratio * len(self))

        self.train_indices = indices[:num_train]
        self.val_indices = indices[num_train:num_train + num_val]
        self.test_indices = indices[num_train + num_val:]

def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses, train_accuracies = [], []
    model.to(device)

    save_dir = r"C:\Users\ashir\OneDrive\Pictures\Deepfake_image_detector"
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    save_path = os.path.join(save_dir, "model.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (torch.round(torch.sigmoid(outputs)) == labels).sum().item()
            total += labels.size(0)

        epoch_loss, epoch_accuracy = running_loss / len(train_loader.dataset), correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_metrics = evaluate_model(model, val_loader, device)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, '
              f'Val Accuracy: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}, F1: {val_metrics[3]:.4f}')

    # Save model only once after training completes
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved at: {save_path}")

    plot_training_progress(train_losses, train_accuracies)

def evaluate_model(model, data_loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds.extend(torch.round(torch.sigmoid(model(inputs))).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return (accuracy_score(targets, preds),
            precision_score(targets, preds, zero_division=1),
            recall_score(targets, preds, zero_division=1),
            f1_score(targets, preds))

def plot_training_progress(losses, accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    data_dir = r"C:\Users\ashir\OneDrive\Pictures\Deepfake_image_detector"
    transform = get_augmentation_transforms()
    dataset = DeepfakeDataset(data_dir, transform=transform, target_size=target_size)
    dataset.split_dataset()

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(dataset.train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(dataset.val_indices))
    
    model = CustomResNet18(num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()
