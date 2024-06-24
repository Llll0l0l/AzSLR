import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
import argparse
from sklearn.metrics import classification_report, accuracy_score
import InternVideo  # Make sure to have this properly set up

# Define the custom dataset for loading videos and labels
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None, delimiter=' ')
        self.root_dir = root_dir
        self.transform = transform

        # Define the columns since there is no header in the CSV
        self.data.columns = ['video_path', 'label']

        # Create a mapping from string labels to integer indices
        self.label_to_index = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}  # For reverse mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label_str = self.data.iloc[idx, 1]

        # Load the video using InternVideo's function or another method
        video = InternVideo.load_video(video_path)  # Load the video as a tensor

        # Apply any transformations to the video
        if self.transform:
            video = self.transform(video)

        # Convert the string label to an integer using the mapping
        label = self.label_to_index[label_str]
        label = torch.tensor(label).long()

        return video, label

# Define transformations (if necessary)
transform = transforms.Compose([
    # Assuming the video is already a tensor, add any tensor-based transforms here
    # For example, normalization (if needed):
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Argument parser for command-line options
parser = argparse.ArgumentParser(description='Train or resume training of the InternVideo model.')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to resume training from.')

args = parser.parse_args()

# Create the dataset and dataloader
train_dataset = SignLanguageDataset('annotations/train1.csv', 'annotations', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

val_dataset = SignLanguageDataset('annotations/val2.csv', 'annotations', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load the pretrained InternVideo model
model = InternVideo.load_model("../../../InternVideo2/multi_modality/ckpt_new/InternVideo-MM-L-14 (1).ckpt").cuda()

# Debugging step to determine output feature size
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    sample_video, _ = next(iter(train_loader))
    sample_video = sample_video.cuda()
    feature_output = model.encode_video(sample_video)
    print(f"Feature output shape: {feature_output.shape}")  # Print the shape of the feature tensor

# Define the custom model class with the correct feature dimension
class InternVideoClassifier(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes):
        super(InternVideoClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(feature_dim, num_classes)  # Adjust based on the actual output dimension

    def forward(self, x):
        features = self.base_model.encode_video(x)
        logits = self.classifier(features)
        return logits

# Use the printed shape to determine feature dimension
feature_dim = feature_output.shape[1]  # Assuming the feature dimension is the second dimension

# Number of classes in your dataset
num_classes = len(train_dataset.label_to_index)

# Wrap the base model with the custom classifier
custom_model = InternVideoClassifier(model, feature_dim, num_classes).cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)

# Function to save a checkpoint
def save_checkpoint(model, optimizer, epoch, batch_idx, filename):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")

# Function to load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    batch_idx = checkpoint.get('batch_idx', 0)  # Get the batch index if it was saved
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, batch {batch_idx}")
    return model, optimizer, start_epoch, batch_idx

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Save checkpoint every half epoch
            if batch_idx % (len(train_loader) // 2) == 0:
                checkpoint_filename = f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth'
                save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint_filename)
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.cuda()
                labels = labels.cuda()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect labels and predictions for F1 score calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        
        # Print overall performance
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

        # Calculate and print F1 scores for each class
        all_labels = [train_dataset.index_to_label[label] for label in all_labels]
        all_preds = [train_dataset.index_to_label[pred] for pred in all_preds]
        report = classification_report(all_labels, all_preds, target_names=list(train_dataset.label_to_index.keys()), zero_division=1)
        print(f'Classification Report for Epoch {epoch+1}:\n{report}')

        # Save checkpoint at the end of each epoch
        checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(model, optimizer, epoch, 0, checkpoint_filename)

    # Save the trained model's state dict
    torch.save(model.state_dict(), 'intern_video_finetuned.pth')

# Check if a checkpoint is provided
start_epoch = 0
if args.checkpoint:
    custom_model, optimizer, start_epoch, _ = load_checkpoint(args.checkpoint, custom_model, optimizer)

# Run the training and evaluation
train_model(custom_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, start_epoch=start_epoch)
