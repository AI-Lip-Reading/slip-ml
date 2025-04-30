# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import random_split

import cv2
import os
import numpy as np
import json

import boto3
import shutil
import wandb

# %%
print("Logging in to W&B")
secret_name = "wandb"
region_name = "us-east-1"

# Create a Secrets Manager client
session = boto3.session.Session()
secretsmanager = session.client(service_name='secretsmanager', region_name=region_name)

get_secret_value_response = secretsmanager.get_secret_value(SecretId=secret_name)

secret = get_secret_value_response['SecretString']
api_key = json.loads(secret)["API_KEY"]
wandb.login(key=api_key)

# Initialize W&B project
wandb.init(project="lipnet-test")

# %%
s3_client = boto3.client('s3')
bucket = 'slip-ml'

# %%
def download_all_files_from_s3(bucket_name):
    """
    List all files in the specified S3 bucket and download them to a local directory.

    Args:
        bucket_name (str): Name of the S3 bucket.
        local_dir (str): Local directory to save the downloaded files.
    """
    local_dir = f"{os.getcwd()}/input_data"
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # loop through folders
    for folder in ['video/', 'text/']:
        try:
            # List all objects in the bucket
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
            if 'Contents' not in response:
                print(f"No files found in bucket {bucket_name}.")
                return

            for obj in response['Contents'][:20]:
                file_key = obj['Key']
                local_file_path = os.path.join(local_dir, file_key)

                # Ensure subdirectories are created
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                s3_client.download_file(bucket_name, file_key, local_file_path)

            print("All files downloaded successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

# %%
def preprocess_video(video_path, img_w, img_h, frames_n):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count < frames_n:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame
        frame = cv2.resize(frame, (img_w, img_h))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        frame = frame / 255.0
        frames.append(frame)
        frame_count += 1

    cap.release()

    # Pad with black frames if the video has fewer than `frames_n` frames
    while len(frames) < frames_n:
        frames.append(np.zeros((img_h, img_w, 3)))

    # Convert to numpy array
    frames = np.array(frames)
    return frames

# %%
def text_to_labels(text_path, char_to_idx):
    # read in json file
    with open(text_path, 'r') as f:
        data = json.load(f)
    # Extract the text from the JSON data
    text = data['text']

    return [char_to_idx[char] for char in text if char in char_to_idx], text

# Example character-to-index mapping
char_to_idx = {char: idx for idx, char in enumerate(" abcdefghijklmnopqrstuvwxyz'.?!")}

# %%
def preprocess_chunked_data(video_dir, text_dir, output_dir, char_to_idx, img_w, img_h, frames_n):
    """
    Preprocess all video and text chunks in the specified directories.

    Args:
        video_dir (str): Directory containing video chunks.
        text_dir (str): Directory containing text chunks.
        output_dir (str): Directory to save preprocessed data.
        char_to_idx (dict): Character-to-index mapping.
        img_w (int): Width to resize video frames.
        img_h (int): Height to resize video frames.
        frames_n (int): Number of frames to extract per video chunk.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all video and label files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    label_files = sorted([f for f in os.listdir(text_dir) if f.endswith(".json")])

    for video_file in video_files:
        # Extract video ID and chunk number from the filename
        video_id, chunk_number = video_file.split("__video__")
        chunk_number = chunk_number.split(".")[0]

        # Find the corresponding label file
        label_file = f"{video_id}__text__{chunk_number}.json"
        if label_file not in label_files:
            print(f"Warning: No matching label file for {video_file}")
            continue

        # Full paths for video and label files
        video_path = os.path.join(video_dir, video_file)
        label_path = os.path.join(text_dir, label_file)

        # Preprocess video
        frames = preprocess_video(video_path, img_w, img_h, frames_n)

        # Convert text to labels
        labels, text = text_to_labels(label_path, char_to_idx)

        # Save preprocessed video as .npy file
        chunk_id = f"{video_id}__chunk__{chunk_number}"
        video_output_path = os.path.join(output_dir, f"video/{chunk_id}.npy")
        os.makedirs(output_dir + 'video/', exist_ok=True)
        np.save(video_output_path, frames)

        # Save labels as .json file
        label_output_path = os.path.join(output_dir, f"text/{chunk_id}.json")
        os.makedirs(output_dir + 'text/', exist_ok=True)
        with open(label_output_path, "w") as f:
            json.dump({"text": text, "labels": labels}, f)

    print("Preprocessing complete. Data saved to:", output_dir)

# %%
class LipNetDataset(Dataset):
    def __init__(self, video_dir, label_dir):
        """
        Initialize the LipNetDataset.

        Args:
            video_dir (str): Directory containing preprocessed video `.npy` files.
            label_dir (str): Directory containing preprocessed label `.json` files.
        """
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".npy")])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])

        # Ensure the number of video and label files match
        assert len(self.video_files) == len(self.label_files), "Mismatch between video and label files."

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (frames, labels), where:
                - frames (torch.Tensor): Preprocessed video frames of shape (C, T, H, W).
                - labels (torch.Tensor): Corresponding label sequence as a tensor of integers.
        """
        # Load video frames
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        frames = np.load(video_path, allow_pickle=True)
        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, T, H, W)

        # Load labels
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, "r") as f:
            label_data = json.load(f)
        labels = torch.tensor(label_data["labels"], dtype=torch.long)

        return frames, labels

    def save(self, save_path):
        """
        Save the dataset object for later use.

        Args:
            save_path (str): Path to save the dataset object.
        """
        torch.save(self, save_path)
        print(f"Dataset saved to {save_path}")

    @staticmethod
    def load(load_path):
        """
        Load a saved dataset object.

        Args:
            load_path (str): Path to the saved dataset object.

        Returns:
            LipNetDataset: Loaded dataset object.
        """
        dataset = torch.load(load_path)
        print(f"Dataset loaded from {load_path}")
        return dataset

# %%
class LipNet(nn.Module):
    def __init__(self, img_c, img_w, img_h, frames_n, output_size=31):
        super(LipNet, self).__init__()
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.output_size = output_size

        # First 3D Convolutional Block
        self.conv1 = nn.Conv3d(img_c, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout(0.5)

        # Second 3D Convolutional Block
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout2 = nn.Dropout(0.5)

        # Third 3D Convolutional Block
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout3 = nn.Dropout(0.5)

        # GRU layers
        feature_map_size = 1728 # To dynamically calculated flattened size: 96 * (img_w // 8) * (img_h // 8)
        self.gru1 = nn.GRU(feature_map_size, 256, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, batch_first=True, bidirectional=True)

        # Dense layer for character predictions
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        # First 3D Convolutional Block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second 3D Convolutional Block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third 3D Convolutional Block
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Reshape for RNN
        batch_size, channels, frames, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, frames, channels, height, width)
        x = x.view(batch_size, frames, -1)  # Flatten height and width

        # GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Dense layer for character predictions
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        return x

# %%
def collate_fn(batch):
    """
    Collate function to handle variable-length sequences in a batch.

    Args:
        batch (list): List of tuples (frames, labels).

    Returns:
        tuple: (frames, labels, input_lengths, label_lengths)
    """
    frames, labels = zip(*batch)

    # Stack frames into a tensor (batch_size, C, T, H, W)
    frames = torch.stack(frames)

    # Compute input lengths (number of frames per video)
    input_lengths = torch.tensor([frames.size(2)] * len(frames), dtype=torch.long)

    # Compute label lengths (number of characters per label sequence)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # Concatenate labels into a single tensor
    labels = torch.cat(labels)

    return frames, labels, input_lengths, label_lengths

# %%
def calculate_accuracy(predictions, labels, label_lengths):
    """
    Calculate accuracy by comparing predictions with ground truth labels.

    Args:
        predictions (torch.Tensor): Predicted sequences (T, batch_size, output_size).
        labels (torch.Tensor): Ground truth labels (concatenated for the batch).
        label_lengths (torch.Tensor): Lengths of each label sequence in the batch.

    Returns:
        float: Accuracy as a percentage.
    """
    # Decode predictions (get the index of the max log-probability for each time step)
    predicted_indices = torch.argmax(predictions, dim=-1).permute(1, 0)  # (batch_size, T)

    # Remove blank tokens (assume blank token index is 0)
    blank_token = 0
    decoded_predictions = []
    for pred in predicted_indices:
        decoded_sequence = []
        prev_token = None
        for token in pred:
            if token != blank_token and token != prev_token:  # Remove blanks and duplicates
                decoded_sequence.append(token.item())
            prev_token = token
        decoded_predictions.append(decoded_sequence)

    # Compare with ground truth labels
    correct = 0
    total = 0
    start_idx = 0
    for i, length in enumerate(label_lengths):
        ground_truth = labels[start_idx:start_idx + length].tolist()
        start_idx += length
        if decoded_predictions[i] == ground_truth:
            correct += 1
        total += 1

    return (correct / total) * 100 if total > 0 else 0.0

# %%
# Training loop with validation
def train_lipnet_with_validation(model, train_loader, val_loader, epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    ctc_loss = CTCLoss(blank=0)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for frames, labels, input_lengths, label_lengths in train_loader:
            frames, labels, input_lengths, label_lengths = (
                frames.to(device),
                labels.to(device),
                input_lengths.to(device),
                label_lengths.to(device),
            )
            optimizer.zero_grad()
            outputs = model(frames).permute(1, 0, 2)  # (T, batch_size, output_size)
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, labels, input_lengths, label_lengths in val_loader:
                frames, labels, input_lengths, label_lengths = (
                    frames.to(device),
                    labels.to(device),
                    input_lengths.to(device),
                    label_lengths.to(device),
                )
                outputs = model(frames).permute(1, 0, 2)
                loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()

                # Calculate accuracy
                val_accuracy += calculate_accuracy(outputs, labels, label_lengths)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_accuracy,
        })

        val_accuracy /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print("Training complete!")
    return model

# %%
# Testing loop
def test_lipnet(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    ctc_loss = CTCLoss(blank=0)
    with torch.no_grad():
        for frames, labels, input_lengths, label_lengths in test_loader:
            frames, labels, input_lengths, label_lengths = (
                frames.to(device),
                labels.to(device),
                input_lengths.to(device),
                label_lengths.to(device),
            )
            outputs = model(frames).permute(1, 0, 2)
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
            test_loss += loss.item()

            # Calculate accuracy
            test_accuracy += calculate_accuracy(outputs, labels, label_lengths)

    test_accuracy /= len(test_loader)

    wandb.log({
        "test_loss": test_loss / len(test_loader),
        "test_accuracy": test_accuracy,
    })

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

# %%
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# %%
download_all_files_from_s3(bucket)

# %%
def calculate_average_frames(video_dir):
    """
    Calculate the average number of frames across all videos in the directory.

    Args:
        video_dir (str): Directory containing video files.

    Returns:
        int: Average number of frames (rounded to the nearest integer).
    """
    total_frames = 0
    video_count = 0

    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):  # Ensure it's a video file
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
            total_frames += frame_count
            video_count += 1
            cap.release()

    if video_count == 0:
        raise ValueError("No video files found in the directory.")

    return round(total_frames / video_count)  # Return the average number of frames

# %%
# Calculate the average number of frames
input_video_dir = f"{os.getcwd()}/input_data/video/"
frames_n = calculate_average_frames(input_video_dir)

print(f"Calculated average frames per video: {frames_n}")

# Set other preprocessing parameters
img_w = 1920//3
img_h = 1080//3

# %%
input_video_dir = f"{os.getcwd()}/input_data/video/"
input_text_dir = f"{os.getcwd()}/input_data/text/"
output_dir = f"{os.getcwd()}/preprocessed_data/"

print("Preprocessing data...")
preprocess_chunked_data(input_video_dir, input_text_dir, output_dir, char_to_idx, img_w=img_w, img_h=img_h, frames_n=frames_n)

# %%
# delete input data
shutil.rmtree(f"{os.getcwd()}/input_data")

# %%
print("Preparing dataset for training...")
# Define dataset split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Load the dataset
dataset = LipNetDataset(video_dir=f"{os.getcwd()}/preprocessed_data/video/",
                        label_dir=f"{os.getcwd()}/preprocessed_data/text/")

# Calculate the sizes of each split
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size  # Ensure all samples are used

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Example: Print dataset sizes
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# %%
print("Initializing model...")
# Initialize the model
model = LipNet(img_c=3, img_w=img_w, img_h=img_h, frames_n=frames_n, output_size=len(char_to_idx))

print("Training model...")
# Train the model
model = train_lipnet_with_validation(model, train_loader, val_loader, epochs=10, device=device)

# %%
print("Testing model...")
test_lipnet(model, test_loader, device=device)

# %%
# Save model weights to W&B
print("Saving model weights to W&B...")
torch.save(model.state_dict(), "lipnet_model.pth")
wandb.save("lipnet_model.pth")

# %%



