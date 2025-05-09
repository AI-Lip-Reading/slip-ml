import argparse
import json
import logging
import os
import sys


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CTCLoss
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from jiwer import wer, cer

import wandb
import boto3
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize S3 client
s3_client = boto3.client("s3")


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
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

        # Calculate feature map dimensions after convolutions and pooling
        # This is an estimate and should be verified with actual forward pass
        h_out = img_h
        w_out = img_w
        
        # Apply conv1 and pool1 dimensions
        h_out = ((h_out + 2*2 - 5) // 2 + 1) // 2
        w_out = ((w_out + 2*2 - 5) // 2 + 1) // 2
        
        # Apply conv2 and pool2 dimensions
        h_out = ((h_out + 2*2 - 5) // 1 + 1) // 2
        w_out = ((w_out + 2*2 - 5) // 1 + 1) // 2
        
        # Apply conv3 and pool3 dimensions
        h_out = ((h_out + 2*1 - 3) // 1 + 1) // 2
        w_out = ((w_out + 2*1 - 3) // 1 + 1) // 2
        
        # Calculate estimated feature map size
        self.estimated_feature_size = 96 * h_out * w_out
        print(f"Estimated feature map size: {self.estimated_feature_size}")
        
        # GRU layers (actual size will be determined in forward pass)
        self.gru1 = nn.GRU(self.estimated_feature_size, 256, batch_first=True, bidirectional=True)
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

        # Reshape for RNN - DYNAMIC CALCULATION
        batch_size, channels, frames, height, width = x.size()
        # Log the actual dimensions to debug
        
        print(f"After conv layers - Channels: {channels}, Height: {height}, Width: {width}")
        
        # Calculate feature map size dynamically
        feature_map_size = channels * height * width
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, frames, channels, height, width)
        x = x.view(batch_size, frames, feature_map_size)  # Use the dynamically calculated size
        
        # GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Dense layer for character predictions
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        return x

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
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".npz")])
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
        # Load video frames from .npz file
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        with np.load(video_path, allow_pickle=True) as data:
            # Assuming the key for frames in the .npz file is 'frames'
            frames = data['frames'] 
        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, T, H, W)
        # Add in your __getitem__ method
        if torch.isnan(frames).any():
            print(f"NaN values found in video file: {self.video_files[idx]}")

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
        logger.info(f"Dataset saved to {save_path}")

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
        logger.info(f"Dataset loaded from {load_path}")
        return dataset


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

    if len(input_lengths) != len(label_lengths):
        logger.error(f"Batch size mismatch: input_lengths={len(input_lengths)}, label_lengths={len(label_lengths)}")
    if sum(label_lengths) != len(labels):
        logger.error(f"Label lengths sum mismatch: sum(label_lengths)={sum(label_lengths)}, len(labels)={len(labels)}")

    if label_lengths.sum() != labels.size(0):
        logger.error(f"Mismatch between sum of label_lengths and total labels: {label_lengths.sum()} vs {labels.size(0)}")

    return frames, labels, input_lengths, label_lengths


def setup_wandb(args):
    logger.info("Logging in to W&B")
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
    wandb.init(project='recreate-lipnet') #args.project_name)

def get_video_chunk_names(path):
    # extract video ID from the filename
    video_id = path.split('__')[0]
    chunk_name = path.split('__')[2].split('.')[0]
    return video_id, chunk_name


def check_data(data_dir):
    logger.info("Get train and validation data loaders")

    # make sure all videos and chunks are in both directories
    # if not remove the ones that are not in both directories
    video_files = sorted([f for f in os.listdir(data_dir + '/videos/') if f.endswith(".npz")])
    label_files = sorted([f for f in os.listdir(data_dir + '/labels/') if f.endswith(".json")])
    logger.info(f"number of video files: {len(video_files)}")
    logger.info(f"number of label files: {len(label_files)}")

    # Remove the ones that are not in both directories
    for video_file in video_files:
        video_id, chunk_name = get_video_chunk_names(video_file)
        should_be_label_file = f"{video_id}__text__{chunk_name}.json"
        if should_be_label_file not in label_files:
            logger.info(f"Removing video file {video_file} because it does not have a corresponding label file")
            os.remove(os.path.join(data_dir + '/videos/', video_file))
    for label_file in label_files:
        video_id, chunk_name = get_video_chunk_names(label_file)
        should_be_video_file = f"{video_id}__numpy__{chunk_name}.npz"
        if should_be_video_file not in video_files:
            logger.info(f"Removing label file {label_file} because it does not have a corresponding video file")
            os.remove(os.path.join(data_dir + '/labels/', label_file))

    logger.info(f"number of video files after removing: {len(os.listdir(data_dir + '/videos/'))}")
    logger.info(f"number of label files after removing: {len(os.listdir(data_dir + '/labels/'))}")

    if len(os.listdir(data_dir + '/videos/')) == 0 or len(os.listdir(data_dir + '/labels/')) == 0:
        logger.info("No video or label files found in the directories")
        return False
    # check if the number of video and label files is the same
    elif len(os.listdir(data_dir + '/videos/')) != len(os.listdir(data_dir + '/labels/')):
        logger.info("Number of video and label files is not the same")
        return False
    else:
        logger.info("Number of video and label files is the same")
        return True


def _get_train_valid_data_loader(train_data_dir, batch_size):

    if not check_data(train_data_dir):
        raise ValueError("Data is not in the correct format")

    # Create dataset and dataloader
    dataset = LipNetDataset(train_data_dir + '/videos/', train_data_dir + '/labels/')

    # Define dataset split ratios
    train_ratio = 0.9

    # Calculate the sizes of each split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # logger.info dataset sizes
    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    return train_loader, val_loader


def _get_test_data_loader(test_data_dir, batch_size):
    logger.info("Get test data loader")

    if not check_data(test_data_dir):
        raise ValueError("Data is not in the correct format")

    # Create dataset and dataloader
    dataset = LipNetDataset(test_data_dir + '/videos/', test_data_dir + '/labels/')


    # Create DataLoaders 
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


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


def decode_ctc(preds, idx_to_char, blank=0):
    """Decode CTC output using greedy decoding."""
    result = []
    prev = None
    for char_idx in preds:
        if char_idx != blank and char_idx != prev:
            if char_idx in idx_to_char:
                result.append(idx_to_char[char_idx])
        prev = char_idx
    return ''.join(result)


def collapse_sequence(seq, blank=0):
    result = []
    prev = None
    for char in seq:
        if char != blank and char != prev:
            result.append(char)
        prev = char
    return result


def validate_test(model, device, data_loader, ctc_loss, idx_to_char):
    model.eval()
    all_loss = 0.0
    cer_scores = []
    wer_scores = []
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels, input_lengths, label_lengths in data_loader:
            frames, labels, input_lengths, label_lengths = (
                frames.to(device),
                labels.to(device),
                input_lengths.to(device),
                label_lengths.to(device),
            )
            outputs = model(frames).permute(1, 0, 2)
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
            all_loss += loss.item()
            
            # Decode predictions
            _, preds = outputs.max(2)  # (T, batch_size)
            preds = preds.transpose(1, 0).cpu().numpy()  # (batch_size, T)
            offset = 0
            for i in range(frames.size(0)):
                pred_seq = decode_ctc(preds[i], idx_to_char)
                true_seq = ''.join(idx_to_char[idx] for idx in labels[offset:offset+label_lengths[i].item()] if idx in idx_to_char)
                offset += label_lengths[i].item()
                if true_seq:
                    cer_scores.append(cer(true_seq, pred_seq))
                    wer_scores.append(wer(true_seq, pred_seq))

                pred = collapse_sequence(preds[i], blank=0)  # Remove blanks/repeats
                true = labels[total:total+label_lengths[i]].cpu().numpy()
                if pred == true.tolist():
                    correct += 1
                total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = all_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0
    return avg_loss, avg_cer, avg_wer, accuracy



def train(args, device):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader = _get_train_valid_data_loader(args.train_data_dir, args.batch_size)
    char_to_idx = {char: idx for idx, char in enumerate(" abcdefghijklmnopqrstuvwxyz'.?!")} 
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    model = model_fn('', device)


    # After defining your optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # We want to minimize the loss
        factor=0.5,           # Multiply learning rate by this factor when reducing
        patience=2,           # Number of epochs with no improvement after which to reduce LR
        threshold=0.01,       # Threshold for measuring improvement
        threshold_mode='rel', # Threshold relative to best value
        cooldown=0,           # Number of epochs to wait before resuming normal operation
        min_lr=1e-6,          # Don't reduce LR below this value
        verbose=True          # Print message when LR is reduced
    )


    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    ctc_loss = CTCLoss(blank=0, zero_infinity=True)  # Handle extreme values
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        for frames, labels, input_lengths, label_lengths in train_loader:
            logger.debug(f"Batch sizes - frames: {frames.size(0)}, input_lengths: {input_lengths.size(0)}, labels: {labels.size(0)}, label_lengths: {label_lengths.size(0)}")
            logger.debug(f"Frames shape: {frames.shape}, Labels shape: {labels.shape}")
            logger.debug(f"Input lengths: {input_lengths}, Label lengths: {label_lengths}")
            
            # Skip batches with zero label lengths
            if (label_lengths == 0).any():
                logger.error(f"Zero label length detected in batch: {label_lengths.tolist()}, skipping")
                continue
            if (label_lengths < 10).any():
                logger.warning(f"Short label lengths detected: {label_lengths.tolist()}, proceeding with caution")
            
            if frames.isnan().any() or frames.isinf().any():
                logger.error(f"Invalid frames in batch, skipping")
                continue
            frames, labels, input_lengths, label_lengths = (
                frames.to(device),
                labels.to(device),
                input_lengths.to(device),
                label_lengths.to(device),
            )
            optimizer.zero_grad()
            outputs = model(frames).permute(1, 0, 2)
            logger.debug(f"Output min: {outputs.min().item()}, max: {outputs.max().item()}")
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.error(f"Invalid model outputs, skipping batch")
                continue
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
            logger.debug(f"Mid Training Loss: {loss.item()}")
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss: {loss.item()}, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logger.error(f"Invalid gradient detected in {name}, skipping update")
                    optimizer.zero_grad()
                    break
            else:
                optimizer.step()
                train_loss += loss.item()
                valid_batches += 1
        if valid_batches > 0:
            logger.info(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss / valid_batches:.4f}")


        val_loss, val_cer, val_wer, val_accuracy = validate_test(model, device, val_loader, ctc_loss, idx_to_char)

        # Update the learning rate based on validation loss
        scheduler.step(val_accuracy)
        
        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "learning_rate": current_lr
        })

        logger.info(f"""Epoch [{epoch + 1}/{args.epochs}], 
                    Train Loss: {train_loss / len(train_loader):.4f}, 
                    Validation Loss: {val_loss / len(val_loader):.4f},
                    Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f},
                    Validation Accuracy: {val_accuracy:.2f}%""")

    logger.info("Training complete!")
    return model, ctc_loss, idx_to_char


def model_fn(model_dir, device):
    frames_n = 90
    img_w = 100  # 1920//5
    img_h = 50  # 1080//5

    model = LipNet(img_c=3, img_w=img_w, img_h=img_h, frames_n=frames_n)
    model = torch.nn.DataParallel(model)

    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model to S3")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

    logger.info("Model saved to W&B")
    wandb.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="The name of the project to use for wandb logging",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test-data-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    setup_wandb(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ctc_loss, idx_to_char = train(args, device)

    test_loader = _get_test_data_loader(args.test_data_dir, args.batch_size)

    test_loss, test_cer, test_wer, test_accuracy = validate_test(model, device, test_loader, ctc_loss, idx_to_char)

    # Log metrics to W&B
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_cer": test_cer,
        "test_wer": test_wer
    })

    logger.info(f"""Test Loss: {test_loss:.4f}, 
                Test Accuracy: {test_accuracy:.2f}%, 
                Test CER: {test_cer:.4f}, 
                Test WER: {test_wer:.4f}""")

    save_model(model, args.model_dir)