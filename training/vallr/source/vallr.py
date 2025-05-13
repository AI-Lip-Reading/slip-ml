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
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
import numpy as np
from jiwer import wer, cer
import wandb
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize S3 client
s3_client = boto3.client("s3")

def get_video_chunk_names(path):
    video_id = path.split('__')[0]
    chunk_name = path.split('__')[2].split('.')[0]
    return video_id, chunk_name

class VALLRModel(nn.Module):
    def __init__(self, num_classes=40, max_frames=75):
        super(VALLRModel, self).__init__()
        self.max_frames = max_frames
        self.num_classes = num_classes

        # Visual Feature Extractor: ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        hidden_size = self.vit_config.hidden_size  # 768

        # Adapter Network (Table 1 in paper)
        self.adapter = nn.Sequential(
            nn.Conv1d(hidden_size, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 384, T/2)
            nn.Conv1d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 192, T/4)
            nn.Conv1d(192, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(9),  # Ensure T=9
            nn.Conv1d(48, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=3),  # (B, 48, T/12)
            #nn.Conv1d(48, 16, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2),  # (B, 16, T/24)
        )
        self.adapter_norm = nn.LayerNorm(16)
        self.adapter_dropout = nn.Dropout(0.1)

        # CTC Head
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x, frame_lengths):
        # x: [batch_size, max_frames, channels=3, height=224, width=224]
        batch_size, max_frames, channels, height, width = x.size()
        #logger.debug(f"Input shape to forward: {x.shape}")

        # Reshape for ViT
        x = x.view(batch_size * max_frames, channels, height, width)  # [batch_size * max_frames, 3, 224, 224]
        #logger.debug(f"After reshape for ViT: {x.shape}")

        # ViT forward pass
        vit_outputs = self.vit(pixel_values=x).last_hidden_state  # [batch_size * max_frames, 197, 768]
        vit_features = vit_outputs[:, 0, :]  # CLS token: [batch_size * max_frames, 768]
        vit_features = vit_features.view(batch_size, max_frames, -1)  # [batch_size, max_frames, 768]

        # Mask padded frames
        mask = torch.arange(max_frames, device=x.device)[None, :] < frame_lengths[:, None]  # [batch_size, max_frames]
        vit_features = vit_features * mask.unsqueeze(-1).float()

        vit_features = vit_features.permute(0, 2, 1)  # [batch_size, 768, max_frames]
        #logger.debug(f"ViT features shape: {vit_features.shape}")

        # Adapter network with channel validation
        x = vit_features
        expected_channels = [768, 384, 384, 384, 192, 192, 192, 48, 48, 48, 16, 16]
        for i, layer in enumerate(self.adapter):
            if isinstance(layer, nn.Conv1d):
                actual_channels = x.size(1)
                expected = expected_channels[i]
                if actual_channels != expected:
                    logger.error(f"Channel mismatch at layer {i} (Conv1d): expected {expected}, got {actual_channels}")
                    raise RuntimeError(f"Channel mismatch in Conv1d at layer {i}")
            x = layer(x)
            #logger.debug(f"Adapter layer {i} ({layer.__class__.__name__}) output shape: {x.shape}")
            if (isinstance(layer, nn.MaxPool1d) or isinstance(layer, nn.AdaptiveAvgPool1d)) and x.size(2) == 0:
                logger.error(f"Zero output size after layer {i}, input shape: {vit_features.shape}, frame_lengths: {frame_lengths}")
                raise RuntimeError(f"Zero output size in MaxPool1d at layer {i}")

        # Adapter network
        adapter_out = x  # [batch_size, 16, reduced_frames]
        adapter_out = self.adapter_norm(adapter_out.permute(0, 2, 1)).permute(0, 2, 1)
        adapter_out = self.adapter_dropout(adapter_out)

        # CTC head
        logits = self.fc(adapter_out.permute(0, 2, 1))  # [batch_size, reduced_frames, num_classes]
        logits = F.log_softmax(logits, dim=-1)

        #logger.debug(f"Adapter output shape: {adapter_out.shape}")
        #logger.debug(f"Logits shape: {logits.shape}")

        return logits

class VALLRDataset(Dataset):
    def __init__(self, video_dir, label_dir):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".npz")])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
        self.phoneme_vocab = ['<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
                              'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 
                              'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 
                              'Y', 'Z', 'ZH']
        self.phoneme_to_index = {p: i for i, p in enumerate(self.phoneme_vocab)}

        # Validate video-label pairing
        self.valid_pairs = []
        for video_file in self.video_files:
            video_id, chunk_name = get_video_chunk_names(video_file)
            label_file = f"{video_id}__text__{chunk_name}.json"
            if label_file in self.label_files:
                self.valid_pairs.append((video_file, label_file))
            else:
                logger.warning(f"No label for video {video_file}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        video_file, label_file = self.valid_pairs[idx]
        video_path = os.path.join(self.video_dir, video_file)
        label_path = os.path.join(self.label_dir, label_file)

        # Load video frames
        with np.load(video_path, allow_pickle=True) as data:
            frames = data['frames']  # [T, H=224, W=224, C=3]
        frame_count = frames.shape[0]
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # [T, C=3, H=224, W=224]
        #logger.debug(f"Video {video_file}: frames shape {frames.shape}, frame_count {frame_count}")

        min_frames = 16
        if frame_count < min_frames:
            logger.warning(f"Low frame count {frame_count} in {video_file}, padding to {min_frames}")
            padding = torch.zeros(min_frames - frame_count, frames.size(1), frames.size(2), frames.size(3))
            frames = torch.cat([frames, padding], dim=0)
            frame_count = min_frames


        # Pad to max_frames (75)
        max_frames = 75
        if frame_count < max_frames:
            padding = torch.zeros(max_frames - frame_count, frames.size(1), frames.size(2), frames.size(3))
            frames = torch.cat([frames, padding], dim=0)
        elif frame_count > max_frames:
            frames = frames[:max_frames]
            frame_count = max_frames

        # Normalize frames
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frames = transform(frames)  # [max_frames, 3, 224, 224]

        # Load labels
        with open(label_path, "r") as f:
            label_data = json.load(f)
        labels = torch.tensor(label_data["phoneme_indices"], dtype=torch.long)
        phoneme_indices = label_data["phoneme_indices"]
        if not all(0 <= idx < len(self.phoneme_vocab) for idx in phoneme_indices):
            logger.error(f"Invalid phoneme indices in {label_file}: {phoneme_indices}")
            raise ValueError(f"Phoneme indices must be in range [0, {len(self.phoneme_vocab)-1}]")

        # Estimate input_length after adapter (approximate reduction per Table 1)
        input_length = max(1, min(9, frame_count // 4)) #input_length = max(1, frame_count // 24) #max(1, min(3, frame_count // 24))
        target_length = len(labels)

        #logger.debug(f"Input length: {input_length}, Target length: {target_length}, Frame count: {frame_count}")

        return frames, labels, input_length, target_length, frame_count


def collate_fn(batch):
    frames, phonemes, input_lengths, target_lengths, frame_counts = zip(*batch)
    max_frames = max(frame_counts)
    frames = torch.stack([f[:max_frames] for f in frames])  # [batch_size, max_frames, 3, 224, 224]
    max_phoneme_len = max(len(p) for p in phonemes)
    phonemes_padded = torch.zeros(len(phonemes), max_phoneme_len, dtype=torch.long)
    for i, p in enumerate(phonemes):
        phonemes_padded[i, :len(p)] = p
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    frame_lengths = torch.tensor(frame_counts, dtype=torch.long)
    #logger.debug(f"Collate output: frames shape {frames.shape}, phonemes shape {phonemes_padded.shape}")
    return frames, phonemes_padded, input_lengths, target_lengths, frame_lengths

def setup_wandb(args):
    logger.info("Logging in to W&B")
    secret_name = "wandb"
    region_name = "us-east-1"
    session = boto3.session.Session()
    secretsmanager = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = secretsmanager.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        api_key = json.loads(secret)["API_KEY"]
        wandb.login(key=api_key)
        wandb.init(project=args.project_name or "vallr-video-to-phoneme")
    except Exception as e:
        logger.error(f"Failed to setup W&B: {str(e)}")

def check_data(data_dir):
    logger.info("Checking train and validation data")
    video_dir = os.path.join(data_dir, "face")
    label_dir = os.path.join(data_dir, "text")
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".npz")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
    logger.info(f"Number of video files: {len(video_files)}")
    logger.info(f"Number of label files: {len(label_files)}")

    valid_videos = set()
    valid_labels = set()
    for video_file in video_files:
        video_id, chunk_name = get_video_chunk_names(video_file)
        label_file = f"{video_id}__text__{chunk_name}.json"
        if label_file in label_files:
            valid_videos.add(video_file)
            valid_labels.add(label_file)
        else:
            os.remove(os.path.join(video_dir, video_file))
    for label_file in label_files:
        if label_file not in valid_labels:
            os.remove(os.path.join(label_dir, label_file))

    remaining_videos = len(os.listdir(video_dir))
    remaining_labels = len(os.listdir(label_dir))
    logger.info(f"Number of video files after cleanup: {remaining_videos}")
    logger.info(f"Number of label files after cleanup: {remaining_labels}")

    if remaining_videos == 0 or remaining_labels == 0:
        logger.error("No video or label files found")
        return False
    if remaining_videos != remaining_labels:
        logger.error("Number of video and label files mismatch")
        return False
    return True

def _get_train_valid_data_loader(data_dir, batch_size):
    video_dir = os.path.join(data_dir, "face")
    label_dir = os.path.join(data_dir, "text")
    if not check_data(data_dir):
        raise ValueError("Data is not in the correct format")

    dataset = VALLRDataset(video_dir, label_dir)
    train_ratio = 0.9
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_loader, val_loader

def _get_test_data_loader(data_dir, batch_size):
    video_dir = os.path.join(data_dir, "face")
    label_dir = os.path.join(data_dir, "text")
    if not check_data(data_dir):
        raise ValueError("Data is not in the correct format")

    dataset = VALLRDataset(video_dir, label_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

def decode_ctc(preds, idx_to_phoneme, blank=0):
    result = []
    prev = None
    for idx in preds:
        if idx != blank and idx != prev:
            if idx in idx_to_phoneme:
                result.append(idx_to_phoneme[idx])
        prev = idx
    return ' '.join(result)

def validate_test(model, device, data_loader, ctc_loss, idx_to_phoneme):
    model.eval()
    all_loss = 0.0
    cer_scores = []
    wer_scores = []
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, phonemes, input_lengths, target_lengths, frame_lengths in data_loader:
            frames, phonemes, input_lengths, target_lengths, frame_lengths = (
                frames.to(device), phonemes.to(device), 
                input_lengths.to(device), target_lengths.to(device), frame_lengths.to(device)
            )
            try:
                outputs = model(frames, frame_lengths).permute(1, 0, 2)
                logger.debug(f"CTC inputs: outputs shape {outputs.shape}, input_lengths {input_lengths}")
                loss = ctc_loss(outputs, phonemes, input_lengths, target_lengths)
                all_loss += loss.item()
            except Exception as e:
                logger.error(f"CTC loss error: {str(e)}, input_lengths {input_lengths}, outputs shape {outputs.shape if 'outputs' in locals() else 'N/A'}")
                continue

            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0).cpu().numpy()
            phonemes_cpu = phonemes.cpu().numpy()  # Move phonemes to CPU for decoding
            #logger.debug(f"Phonemes device: {phonemes.device}, Phonemes CPU shape: {phonemes_cpu.shape}")
            for i in range(frames.size(0)):
                pred_seq = decode_ctc(preds[i][:input_lengths[i]], idx_to_phoneme)
                true_seq = ' '.join(idx_to_phoneme[idx] for idx in phonemes_cpu[i, :target_lengths[i].item()])
                if true_seq and pred_seq:
                    cer_scores.append(cer(true_seq, pred_seq))
                    wer_scores.append(wer(true_seq, pred_seq))
                if list(preds[i][:input_lengths[i]]) == list(phonemes_cpu[i, :target_lengths[i].item()]):
                    correct += 1
                total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = all_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0
    return avg_loss, avg_cer, avg_wer, accuracy

def train(args, device):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.info(f"Distributed training - {is_distributed}")
    use_cuda = args.num_gpus > 0
    logger.info(f"Number of GPUs available - {args.num_gpus}")

    if is_distributed:
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(f"Initialized distributed environment: '{args.backend}' backend, rank {host_rank}")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader = _get_train_valid_data_loader(args.train_data_dir, args.batch_size)
    phoneme_vocab = ['<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
                     'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 
                     'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 
                     'Y', 'Z', 'ZH']
    idx_to_phoneme = {i: p for i, p in enumerate(phoneme_vocab)}

    model = VALLRModel(num_classes=40, max_frames=75).to(device)
    if is_distributed and use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=0.01, verbose=True
    )
    ctc_loss = CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        for frames, phonemes, input_lengths, target_lengths, frame_lengths in train_loader:
            #logger.info(f"Frames shape: {frames.shape}, Phonemes shape: {phonemes.shape}")
            if (target_lengths == 0).any():
                logger.error(f"Zero target length detected, skipping")
                continue
            if frames.isnan().any() or frames.isinf().any():
                logger.error(f"Invalid frames, skipping")
                continue

            frames, phonemes, input_lengths, target_lengths, frame_lengths = (
                frames.to(device), phonemes.to(device), 
                input_lengths.to(device), target_lengths.to(device), frame_lengths.to(device)
            )
            optimizer.zero_grad()
            outputs = model(frames, frame_lengths).permute(1, 0, 2)  # [T, batch_size, num_classes]
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.error(f"Invalid outputs, skipping")
                continue

            loss = ctc_loss(outputs, phonemes, input_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss: {loss.item()}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logger.error(f"Invalid gradient in {name}, skipping")
                    optimizer.zero_grad()
                    break
            else:
                optimizer.step()
                train_loss += loss.item()
                valid_batches += 1

        if valid_batches > 0:
            logger.info(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss / valid_batches:.4f}")

        val_loss, val_cer, val_wer, val_accuracy = validate_test(model, device, val_loader, ctc_loss, idx_to_phoneme)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / valid_batches if valid_batches > 0 else 0.0,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "learning_rate": current_lr
        })

        logger.info(f"Epoch [{epoch + 1}/{args.epochs}], "
                    f"Train Loss: {train_loss / valid_batches:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%")

    logger.info("Training complete!")
    return model, ctc_loss, idx_to_phoneme

def save_model(model, model_dir):
    logger.info("Saving the model to S3")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    wandb.save(path)
    logger.info("Model saved to W&B")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend")
    parser.add_argument("--project-name", type=str, default="vallr-video-to-phoneme", help="W&B project name")
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test-data-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    args = parser.parse_args()

    setup_wandb(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ctc_loss, idx_to_phoneme = train(args, device)

    test_loader = _get_test_data_loader(args.test_data_dir, args.batch_size)
    test_loss, test_cer, test_wer, test_accuracy = validate_test(model, device, test_loader, ctc_loss, idx_to_phoneme)

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_cer": test_cer,
        "test_wer": test_wer
    })

    logger.info(f"Test Loss: {test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.2f}%, "
                f"Test CER: {test_cer:.4f}, "
                f"Test WER: {test_wer:.4f}")

    save_model(model, args.model_dir)