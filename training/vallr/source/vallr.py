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
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize S3 client
s3_client = boto3.client("s3")

def get_video_chunk_names(path):
    video_id = path.split('__')[0]
    chunk_name = path.split('__')[2].split('.')[0]
    return video_id, chunk_name

# Positional embedding helpers
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = grid_size
    grid_w = grid_size
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, np.arange(grid_h), np.arange(grid_w))
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid_h, grid_w):
    assert embed_dim % 2 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (W, D/2)
    
    # Create position embeddings for each spatial position
    emb_h_expanded = np.repeat(emb_h, len(grid_w), axis=0)  # (H*W, D/2)
    emb_w_expanded = np.tile(emb_w, (len(grid_h), 1))  # (H*W, D/2)
    
    emb = np.concatenate([emb_h_expanded, emb_w_expanded], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class SpatioTemporalEmbeddingModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=(3, 5, 5), stride=(1, 1, 1), vit_hidden_size=768, patch_size=16, image_size=224, output_frames=4):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1, 2, 2)  # Maintain spatial size, pad temporal
        )
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.temporal_pool = nn.AdaptiveAvgPool3d((output_frames, image_size, image_size))  # Reduce to 4 frames
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2  # 196 patches
        self.num_patches = self.num_patches_per_frame * output_frames  # 784 patches
        self.projection = nn.Linear(out_channels * patch_size * patch_size, vit_hidden_size)  # Project to 768
        self.dropout = nn.Dropout(0.1)  # Regularization

    def forward(self, x):
        # Input: (B, T, C, H, W), e.g., (B, 25, 3, 224, 224)
        # Permute to (B, C, T, H, W) format for 3D Conv
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.conv3d(x)  # (B, C', T', H', W')
        x = self.bn3d(x)
        x = self.prelu(x)
        x = self.temporal_pool(x)  # (B, C', 4, H, W)
        
        # Reshape into patches
        B, C, T, H, W = x.shape
        x = x.view(B, C, T * H, W).unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, T*H//P, W//P, C, P, P)
        x = x.view(B, self.num_patches, -1)  # (B, 784, C*P*P)
        x = self.projection(x)  # (B, 784, 768)
        x = self.dropout(x)
        return x

class VALLRModel(nn.Module):
    def __init__(self, num_classes=40, num_frames=25, output_frames=4):
        super(VALLRModel, self).__init__()
        self.num_frames = num_frames
        self.output_frames = output_frames
        self.num_classes = num_classes
        
        # Add these two lines to define the variables
        self.image_size = 224
        self.patch_size = 16
        
        # 3D Spatio-Temporal Embedding Module
        self.st_module = SpatioTemporalEmbeddingModule(output_frames=output_frames)
        
        # ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        hidden_size = self.vit_config.hidden_size  # 768
        self.vit_encoder = self.vit.encoder
    
        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Calculate num_patches using self variables
        self.num_patches_per_frame = (self.image_size // self.patch_size) ** 2  # 196 patches for 224/16
        self.num_patches = self.num_patches_per_frame * output_frames  # 784 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, hidden_size))
        
        # CTC Head
        self.ctc_head = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
        # Initialize with sinusoidal positional embeddings
        self._init_pos_embed()

    def _init_pos_embed(self):
        # For a structure with 14×14 patches per frame and 4 frames
        # We create a grid of size (14×4)×14 = 56×14
        h_grid_size = int(math.sqrt(self.num_patches_per_frame)) * self.output_frames  # 14×4=56
        w_grid_size = int(math.sqrt(self.num_patches_per_frame))  # 14
        
        # Create a non-square positional embedding grid
        pos_embed = get_2d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1],
            np.arange(h_grid_size),
            np.arange(w_grid_size)
        )
        
        # Add CLS token embedding
        pos_embed = np.concatenate([np.zeros([1, self.pos_embed.shape[-1]]), pos_embed], axis=0)
        
        # Ensure the correct number of positions
        assert pos_embed.shape[0] == 1 + self.num_patches, f"Position embedding size mismatch: got {pos_embed.shape[0]}, expected {1 + self.num_patches}"
        
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        # Input: (B, T, C, H, W), e.g., (B, 25, 3, 224, 224)
        B = x.shape[0]
        x = self.st_module(x)  # (B, 784, 768)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 785, 768)
        
        # Add positional embeddings
        x = x + self.pos_embed  # (B, 785, 768)
        
        # Important: Apply layer norm before feeding to transformer
        # This mimics what happens in the ViT embeddings layer
        x = nn.LayerNorm(x.shape[-1], eps=1e-12).to(x.device)(x)
        
        # Pass through encoder directly - encoder expects sequence already embedded
        hidden_states = self.vit_encoder(x)[0]  # Extract last_hidden_state
            
        # CTC head: Use per-frame embeddings (skip [CLS])
        frame_embeddings = hidden_states[:, 1:, :]  # (B, 784, 768)
        frame_embeddings = frame_embeddings.view(B, self.output_frames, 196, 768)  # (B, 4, 196, 768)
        frame_embeddings = frame_embeddings.mean(dim=2)  # (B, 4, 768), average per frame
        logits = self.ctc_head(self.dropout(frame_embeddings))  # (B, 4, 40)
        logits = F.log_softmax(logits, dim=-1)
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

        # Ensure exactly 25 frames
        target_frames = 25
        if frame_count < 25:
            # Repeat last frame to pad
            while frame_count < target_frames:
                # Pad with last frame
                padding = frames[-1:].repeat(1, 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)
                frame_count += 1
            frame_count = 25
        elif frame_count > 25:
            # Uniform sampling
            indices = torch.linspace(0, frame_count-1, target_frames).long()
            frames = frames[indices]
            frame_count = 25
        elif frame_count != 25:
            logger.error(f"Unexpected frame count {frame_count} in {video_file}")
            raise ValueError(f"Expected 24, 25, or 30 frames, got {frame_count}")

        # Normalize frames
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frames = transform(frames)  # [25, 3, 224, 224]

        # Load labels
        with open(label_path, "r") as f:
            label_data = json.load(f)
        phoneme_indices = label_data["phoneme_indices"]
        if not all(0 <= idx < len(self.phoneme_vocab) for idx in phoneme_indices):
            logger.error(f"Invalid phoneme indices in {label_file}: {phoneme_indices}")
            raise ValueError(f"Phoneme indices must be in range [0, {len(self.phoneme_vocab)-1}]")
        labels = torch.tensor(phoneme_indices, dtype=torch.long)

        # Input length for CTC (4 output frames from 3D module)
        input_length = 4  # Fixed due to temporal pooling to 4 frames
        target_length = len(labels)

        return frames, labels, input_length, target_length

def collate_fn(batch):
    frames, phonemes, input_lengths, target_lengths = zip(*batch)
    frames = torch.stack(frames)  # [batch_size, 25, 3, 224, 224]
    max_phoneme_len = max(len(p) for p in phonemes)
    phonemes_padded = torch.zeros(len(phonemes), max_phoneme_len, dtype=torch.long)
    for i, p in enumerate(phonemes):
        phonemes_padded[i, :len(p)] = p
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return frames, phonemes_padded, input_lengths, target_lengths

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
        for frames, phonemes, input_lengths, target_lengths in data_loader:
            frames, phonemes, input_lengths, target_lengths = (
                frames.to(device), phonemes.to(device), 
                input_lengths.to(device), target_lengths.to(device)
            )
            try:
                outputs = model(frames).permute(1, 0, 2)  # (T=4, B, 40)
                loss = ctc_loss(outputs, phonemes, input_lengths, target_lengths)
                all_loss += loss.item()
            except Exception as e:
                logger.error(f"CTC loss error: {str(e)}, input_lengths {input_lengths}, outputs shape {outputs.shape if 'outputs' in locals() else 'N/A'}")
                continue

            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0).cpu().numpy()
            phonemes_cpu = phonemes.cpu().numpy()
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

    model = VALLRModel(num_classes=40, num_frames=25, output_frames=4).to(device)
    if is_distributed and use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model)

    # Sequential Training: Phase 1 - Train 3D module and CTC head
    logger.info("Starting Phase 1: Training 3D Spatio-Temporal Module and CTC Head")
    for param in model.module.vit.parameters():
        param.requires_grad = False  # Freeze ViT
    optimizer_phase1 = optim.Adam([
        {'params': model.module.st_module.parameters(), 'lr': args.lr},
        {'params': model.module.ctc_head.parameters(), 'lr': args.lr},
        {'params': model.module.cls_token, 'lr': args.lr},
        {'params': model.module.pos_embed, 'lr': args.lr}
    ], weight_decay=0.1)
    scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1, mode='min', factor=0.5, patience=2, threshold=0.01, verbose=True
    )
    ctc_loss = CTCLoss(blank=0, zero_infinity=True)

    phase1_epochs = args.epochs // 2  # Half epochs for Phase 1
    for epoch in range(phase1_epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        for frames, phonemes, input_lengths, target_lengths in train_loader:
            if (target_lengths == 0).any():
                logger.error(f"Zero target length detected, skipping")
                continue
            if frames.isnan().any() or frames.isinf().any():
                logger.error(f"Invalid frames, skipping")
                continue

            frames, phonemes, input_lengths, target_lengths = (
                frames.to(device), phonemes.to(device), 
                input_lengths.to(device), target_lengths.to(device)
            )
            optimizer_phase1.zero_grad()
            outputs = model(frames).permute(1, 0, 2)  # (T=4, B, 40)
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
                    optimizer_phase1.zero_grad()
                    break
            else:
                optimizer_phase1.step()
                train_loss += loss.item()
                valid_batches += 1

        if valid_batches > 0:
            logger.info(f"Phase 1 Epoch [{epoch + 1}/{phase1_epochs}], Train Loss: {train_loss / valid_batches:.4f}")

        val_loss, val_cer, val_wer, val_accuracy = validate_test(model, device, val_loader, ctc_loss, idx_to_phoneme)
        scheduler_phase1.step(val_loss)
        current_lr = optimizer_phase1.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1,
            "phase": 1,
            "train_loss": train_loss / valid_batches if valid_batches > 0 else 0.0,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "learning_rate": current_lr
        })

        logger.info(f"Phase 1 Epoch [{epoch + 1}/{phase1_epochs}], "
                    f"Train Loss: {train_loss / valid_batches:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%")

    # Sequential Training: Phase 2 - Fine-tune entire model
    logger.info("Starting Phase 2: Fine-tuning entire model")
    for param in model.module.vit.parameters():
        param.requires_grad = True  # Unfreeze ViT
    optimizer_phase2 = optim.Adam([
        {'params': model.module.st_module.parameters(), 'lr': args.lr},
        {'params': model.module.vit.parameters(), 'lr': args.lr * 0.1},  # Smaller LR for ViT
        {'params': model.module.ctc_head.parameters(), 'lr': args.lr},
        {'params': model.module.cls_token, 'lr': args.lr},
        {'params': model.module.pos_embed, 'lr': args.lr}
    ], weight_decay=0.1)
    scheduler_phase2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2, mode='min', factor=0.5, patience=2, threshold=0.01, verbose=True
    )

    phase2_epochs = args.epochs - phase1_epochs
    for epoch in range(phase2_epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        for frames, phonemes, input_lengths, target_lengths in train_loader:
            if (target_lengths == 0).any():
                logger.error(f"Zero target length detected, skipping")
                continue
            if frames.isnan().any() or frames.isinf().any():
                logger.error(f"Invalid frames, skipping")
                continue

            frames, phonemes, input_lengths, target_lengths = (
                frames.to(device), phonemes.to(device), 
                input_lengths.to(device), target_lengths.to(device)
            )
            optimizer_phase2.zero_grad()
            outputs = model(frames).permute(1, 0, 2)  # (T=4, B, 40)
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
                    optimizer_phase2.zero_grad()
                    break
            else:
                optimizer_phase2.step()
                train_loss += loss.item()
                valid_batches += 1

        if valid_batches > 0:
            logger.info(f"Phase 2 Epoch [{epoch + 1}/{phase2_epochs}], Train Loss: {train_loss / valid_batches:.4f}")

        val_loss, val_cer, val_wer, val_accuracy = validate_test(model, device, val_loader, ctc_loss, idx_to_phoneme)
        scheduler_phase2.step(val_loss)
        current_lr = optimizer_phase2.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1 + phase1_epochs,
            "phase": 2,
            "train_loss": train_loss / valid_batches if valid_batches > 0 else 0.0,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "learning_rate": current_lr
        })

        logger.info(f"Phase 2 Epoch [{epoch + 1}/{phase2_epochs}], "
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