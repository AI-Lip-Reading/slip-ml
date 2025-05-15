import os
import json
import tarfile
import logging
import boto3
import torch
import numpy as np
import cv2
import mediapipe as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# Initialize S3 client
s3_client = boto3.client("s3")

# Phoneme vocabulary
PHONEME_VOCAB = ['<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
                 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 
                 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 
                 'Y', 'Z', 'ZH']
IDX_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_VOCAB)}

class VALLRModel(torch.nn.Module):
    def __init__(self, hidden_size=512, num_phonemes=len(PHONEME_VOCAB)):
        super(VALLRModel, self).__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = torch.nn.Flatten(start_dim=2)
        self.backbone_fc = torch.nn.Linear(256 * 4 * 4, hidden_size)
        self.adapter = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(384, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(192, 48, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(48, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Conv1d(16, num_phonemes, kernel_size=3, padding=1)

    def forward(self, frames):
        batch_size, num_frames, c, h, w = frames.shape
        frames = frames.view(-1, c, h, w)
        features = self.backbone(frames)
        features = self.flatten(features)
        features = features.view(batch_size, num_frames, -1)
        features = self.backbone_fc(features.transpose(1, 2))
        features = self.adapter(features)
        logits = self.classifier(features)
        return logits

def detect_face_intervals(video_file):
    face_intervals = []
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1 / fps

        face_present = False
        start_time = None
        no_face_frames = 0
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            if results.detections:
                if not face_present:
                    start_time = frame_index * frame_duration
                    face_present = True
                no_face_frames = 0
            else:
                if face_present:
                    no_face_frames += 1
                    if no_face_frames >= fps * 1.5:
                        end_time = frame_index * frame_duration
                        face_intervals.append({"start": start_time, "end": end_time})
                        face_present = False
            frame_index += 1

        if face_present:
            end_time = frame_index * frame_duration
            face_intervals.append({"start": start_time, "end": end_time})

        cap.release()
    except Exception as e:
        logger.error(f"Error in detect_face_intervals: {e}")
        return []
    return face_intervals

def preprocess_video_to_frames(video_path, target_frames=75):
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_indices = []

        # Get face intervals
        face_intervals = detect_face_intervals(video_path)
        if not face_intervals:
            logger.warning("No face intervals detected, processing all frames")
            face_intervals = [{"start": 0, "end": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps}]

        # Process frames within face intervals
        for interval in face_intervals:
            start_frame = int(interval["start"] * fps)
            end_frame = int(interval["end"] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    face = frame[max(0, y):y+height, max(0, x):x+width]
                    if face.size > 0:
                        face = cv2.resize(face, (224, 224))
                        frames.append(face)
                        frame_indices.append(frame_idx)
                frame_idx += 1
        cap.release()

        # Standardize to 75 frames
        if not frames:
            logger.error("No valid face frames detected")
            return None
        frames_array = np.array(frames)
        if len(frames_array) > target_frames:
            indices = np.linspace(0, len(frames_array) - 1, target_frames, dtype=int)
            frames_array = frames_array[indices]
        elif len(frames_array) < target_frames:
            pad_length = target_frames - len(frames_array)
            frames_array = np.pad(frames_array, ((0, pad_length), (0, 0), (0, 0), (0, 0)), mode="edge")
        return frames_array[:target_frames]
    except Exception as e:
        logger.error(f"Error in preprocess_video_to_frames: {e}")
        return None

class InferenceDataset(Dataset):
    def __init__(self, input_data, is_npz=False, transform=None):
        if is_npz:
            self.frames = np.load(input_data)['frames']
        else:
            self.frames = preprocess_video_to_frames(input_data)
        if self.frames is None:
            raise ValueError("Failed to preprocess video or load .npz")
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = self.frames
        if self.transform:
            frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in frames])
        return frames

def load_model_from_s3(s3_bucket, s3_key, local_path, model_type="llama"):
    logger.info(f"Downloading {s3_key} from s3://{s3_bucket}")
    s3_client.download_file(s3_bucket, s3_key, local_path)
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(os.path.dirname(local_path))
    model_dir = os.path.splitext(local_path)[0]
    
    if model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            load_in_4bit=True,
            device_map={"": 0}
        )
        model = PeftModel.from_pretrained(model, model_dir)
        return model, tokenizer
    else:
        model = VALLRModel()
        checkpoint = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cuda:0")
        model.load_state_dict(checkpoint)
        model.to("cuda:0")
        return model, None

def predict_phonemes(vallr_model, frames):
    vallr_model.eval()
    with torch.no_grad():
        logits = vallr_model(frames.to("cuda:0"))
        phoneme_indices = torch.argmax(logits, dim=1).cpu().numpy()
    phoneme_sequence = " ".join([IDX_TO_PHONEME[idx] for idx in phoneme_indices[0] if idx != 0])
    return phoneme_sequence

def predict_text(llama_model, tokenizer, phoneme_sequence):
    llama_model.eval()
    input_text = f"{phoneme_sequence} ->"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = llama_model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split('->', 1)[-1].strip() if '->' in generated_text else generated_text

def end_to_end_inference(vallr_model, llama_model, tokenizer, input_data, is_npz=False):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = InferenceDataset(input_data, is_npz=is_npz, transform=transform)
    frames = dataset[0].unsqueeze(0).to("cuda:0")
    phoneme_sequence = predict_phonemes(vallr_model, frames)
    text = predict_text(llama_model, tokenizer, phoneme_sequence)
    return phoneme_sequence, text

if __name__ == "__main__":
    s3_bucket = "slip-ml"
    vallr_s3_key = "models/vallr_model.tar.gz"
    llama_s3_key = "models/llama_model.tar.gz"
    local_vallr_path = "/tmp/vallr_model.tar.gz"
    local_llama_path = "/tmp/llama_model.tar.gz"
    input_file = "path/to/test_video.mp4"  # Or .npz for preprocessed

    vallr_model, _ = load_model_from_s3(s3_bucket, vallr_s3_key, local_vallr_path, model_type="vallr")
    llama_model, tokenizer = load_model_from_s3(s3_bucket, llama_s3_key, local_llama_path, model_type="llama")

    phonemes, text = end_to_end_inference(vallr_model, llama_model, tokenizer, input_file, is_npz=input_file.endswith(".npz"))
    logger.info(f"Phonemes: {phonemes}")
    logger.info(f"Generated Text: {text}")