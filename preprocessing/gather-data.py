
import os
import pronouncing
import mediapipe as mp
import cv2
import json
import boto3
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import math
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import argparse
import random


region_name = "us-east-1"

s3_client = boto3.client('s3')
bucket = 'slip-ml'


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
torch_dtype = torch.float32


model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def weighted_train_test_split():
    """
    Randomly returns 'train' or 'test' with a 90/10 weighted probability.

    Returns:
        str: 'train' (90% probability) or 'test' (10% probability).
    """
    return random.choices(['train', 'test'], weights=[90, 10], k=1)[0]

def upload_to_s3(local_file, s3_folder):
    s3_file = f"{s3_folder}/{local_file}"
    try:
        s3_client.upload_file(local_file, bucket, s3_file)
        print(f"Upload Successful: {local_file} -> {s3_file}")
    except FileNotFoundError:
        print(f"The file was not found: {local_file}")
    except Exception as e:
        print(f"An error occurred: {e}")



def detect_face_intervals(video_file):
    """
    Detects intervals in a video where a face is present, with a 1-second buffer for ending intervals.

    Args:
        video_path (str): Path to the input video.

    Returns:
        list: A list of dictionaries with 'start' and 'end' times for each interval where a face is detected.
    """
    face_intervals = []
    try:
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        frame_duration = 1 / fps  # Duration of each frame in seconds

        face_present = False
        start_time = None
        no_face_frames = 0  # Counter for frames without a face

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # Check if a face is detected
            if results.detections:
                if not face_present:
                    # Start a new interval
                    start_time = frame_index * frame_duration
                    face_present = True
                no_face_frames = 0  # Reset no-face counter
            else:
                if face_present:
                    no_face_frames += 1
                    # If no face is detected for 1 second, end the interval
                    if no_face_frames >= fps*1.5:
                        end_time = frame_index * frame_duration
                        face_intervals.append({"start": start_time, "end": end_time})
                        face_present = False

            frame_index += 1

        # Handle the case where the video ends while a face is still present
        if face_present:
            end_time = frame_index * frame_duration
            face_intervals.append({"start": start_time, "end": end_time})

        # Release resources
        cap.release()

    except Exception as e:
        print(f"An error occurred during face detection: {e}")
        return []
    
    return face_intervals


def chunk_video(video_file, face_intervals):
    """
    Splits the video into chunks based on detected face intervals and further splits chunks longer than 3 seconds.

    Args:
        video_id (str): The ID of the video.
        face_intervals (list): List of dictionaries with 'start' and 'end' times for each interval.

    Returns:
        list: List of paths to the created video chunks and sub-chunks.
    """
    chunks = []
    video_id = video_file.split('.')[0]  # Remove file extension
    try:
        # Load the video
        video = VideoFileClip(video_file)

        for i, seg in enumerate(face_intervals):
            print(f"Segment: {i}, Start: {seg['start']}, End: {seg['end']}")
            start = int(seg['start'])
            end = math.ceil(seg['end'])
            chunk = video.subclipped(start, end)
            chunk_filename = f"{video_id}__chunk__{i}.mp4"
            chunk.write_videofile(chunk_filename, codec="libx264")
            chunks.append(chunk_filename)

            # Check if the chunk is longer than 3 seconds
            if chunk.duration > 3:
                print(f"Chunk {chunk_filename} is longer than 3 seconds. Splitting into sub-chunks.")
                sub_chunks = []
                for sub_start in range(0, int(chunk.duration), 3):
                    sub_end = min(sub_start + 3, int(chunk.duration))
                    sub_chunk = chunk.subclipped(sub_start, sub_end)
                    sub_chunk_filename = chunk_filename.replace(f'__{i}', f'__{i}-{sub_start // 3}')
                    sub_chunk.write_videofile(sub_chunk_filename, codec="libx264")
                    sub_chunks.append(sub_chunk_filename)

                # Add sub-chunks to the list and remove the original chunk
                chunks.remove(chunk_filename)
                chunks.extend(sub_chunks)
                os.remove(chunk_filename)  # Remove the original chunk file

        video.close()
        os.remove(video_id + ".mp4")  # Remove the original video file

    except Exception as e:
        print(f"An error occurred while chunking the video: {e}")

    return chunks


def get_video_chunk_names(path):
    # extract video ID from the filename
    video_id = path.split('__')[0]
    chunk_name = path.split('__')[2].split('.')[0]
    return video_id, chunk_name


def split_audio_video(video_file):
    try:
        # extract video ID from the filename
        video_id, chunk_name = get_video_chunk_names(video_file)
        print(f"Video ID: {video_id}, Chunk Name: {chunk_name}")

        # import video
        video_chunk = VideoFileClip(video_file)

        # Split audio and video
        audio_path = os.path.join(f"{video_id}__audio__{chunk_name}.mp3")
        video_path = os.path.join(f"{video_id}__video__{chunk_name}.mp4")
        print(f"Audio path: {audio_path}, Video path: {video_path}")


        # Write audio to file
        video_chunk.audio.write_audiofile(audio_path)

        # Write video to file
        video_only = video_chunk.without_audio()
        video_only.write_videofile(video_path, codec="libx264", audio_codec="aac")
        #upload_to_s3(video_path, "data/video")

        # Close the video clip
        video_chunk.close()
        video_only.close()

        # delete chunk video file
        os.remove(video_file)

        print("Audio and video split successfully!")
        return audio_path, video_path
    except Exception as e:
        print(f"An error occurred: {e}")


# Phoneme vocabulary (39 phonemes + blank, as per VALLR paper)
phoneme_vocab = ['<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 
                 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 
                 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
phoneme_to_index = {p: i for i, p in enumerate(phoneme_vocab)}


def audio_to_text_phoneme(audio_file):
    try:
        # Use Whisper with English speech detection
        result = pipe(
            f"{os.getcwd()}/{audio_file}",
            generate_kwargs={"language": "en", "task": "transcribe"}
        )
        text = result["text"].strip()
        video_id, chunk_name = get_video_chunk_names(audio_file)
        
        # Validate text
        MAX_TEXT_LENGTH = 100  # ~10-20 words for 3 seconds
        MIN_TEXT_LENGTH = 15   # Ensure meaningful transcription
        if not text or len(text) < MIN_TEXT_LENGTH:
            print(f"Insufficient text length {len(text)} for {audio_file}: '{text}'")
            os.remove(audio_file)
            return False, None
        if len(text) > MAX_TEXT_LENGTH:
            print(f"Excessive text length {len(text)} for {audio_file}: '{text[:50]}...'")
            os.remove(audio_file)
            return False, None
        
        print(f"Transcription for {audio_file}: '{text}' (length: {len(text)})")

        phoneme_sequence = []
        # Get phonemes for the word
        for word in text.split():
            phonemes = pronouncing.phones_for_word(word)
            if phonemes:
                # remove stress markers and split by spaces
                phonemes = phonemes[0].split()
                phonemes = [p.replace("0", "").replace("1", "").replace("2", "") for p in phonemes]
                phoneme_sequence.extend(phonemes)

        phoneme_indices = [phoneme_to_index.get(p, 0) for p in phoneme_sequence]
        
        data = {
            "video_id": video_id,
            "chunk_name": chunk_name,
            "text": text,
            "phonemes": phoneme_sequence,
            "phoneme_indices": phoneme_indices,

        }
        json_file = f"{video_id}__text__{chunk_name}.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Text saved to {json_file}")

        os.remove(audio_file)

        return True, json_file
    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")
        return False, None


def crop_face_video_and_save(video_path):
    """
    Detects faces in a video, crops them, resizes to 224x224, and saves the frames to a .npz file.

    Args:
        video_path (str): Path to the input video.
        output_npz_path (str): Path to save the .npz file containing the cropped frames.

    Returns:
        None
    """
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in the frame
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                # Get the bounding box of the first detected face
                bbox = results.detections[0].location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)

                # Crop and resize the face
                face = frame[y:y+height, x:x+width]
                face = cv2.resize(face, (224, 224))
                frames.append(face)

        cap.release()

        # Convert frames to a NumPy array and save to .npz
        if frames:
            frames_array = np.array(frames)
            output_npz_path = video_path.replace('.mp4', '.npz').replace('video', 'face')
            np.savez_compressed(output_npz_path, frames=frames_array)
            print(f"Frames saved to {output_npz_path}")
            train_test_split = weighted_train_test_split()
            upload_to_s3(output_npz_path, f"data/vallr/{train_test_split}/face")
            os.remove(output_npz_path)
            os.remove(video_path)
            return False, train_test_split
        else:
            return True, None
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return True, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-folder", type=str, help="s3 folder")
    parser.add_argument("--local-file", type=str, help="local filer")

    args = parser.parse_args()
    s3_folder = args.s3_folder
    local_file = args.local_file
    if not s3_folder or not local_file:
        print("Please provide a YouTube URL.")
        exit(1)


    try:
        # download video from s3
        print(f"Downloading {local_file} from S3 bucket {bucket}...")
        print(f"Folder: {s3_folder}")
        s3_client.download_file(bucket, s3_folder + '/' + local_file, local_file)
        face_intervals = detect_face_intervals(local_file)
        chunks_list = chunk_video(local_file, face_intervals)
        for chunk in chunks_list:
            audio_path, video_path = split_audio_video(chunk)
            crop_video, json_file = audio_to_text_phoneme(audio_path)
            if crop_video:
                remove_json, train_test_split = crop_face_video_and_save(video_path)
                if remove_json:
                    print(f"Removing {json_file} due to no faces detected.")
                    os.remove(json_file)
                else:
                    print(f"Uploading {json_file} to S3.")
                    upload_to_s3(json_file, f"data/vallr/{train_test_split}/text")
                    os.remove(json_file)
            else:
                print(f"Skipping {chunk} due to transcription failure.")
                os.remove(video_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        


    # remove any files that end in .mp4 
    for file in os.listdir('.'):
        if file.endswith('.mp4'):
            os.remove(file)
            print(f"Removed {file}")
        if file.endswith('.json'):
            os.remove(file)
            print(f"Removed {file}")
        if file.endswith('.npz'):
            os.remove(file)
            print(f"Removed {file}")
        if file.endswith('.mp3'):
            os.remove(file)
            print(f"Removed {file}")





