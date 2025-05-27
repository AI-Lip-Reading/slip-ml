
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
import re
#import resource

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
    device=device
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



def detect_face_intervals(video_file, sample_every=5):
    """
    Detects intervals in a video where a face is present, with tolerance for brief occlusions.
    
    Args:
        video_path (str): Path to the input video.
        sample_every (int): Process every Nth frame to save resources.
        
    Returns:
        list: A list of dictionaries with 'start' and 'end' times for each interval.
    """
    face_intervals = []
    try:
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # Use model optimized for longer range detection
            min_detection_confidence=0.3  # Lower threshold to catch more faces
        )

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        frame_duration = 1 / fps  # Duration of each frame in seconds
        
        print(f"Video has {total_frames} frames at {fps} FPS")

        face_present = False
        start_time = None
        no_face_frames = 0  # Counter for frames without a face
        face_frames = 0  # Counter for frames with a face
        tolerance = int(fps)  # Allow up to 1 second without faces before ending interval

        frame_index = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every Nth frame to save resources
            if frame_index % sample_every != 0:
                frame_index += 1
                continue
                
            processed_frames += 1
            if processed_frames % 1000 == 0:
                print(f"Processed {processed_frames} frames ({frame_index}/{total_frames}, {100*frame_index/total_frames:.1f}%)")

            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # Check if a face is detected
            has_face = results.detections and len(results.detections) > 0
            
            if has_face:
                face_frames += 1
                if not face_present:
                    # Start a new interval
                    start_time = frame_index * frame_duration
                    face_present = True
                    print(f"Face detected at {start_time:.2f}s (frame {frame_index})")
                no_face_frames = 0  # Reset no-face counter
            else:
                if face_present:
                    no_face_frames += 1
                    # End interval only after significant time without faces
                    if no_face_frames >= tolerance:
                        end_time = (frame_index - no_face_frames * sample_every) * frame_duration
                        interval_duration = end_time - start_time
                        # Only record intervals of meaningful duration
                        if interval_duration >= 1.0:
                            face_intervals.append({"start": start_time, "end": end_time})
                            print(f"Face interval ended at {end_time:.2f}s, duration: {interval_duration:.2f}s")
                        face_present = False

            frame_index += 1

        # Handle the case where the video ends while a face is still present
        if face_present:
            end_time = frame_index * frame_duration
            interval_duration = end_time - start_time
            if interval_duration >= 1.0:
                face_intervals.append({"start": start_time, "end": end_time})
                print(f"Final face interval: {start_time:.2f}s to {end_time:.2f}s, duration: {interval_duration:.2f}s")

        # Release resources
        cap.release()
        face_detection.close()
        
        print(f"Found {len(face_intervals)} face intervals with {face_frames} frames containing faces")
        for i, interval in enumerate(face_intervals):
            print(f"Interval {i+1}: {interval['start']:.2f}s to {interval['end']:.2f}s, duration: {interval['end'] - interval['start']:.2f}s")

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
            start = int(seg['start'])
            end = math.ceil(seg['end'])
            if end - start < 1:
                print(f"Chunk {i} is less than 1 second. Skipping.")
                continue
            print(f"Segment: {i}, Start: {start}, End: {end}")
            chunk = video.subclipped(start, end)
            chunk_filename = f"{video_id}__chunk__{i}.mp4"
            chunk.write_videofile(chunk_filename,
                                codec="libx264", 
                                audio_codec="aac",
                                logger=None,
                                ffmpeg_params=["-preset", "ultrafast"])
            chunks.append(chunk_filename)
    except Exception as e:
        print(f"An error occurred while chunking {video_id}: {e}")

    video.close()
    os.remove(video_id + ".mp4")  # Remove the original video file

    
    print(f"Created Face Chunks {len(chunks)} chunks for {video_id}")
    sub_chunks = []
    for i, chunk_filename in enumerate(chunks):
        try:
            # Load the chunk
            chunk = VideoFileClip(chunk_filename)
        except Exception as e:
            print(f"Error loading chunk {chunk_filename}: {e}")
            chunk.close()
            continue

        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk_filename}")

        # Check if the chunk is longer than 1 seconds
        if chunk.duration >= 1:
            print(f"Chunk {chunk_filename} is longer than 1 seconds. Splitting into sub-chunks.")
            for sub_start in range(0, int(chunk.duration)):
                try:
                    sub_end = min(sub_start + 1, int(chunk.duration))
                    print(f"Creating sub-chunk from {sub_start} to {sub_end} seconds")
                    sub_chunk = chunk.subclipped(sub_start, sub_end)
                    sub_chunk_filename = chunk_filename.replace(f'__{i}', f'__{i}-normal-{sub_start}')
                    sub_chunk.write_videofile(sub_chunk_filename, 
                                            codec="libx264", 
                                            audio_codec="aac",
                                            logger=None,
                                            ffmpeg_params=["-preset", "ultrafast"]
                                            ) 
                    sub_chunks.append(sub_chunk_filename)
                except Exception as e:
                    print(f"Error processing sub-chunk {sub_start} for chunk {chunk_filename}: {e}")
                    continue

            for sub_start in range(0, int(chunk.duration)):
                try:
                    half_sub_start = sub_start + 0.5
                    sub_end = min(half_sub_start + 1, int(chunk.duration))
                    print(f"Creating sub-chunk from {half_sub_start} to {sub_end} seconds")
                    sub_chunk = chunk.subclipped(half_sub_start, sub_end)
                    sub_chunk_filename = chunk_filename.replace(f'__{i}', f'__{i}-halfsec-{str(half_sub_start).replace(".", "-")}')
                    sub_chunk.write_videofile(sub_chunk_filename, codec="libx264")
                    sub_chunks.append(sub_chunk_filename)
                except Exception as e:
                    print(f"Error processing half-second sub-chunk {half_sub_start} for chunk {chunk_filename}: {e}")
                    continue
                    
        chunk.close()

    return sub_chunks


def get_video_chunk_names(path):
    try:
        # extract video ID from the filename
        video_id = path.split('__')[0]
        chunk_name = path.split('__')[2].split('.')[0]
        return video_id, chunk_name
    except IndexError:
        return path.replace('.mp4', ''), 'transcript'


def split_audio_video(video_file, both=True):
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

        if both:
            # Write video to file
            video_only = video_chunk.without_audio()
            video_only.write_videofile(video_path, codec="libx264", audio_codec="aac")
            video_only.close()

        # Close the video clip
        video_chunk.close()

        print("Audio and video split successfully!")
        return audio_path, video_path
    except Exception as e:
        print(f"An error occurred: {e}")



def get_phonemes_from(text):
    """
    Convert a text string into a list of phonemes.

    Args:
        text (str): The input text string.

    Returns:
        list: A list of phonemes corresponding to the input text.
    """
        # Phoneme vocabulary (39 phonemes + blank, as per VALLR paper)
    phoneme_vocab = ['<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 
                    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 
                    'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    phoneme_to_index = {p: i for i, p in enumerate(phoneme_vocab)}


    phoneme_sequence = []
    # Get phonemes for the word
    for word in text.split():
        phonemes = pronouncing.phones_for_word(word)
        if phonemes:
            # remove stress markers and split by spaces
            phonemes = phonemes[0].split()
            phonemes = [p.replace("0", "").replace("1", "").replace("2", "") for p in phonemes]
            phoneme_sequence.extend(phonemes)

    # Convert phonemes to indices
    phoneme_indices = [phoneme_to_index.get(p, 0) for p in phoneme_sequence]

    return phoneme_sequence, phoneme_indices


def process_video_to_sentence_phonemes(video_file):
    """
    Process a video file by:
    1. Splitting audio and video
    2. Splitting the audio into 25-second chunks
    3. Transcribing each audio chunk with Whisper
    4. Consolidating the transcriptions
    5. Splitting the transcription into sentences
    6. Converting each sentence to phonemes
    7. Saving each sentence/phoneme pair in a JSON file and uploading to S3
    
    Args:
        video_file (str): Path to the video file
        
    Returns:
        list: List of paths to the created JSON files
    """
    try:
        
        # Step 2: Split the audio into 25-second chunks
        print(f"Processing file: {video_file}")
        video = VideoFileClip(video_file)
        duration = video.duration
        print(f"Audio duration: {duration} seconds")
        
        # Define chunk length (25 seconds)
        chunk_length = 25
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration / chunk_length)
        print(f"Splitting audio into {num_chunks} chunks of {chunk_length} seconds each")
        
        # List to store all sentences from all chunks
        all_sentences = []
        video_id = video_file.split('.')[0]
        
        # Step 3: Process each audio chunk
        for i in range(num_chunks):
            start_time = i * chunk_length
            end_time = min((i + 1) * chunk_length, duration)
            
            print(f"Processing chunk {i+1}/{num_chunks}: {start_time}s to {end_time}s")
            
            try:
                # Extract the audio chunk
                chunk = video.subclipped(start_time, end_time)
                chunk_filename = f"{video_id}__chunk__{i}.mp3"
                print(f"Writing chunk to {chunk_filename}")
                chunk.audio.write_audiofile(chunk_filename, logger=None)
                chunk.close()
                
                if not os.path.exists(chunk_filename):
                    print(f"Chunk file {chunk_filename} does not exist. Skipping transcription.")
                    continue

                # Transcribe the chunk with Whisper
                print(f"Transcribing chunk {i+1}/{num_chunks}...")
                transcription_result = pipe(
                    f"/{chunk_filename}",
                    generate_kwargs={"language": "en", "task": "transcribe", "return_timestamps": True}
                )
                
                chunk_text = transcription_result["text"].strip()
                if not chunk_text:
                    print(f"No transcription found for chunk {i+1}")
                    os.remove(chunk_filename)
                    continue
                
                print(f"Processing transcription for chunk {i+1}: '{chunk_text}'")
                # Split chunk text into sentences
                chunk_sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                chunk_sentences = [s.strip() for s in chunk_sentences if s.strip()]
                
                # For cases where there are no proper sentence endings
                if not chunk_sentences and len(chunk_text) > 0:
                    # Split by commas or natural pauses if no sentences are found
                    chunk_sentences = re.split(r'(?<=[,:])\s+', chunk_text)
                    chunk_sentences = [s.strip() for s in chunk_sentences if s.strip()]
                    
                    # If still no splits, use the full text as a single sentence
                    if not chunk_sentences:
                        chunk_sentences = [chunk_text]
                
                # Add sentences from this chunk to the overall list
                all_sentences.extend(chunk_sentences)
                
                # Clean up temporary chunk file
                os.remove(chunk_filename)
                
            except Exception as chunk_error:
                print(f"Error processing chunk {i+1}: {chunk_error}")
                # Continue with the next chunk even if this one fails
                continue
        
        # Close the main audio clip
        video.close()
        
        # Check if we got any sentences
        if not all_sentences:
            print(f"No sentences found in any of the audio chunks for {video_file}")
            os.remove(video_file)
            return []
        
        print(f"Total sentences found: {len(all_sentences)}")
        json_files = []
        
        # Step 4 & 5: Convert sentences to phonemes, save and upload
        for i, sentence in enumerate(all_sentences):
            # Skip very short sentences
            if len(sentence) < 3:
                continue
                
            # Convert to phonemes
            phoneme_sequence, phoneme_indices = get_phonemes_from(sentence)
            
            # Skip if no phonemes were found
            if not phoneme_sequence:
                print(f"No phonemes found for sentence: {sentence}")
                continue
                
            # Create data structure
            data = {
                "video_id": video_id,
                "text": sentence,
                "phonemes": phoneme_sequence,
                "phoneme_indices": phoneme_indices,
            }
            
            # Save to JSON file
            json_filename = f"{video_id}__sentence__{i}.json"
            with open(json_filename, "w") as f:
                json.dump(data, f, indent=4)
                
            print(f"Saved sentence {i+1}/{len(all_sentences)} to {json_filename}")
            
            train_test_split = weighted_train_test_split()
            # Upload to S3
            upload_to_s3(json_filename, f"data/transcriptions/{train_test_split}")
            json_files.append(json_filename)
            
            # Clean up JSON file after upload
            os.remove(json_filename)
            
        # Clean up audio file
        os.remove(video_file)
        
        return json_files
        
    except Exception as e:
        print(f"Error processing video {video_file}: {e}")        
        return []



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
        #MAX_TEXT_LENGTH = 100  # ~10-20 words for 3 seconds
        MIN_TEXT_LENGTH = 3   # Ensure meaningful transcription
        if not text or len(text) < MIN_TEXT_LENGTH:
            print(f"Insufficient text length {len(text)} for {audio_file}: '{text}'")
            os.remove(audio_file)
            return False, None
        # if len(text) > MAX_TEXT_LENGTH:
        #     print(f"Excessive text length {len(text)} for {audio_file}: '{text[:50]}...'")
        #     os.remove(audio_file)
        #     return False, None
        
        print(f"Transcription for {audio_file}: '{text}' (length: {len(text)})")

        # Get phonemes and their indices
        phoneme_sequence, phoneme_indices = get_phonemes_from(text)
        
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
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # Use model optimized for longer range detection
            min_detection_confidence=0.3  # Lower threshold to catch more faces
        )
        
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
        face_detection.close()  # Explicitly close the face detection object

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
        print("Please provide a s3 folder and a local file.")
        exit(1)


    # download the video from S3
    print(f"Downloading {local_file} from {s3_folder}")
    s3_client.download_file(bucket, f"{s3_folder}/{local_file}", local_file)

    print(f"Processing {local_file} into sentence-level transcriptions...")
    json_files = process_video_to_sentence_phonemes(local_file)
    print(f"Created {len(json_files)} sentence-phoneme pairs.")

    print(f"Processing {local_file} with the standard pipeline...")
    face_intervals = detect_face_intervals(local_file)
    print(f"Face intervals: {face_intervals}")

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
        

    # Clean up temporary files
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

