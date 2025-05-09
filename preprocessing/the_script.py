#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yt_dlp
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import boto3
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import random


# # Setup

# In[ ]:


secret_name = "youtube"
region_name = "us-east-1"

# Create a Secrets Manager client
session = boto3.session.Session()
secretsmanager = session.client(service_name='secretsmanager', region_name=region_name)

get_secret_value_response = secretsmanager.get_secret_value(SecretId=secret_name)

secret = get_secret_value_response['SecretString']
api_key = json.loads(secret)["API_KEY"]


# In[ ]:


total_duration = 0


# In[ ]:


s3_client = boto3.client('s3')
bucket = 'slip-ml'


# In[ ]:


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# In[ ]:


torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using {torch_dtype} dtype")


# In[ ]:


model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)


# In[ ]:


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


# # Gather and Save Raw Data

# In[ ]:


def get_playlist_videos(playlist_url):
    """
    Extract all video URLs from a YouTube playlist using the YouTube Data API.

    Args:
        playlist_url (str): The YouTube playlist URL (e.g., https://www.youtube.com/playlist?list=PL86SiVwkw_odmp-RVzD8yef3wU7Z2uD5a)
        api_key (str): Your YouTube Data API key

    Returns:
        list: List of video URLs
    """
    # Extract playlist ID from URL
    playlist_id = playlist_url.split("list=")[-1].split("&")[0]

    # Initialize YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    video_urls = []
    next_page_token = None

    try:
        while True:
            # Request playlist items
            request = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,  # Max allowed per request
                pageToken=next_page_token
            )
            response = request.execute()

            # Extract video IDs and create URLs
            for item in response['items']:
                video_id = item['contentDetails']['videoId']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_urls.append(video_url)

            # Check for next page
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    except HttpError as e:
        print(f"An error occurred: {e}")
        return []

    return video_urls


# In[ ]:


def upload_to_s3(local_file, s3_folder):
    s3_file = f"{s3_folder}/{local_file}"

    # Upload the file
    s3_client.upload_file(local_file, bucket, s3_file)
    print(f"Upload Successful: {local_file} -> {s3_file}")


# In[ ]:


def download_youtube_video_yt_dlp(url):
    # extract video ID from the URL
    video_id = url.split("v=")[-1]
    if "&" in video_id:
        video_id = video_id.split("&")[0]

    ydl_opts = {
        "outtmpl": f"{video_id}.%(ext)s",  # Output path and filename
        "format": "best",  # Select the best single file (video + audio)
        "merge_output_format": None,  # Avoid merging, stick to single stream
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully!")
        upload_to_s3(video_id + '.mp4', 'data/raw')
        return video_id
    except Exception as e:
        print(f"An error occurred: {e}")


# In[ ]:


def get_video_chunk_names(path):
    # extract video ID from the filename
    video_id = path.split('__')[0]
    chunk_name = path.split('__')[2].split('.')[0]
    return video_id, chunk_name


# In[ ]:


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
        upload_to_s3(video_path, "data/video")

        # Close the video clip
        video_chunk.close()
        video_only.close()

        # delete chunk video file
        os.remove(video_file)

        print("Audio and video split successfully!")
        return audio_path, video_path
    except Exception as e:
        print(f"An error occurred: {e}")


# In[ ]:


def clip_video_into_chunks(video_id):
    global total_duration
    input_file = video_id + '.mp4'
    try:
        video = VideoFileClip(input_file)
        duration = int(video.duration) - 30  # Get the duration of the video in seconds
        total_duration += duration

        chunk_length = (duration // 10)+1  # Length of each chunk in seconds
        chunks = []

        for start in range(0, duration, chunk_length):
            end = min(start + chunk_length, duration)
            chunk = video.subclipped(start, end)
            chunk_filename = f"{video_id}__chunk__{start // chunk_length}.mp4"
            chunk.write_videofile(chunk_filename, codec="libx264")
            chunks.append(chunk_filename)

        video.close()

        # detele original video file
        os.remove(input_file)

        return chunks
    except Exception as e:
        print(f"An error occurred while splitting the video: {e}")
        return []


# In[ ]:


def audio_to_text(audio_file):
    result = pipe(f"{os.getcwd()}/{audio_file}")
    text = result["text"]
    # extract video ID from the filename
    video_id, chunk_name = get_video_chunk_names(audio_file)
    # save text to JSON file
    data = {

        "video_id": video_id,
        "chunk_name": chunk_name,
        "text": text
    }

    # Save the data to a JSON file
    json_file = f"{video_id}__text__{chunk_name}.json"
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Text saved to {json_file}")
    upload_to_s3(json_file, "data/text")

    # remove audio file
    os.remove(audio_file)
    # remove json file
    os.remove(json_file)

    return text, video_id, chunk_name


# In[ ]:


def weighted_train_test_split():
    """
    Randomly returns 'train' or 'test' with a 90/10 weighted probability.

    Returns:
        str: 'train' (90% probability) or 'test' (10% probability).
    """
    return random.choices(['train', 'test'], weights=[90, 10], k=1)[0]


# # Preprocessing Option 1

# In[ ]:


def save_numpy_file(numpy_path, frames, train_test_split):
    print('saving video frames to numpy file')
    np.savez_compressed(numpy_path, frames=frames)
    upload_to_s3(numpy_path, "data/preprocessing-1/{}/videos".format(train_test_split))
    os.remove(numpy_path)


# In[ ]:


def preprocess_video(video_path, img_w, img_h, frames_n):

    video_id, chunk_name = get_video_chunk_names(video_path)

    print("setting up video preprocessing")
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    print("reading video frames")
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

    print("converting frames to numpy array")
    # Convert to numpy array
    frames = np.array(frames)

    train_test_split = weighted_train_test_split()

    # Save preprocessed video as .npy file
    numpy_path = f"{video_id}__numpy__{chunk_name}.npy"
    with ThreadPoolExecutor() as executor:
        executor.submit(save_numpy_file, numpy_path, frames, train_test_split)


    # remove video file
    os.remove(video_path)

    return train_test_split


# In[ ]:


# Example character-to-index mapping
char_to_idx = {char: idx for idx, char in enumerate(" abcdefghijklmnopqrstuvwxyz'.?!")} 

def text_to_labels(text, video_id, chunk_name, train_test_split):

    labels = [char_to_idx[char] for char in text if char in char_to_idx]

    json_file = f"{video_id}__text__{chunk_name}.json"
    with open(json_file, "w") as f:
        json.dump({"text": text, "labels": labels}, f)

    print(f"Text saved to {json_file}")
    upload_to_s3(json_file, "data/preprocessing-1/{}/labels".format(train_test_split))

    # remove json file
    os.remove(json_file)


# # Do the Work

# In[ ]:


# get list of YT video urls
playlist_urls = ["https://www.youtube.com/playlist?list=PL86SiVwkw_odmp-RVzD8yef3wU7Z2uD5a",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_ocn2nwGFoFWkBN8pFwgUshe",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_oe-sPwrWqm0k7t8bOK8dFpB",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_ofCWfjyBWs4PES8w-5AwPbx",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_oeR6BsaVjOwHunDOyAmDYxc",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_ocJPhcA3xiqszDcyIiGIt5y",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_oeLwHETCekBrdfP7M93LOpU",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_odq_rn2jUfdDAYgQfvijNtp",
                 "https://www.youtube.com/playlist?list=PL86SiVwkw_ofqbtdqZzFgzd--kxXhdRB4"]

all_video_urls = []
for playlist_url in playlist_urls:
    video_urls = get_playlist_videos(playlist_url)
    all_video_urls.extend(video_urls)

# remove duplicate video urls
all_video_urls = list(set(all_video_urls))
print(f"Total videos after removing duplicates: {len(all_video_urls)}")


# In[ ]:


last_video_num = 0
# for i, video_url in enumerate(all_video_urls):
#     if '74M0hPAeFHs' in video_url:
#         print(f"Skipping video {i+1}/{len(all_video_urls)}: {video_url}")
#         last_video_num = i
#         continue


# In[ ]:


frames_n = 300
img_w = 100 #1920//5
img_h = 50 #1080//5


# In[ ]:


for i, url in enumerate(all_video_urls[last_video_num:3], start=last_video_num+1):
    print(f"*******************************************************Processing video {i} of {len(all_video_urls)}********************************************************")
    #try:
    video_name = download_youtube_video_yt_dlp(url)
    if not video_name:
        print(f"Failed to download video {i} of {len(all_video_urls)}")
        continue
    video_chunks = clip_video_into_chunks(video_name)
    for chunk in video_chunks:
        #### General Preprocessing Steps ####
        audio_path, video_path = split_audio_video(chunk)
        text, video_id, chunk_name = audio_to_text(audio_path)
        ##### Preprocessing Steps #####
        train_test_split = preprocess_video(video_path, img_w, img_h, frames_n)
        text_to_labels(text, video_id, chunk_name, train_test_split)
        #### Remove files after processing ####
        try:
            os.remove(f"{os.getcwd()}/{video_name + 'mp4'}")
            os.remove(f"{os.getcwd()}/{video_path}")
            os.remove(f"{os.getcwd()}/{audio_path}")
        except:
            continue
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     continue


# In[ ]:


print(f"Total duration: {total_duration/60/60} hours")

