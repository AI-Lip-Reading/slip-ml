# Preprocessing 

This folder has all the code required for gathering and processing data for model training.

### How it Works

The `kickoff-data-gather.ipynb` gathers the urls from a given YouTube playlist.  It checks what videos have already been downloaded so it's not downloaded again.  
It then downloads the video loacally and uploads to s3.  From there it kicks off a ECS task which the code in `gather-data.py` is ran.
This script is put into a docker container via the `start.sh` script.
Inside `gather-data.py` the YouTube video is downloaded from s3, it's audio is extracted and transcript to create sentance level phoneme translations to train the LLM.  This data is saved to s3.
From there the orginal YouTube video is sent to a face detection model to find the intervals of when a face is present. These intervals are used to clip the video.  If the clip is longer than 1 second the clip is further refined to 1 second clips.  These 1 second clips are split into just video and just audio.  The video only is sent to a similar face detection model to create 224x224 clips around the face.  These are convert to tensors and saved to s3.  The audio only clip is transcribed and converted to phonemes.  The transcription and phonemes are saved in a json file and sent to s3. 
