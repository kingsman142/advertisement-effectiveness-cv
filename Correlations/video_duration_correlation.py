import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt

import requests
import json
import glob
from moviepy.editor import VideoFileClip

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

# Closes a video file clip
def close_clip(clip):
    try:
        clip.reader.close()
        del clip.reader
        if clip.audio != None:
            clip.audio.reader.close_proc()
            del clip.audio
        del clip
    except Exception as e:
        sys.exc_clear()

SET_UP_DATA = False # Set this to true if you don't have video_Duration_raw.json and need to send the API requests to get video duration data
if SET_UP_DATA:
    API_KEY = "YOUR_API_KEY_HERE" # Please change this if you plan on making requests to the Youtube API
else:
    VIDEO_DURATION_RAW_FILE = "video_Duration_new_raw.json"
VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"

def convert_duration_to_int(duration_str): # Duration string comes in format PTxMyS where x is the number of minutes and y is seconds
    min_sec = duration_str.replace('PT', ',').replace('M', ',').replace('S', ',').split(',') # Easy way to obtain minutes and seconds from the string
    minutes, seconds = 0, int(min_sec[1])
    if len(min_sec[0]) > 0: # The video is at least one minute long; otherwise, this will be an empty string of length 0
        minutes = int(min_sec[0])
    total_time = minutes*60 + seconds # Calculate time in seconds
    return total_time

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

if not SET_UP_DATA:
    with open(VIDEO_DURATION_RAW_FILE, 'r') as video_duration_data:
        data = video_duration_data.read()
        duration_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
if SET_UP_DATA:
    youtube_data_duration = {} # Map youtube IDs to their duration in seconds (integers)
else:
    youtube_data_duration = dict(duration_data)

effective_data_keys = effective_data.keys()

batch1_filenames = glob.glob("./batch1/*.3gp")
batch1_filenames = [filename for filename in batch1_filenames if extract_video_id(filename, "1") in effective_data_keys]
batch1_ids = [extract_video_id(filename, "1") for filename in batch1_filenames]

batch2_filenames = glob.glob("./batch2/*.3gp")
batch2_filenames = [filename for filename in batch2_filenames if extract_video_id(filename, "2") in effective_data_keys]
batch2_ids = [extract_video_id(filename, "2") for filename in batch2_filenames]
batch2_ids = [id for id in batch2_ids if id not in batch1_ids]

i = 0
for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)
    effective_data_stats[video_id] = ratings_mean

    if SET_UP_DATA:
        clip = None
        if video_id in batch1_ids:
            clip = VideoFileClip("./batch1/" + video_id + ".3gp")
        elif video_id in batch2_ids:
            clip = VideoFileClip("./batch2/" + video_id + ".3gp")
        length = clip.duration
        close_clip(clip)
        youtube_data_duration[video_id] = int(round(length))
        i += 1
        if i % 100 == 0:
            print(i)

# Calculate the correlation between number of language and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [effective_data_stats[x] for x in video_ids if x in youtube_data_duration]
duration_times = [youtube_data_duration[x] for x in video_ids if x in youtube_data_duration]
correlation = pearsonr(effective_ratings, duration_times)[0]

# NOTE: This commented code saves the duration data to a JSON file.  This should only be run once and only when you do not already possess the JSON file.
if SET_UP_DATA:
    with open('video_Duration_new_raw.json', 'w') as duration_file:
        json.dump(youtube_data_duration, duration_file)

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between duration and effectiveness rating: %.3f" % (correlation))
print("Number of timed videos: %d" % (len(youtube_data_duration)))
plt.scatter(effective_ratings, duration_times, s = 5)
plt.show()
