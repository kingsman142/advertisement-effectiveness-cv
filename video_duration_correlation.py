import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt

import requests
import json

SET_UP_DATA = False # Set this to true if you don't have video_Duration_raw.json and need to send the API requests to get video duration data
if SET_UP_DATA:
    API_KEY = "YOUR_API_KEY_HERE" # Please change this if you plan on making requests to the Youtube API
else:
    VIDEO_DURATION_RAW_FILE = "video_Duration_raw.json"
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

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)
    effective_data_stats[video_id] = ratings_mean

    if SET_UP_DATA:
        youtube_data = requests.get("https://www.googleapis.com/youtube/v3/videos", params = {'key': API_KEY, 'id': video_id, 'part': 'contentDetails'})
        youtube_data_json = youtube_data.json()
        if 'items' in youtube_data_json:
            if len(youtube_data_json['items']) > 0:
                duration_string = youtube_data_json['items'][0]['contentDetails']['duration']
                duration = convert_duration_to_int(duration_string)
                youtube_data_duration[video_id] = duration

# Calculate the correlation between number of language and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [effective_data_stats[x] for x in video_ids if x in youtube_data_duration]
duration_times = [youtube_data_duration[x] for x in video_ids if x in youtube_data_duration]
correlation = pearsonr(effective_ratings, duration_times)[0]

# NOTE: This commented code saves the duration data to a JSON file.  This should only be run once and only when you do not already possess the JSON file.
if SET_UP_DATA:
    with open('video_Duration_raw.json', 'w') as duration_file:
        json.dump(youtube_data_duration, duration_file)

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between duration and effectiveness rating: %.3f" % (correlation))
print("Number of timed videos: %d" % (len(youtube_data_duration)))
plt.scatter(effective_ratings, duration_times, s = 5)
plt.show()
