import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_RAW_FILE = "./annotations_videos/video/raw_result/video_Sentiments_raw.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(SENTIMENTS_RAW_FILE, "r") as sentiments_data_file:
    data = sentiments_data_file.read()
    sentiments_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
sentiments_data_stats = dict(sentiments_data) # Make a copy

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    sentiments_arr = sentiments_data_stats[video_id]
    num_unique_sentiments = len(set(sentiments_arr))

    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    effective_data_stats[video_id] = ratings_mean
    sentiments_data_stats[video_id] = num_unique_sentiments

# Calculate the correlation between number of sentiments and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [(effective_data_stats[x], x) for x in video_ids]
effective_ratings.sort()
num_sentiments
num_sentiments = [sentiments_data_stats[x] for x in video_ids]
correlation = pearsonr(effective_ratings, num_sentiments)[0]

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between # of sentiments and effectiveness rating: %.3f" % (correlation))
print("Highest number of sentiments on a video: %d" % (max(num_sentiments)))
plt.scatter(effective_ratings, num_sentiments, s = 5)
plt.show()
