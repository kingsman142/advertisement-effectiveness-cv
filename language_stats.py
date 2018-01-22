import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
LANGUAGE_RAW_FILE = "./annotations_videos/video/raw_result/video_Language_raw.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(LANGUAGE_RAW_FILE, "r") as language_data_file:
    data = language_data_file.read()
    language_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
language_data_stats = dict(language_data) # Make a copy

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    if video_id in language_data_stats:
        language_arr = language_data_stats[video_id]
        language_mode = scipy.stats.mstats.mode(language_arr).mode[0] # Arbitrarily choose the 0th mode value if there are several

        ratings_mean = np.mean(ratings)
        ratings_mean = round(ratings_mean, 3)

        effective_data_stats[video_id] = ratings_mean
        if language_mode in [0, 1]: # only include non-english and english videos, -1 = indistinguishable
            language_data_stats[video_id] = language_mode
        else:
            language_data_stats.pop(video_id)

# Calculate the correlation between number of language and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [effective_data_stats[x] for x in video_ids if x in language_data_stats]
num_language = [language_data_stats[x] for x in video_ids if x in language_data_stats]
correlation = pearsonr(effective_ratings, num_language)[0]

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between # of language and effectiveness rating: %.3f" % (correlation))
print("Highest number of language on a video: %d" % (max(num_language)))
plt.scatter(effective_ratings, num_language, s = 5)
plt.show()
