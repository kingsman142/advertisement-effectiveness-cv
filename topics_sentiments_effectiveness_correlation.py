import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Topics_clean.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(SENTIMENTS_CLEAN_FILE, "r") as sentiments_data_file:
    data = sentiments_data_file.read()
    sentiments_data = json.loads(data)

with open(TOPICS_CLEAN_FILE, "r") as topics_data_file:
    data = topics_data_file.read()
    topics_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
sentiments_data_stats = dict(sentiments_data) # Make a copy
topics_data_stats = dict(topics_data) # Make a copy

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    effective_data_stats[video_id] = ratings_mean

# Calculate the correlation between number of sentiments and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [effective_data_stats[x] for x in video_ids if x in sentiments_data_stats and x in topics_data_stats]
sentiments = [sentiments_data_stats[x] for x in video_ids if x in sentiments_data_stats and x in topics_data_stats]
topics = [topics_data_stats[x] for x in video_ids if x in sentiments_data_stats and x in topics_data_stats]
sentiments_correlation = pearsonr(effective_ratings, sentiments)[0]
topics_correlation = pearsonr(effective_ratings, topics)[0]

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between sentiments and effectiveness rating: %.3f" % (sentiments_correlation))
print("Correlation between topics and effectiveness rating: %.3f" % (topics_correlation))
plt.scatter(effective_ratings, sentiments, s = 5)
plt.xlabel("Effectiveness rating (1 to 5)")
plt.ylabel("Sentiment Number (e.g. 5 = Amazed, 29 = Thrifty (frugal))")
plt.title("Sentiments vs. Effectiveness")
plt.figure()
plt.scatter(effective_ratings, topics, s = 5)
plt.xlabel("Effectiveness rating (1 to 5)")
plt.ylabel("Topic Number (e.g. 11 = Phone/TV/Internet, 36 = Self-esteem/Bullying)")
plt.title("Topics vs. Effectiveness")
plt.show()
