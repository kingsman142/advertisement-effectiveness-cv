import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt
import os

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Topics_clean.json"
topics_list = ["restaurant", "chocolate", "chips", "seasoning", "petfood", "alcohol", "coffee", "soda", "cars", "electronics", "phone_tv_internet_providers", "financial", "education", "security", "software", "other_service", "beauty", "healthcare", "clothing", "baby", "game", "cleaning", "home_improvement", "home_appliance", "travel", "media", "sports", "shopping", "gambling", "environment", "animal_right", "human_right", "safety", "smoking_alcohol_abuse", "domestic_violence", "self_esteem", "political", "charities"]

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

topics_ratings = {}
for i in range(0, 38):
    topics_ratings[i] = []

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    video_topic = topics_data_stats[video_id]
    topics_ratings[video_topic-1].append(ratings_mean)

    effective_data_stats[video_id] = ratings_mean

# Calculate the correlation between number of sentiments and average ratings
video_ids = effective_data_stats.keys()
if not os.path.exists("topics/"):
    os.makedirs("topics/")

fig = plt.figure(dpi=200)
for topic in topics_ratings:
    topics_ratings[topic].sort()
    x = [x for x in range(1, len(topics_ratings[topic])+1)]
    ticks = []
    if len(x) > 10:
        steps = int(len(x)/10)
        ticks = x[::steps]
    else:
        ticks = x # Every 5 elements for the x-axis tick marks
    fig2 = fig.add_subplot(7, 6, topic+1)
    fig2.scatter(x, topics_ratings[topic], s = 5)
    fig2.set_yticklabels([])
    fig2.set_xticklabels([])
    '''plt.figure()
    plt.scatter(x, topics_ratings[topic], s = 5)
    plt.xlabel("All videos with \"" + str(topics_list[topic]) + "\" topic")
    plt.ylabel("Effectiveness")
    plt.title("Effectiveness vs. \"" + str(topics_list[topic]) + "\" topic")
    plt.savefig("topics/topic_" + str(topics_list[topic]))'''
#plt.show()
fig.savefig("topics_grid.png")
fig.show()

print("Number of video ids: %d" % (len(video_ids)))
