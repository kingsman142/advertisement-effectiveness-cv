import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt
import os

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Topics_clean.json"
sentiments_list = ["active", "afraid", "alarmed", "alert", "amazed", "amused", "angry", "calm", "cheerful", "confident", "conscious", "creative", "disturbed", "eager", "educated", "emotional", "empathetic", "fashionable", "feminine", "grateful", "inspired", "jealous", "loving", "manly", "persuaded", "pessimistic", "proud", "sad", "thrifty", "youthful"]

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

sentiments_ratings = {}
for i in range(0, 30):
    sentiments_ratings[i] = []

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    video_sentiment = sentiments_data_stats[video_id]
    sentiments_ratings[video_sentiment-1].append(ratings_mean)

    effective_data_stats[video_id] = ratings_mean

# Calculate the correlation between number of sentiments and average ratings
video_ids = effective_data_stats.keys()
if not os.path.exists("sentiments/"):
    os.makedirs("sentiments/")

fig = plt.figure(dpi=200)
for sentiment in sentiments_ratings:
    sentiments_ratings[sentiment].sort()
    x = [x for x in range(1, len(sentiments_ratings[sentiment])+1)]
    ticks = []
    if len(x) > 10:
        steps = int(len(x)/10)
        ticks = x[::steps]
    else:
        ticks = x # Every 5 elements for the x-axis tick marks

    fig2 = fig.add_subplot(5, 6, sentiment+1)
    fig2.scatter(x, sentiments_ratings[sentiment], s = 5)
    fig2.set_yticklabels([])
    fig2.set_xticklabels([])
    '''plt.figure()
    plt.xticks(ticks)
    plt.scatter(x, sentiments_ratings[sentiment], s = 5)
    plt.xlabel("All videos with \"" + str(sentiments_list[sentiment]) + "\" sentiment")
    plt.ylabel("Effectiveness")
    plt.title("Effectiveness vs. \"" + str(sentiments_list[sentiment]) + "\" sentiment")
    plt.savefig("sentiments/sentiment_" + str(sentiments_list[sentiment]))'''
fig.savefig("sentiments_grid.png")
fig.show()

print("Number of video ids: %d" % (len(video_ids)))
