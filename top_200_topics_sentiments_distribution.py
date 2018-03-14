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

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    effective_data_stats[video_id] = ratings_mean

# You must first run statistics.py first to generate useful-videos-effectiveness-ratings.txt file
topics_count = np.zeros(len(topics_list))
sentiments_count = np.zeros(len(sentiments_list))
with open('useful-videos-effectiveness-ratings.txt', 'r') as useful_videos:
    top_200 = [next(useful_videos) for x in range(200)]
for i in range(0, 200):
    top_200[i] = top_200[i].split("www.youtube.com/watch?v=")[1].split(" ")[0] # Grab the video ID for all 200 videos
    topic = topics_data_stats[top_200[i]]
    sentiment = sentiments_data_stats[top_200[i]]
    topics_count[topic-1] += 1
    sentiments_count[sentiment-1] += 1
topics_count_dict = dict((topic, topics_count[topics_list.index(topic)]) for topic in topics_list)
sentiments_count_dict = dict((sentiment, sentiments_count[sentiments_list.index(sentiment)]) for sentiment in sentiments_list)
topics_labels = []
topics_height = []
sentiments_labels = []
sentiments_height = []
for (count, topic) in sorted([(count, topic) for (topic, count) in topics_count_dict.items()], reverse=True):
    topics_labels.append(topic)
    topics_height.append(count)
for (count, sentiment) in sorted([(count, sentiment) for (sentiment, count) in sentiments_count_dict.items()], reverse=True):
    sentiments_labels.append(sentiment)
    sentiments_height.append(count)
topics_pie = [height/sum(topics_height) for height in topics_height]
sentiments_pie = [height/sum(sentiments_height) for height in sentiments_height]

zero_to_thirty = [-i for i in range(0, 30)]
zero_to_thirtyeight = [-i for i in range(0, 38)]
plt.rcParams["figure.figsize"] = (17, 8)

plt.barh(zero_to_thirtyeight, topics_height, align='center', tick_label=topics_labels)
plt.ylabel("Topics")
plt.xlabel("Counts")
plt.title("Topics Distribution of 200 Most Effective Ads")
plt.savefig("top_200_topics_distribution_bar.png")

plt.figure()
plt.barh(zero_to_thirty, sentiments_height, align='center', tick_label=sentiments_labels)
plt.ylabel("Sentiments")
plt.xlabel("Counts")
plt.title("Sentiments Distribution of 200 Most Effective Ads")
plt.savefig("top_200_sentiments_distribution_bar.png")

plt.rcParams["figure.figsize"] = (8, 8)
plt.figure()
topics_colors = ["C"+str(topics_list.index(topic) % 10) for topic in topics_labels]
plt.title("Topics Distribution of 200 Most Effective Ads")
plt.pie(topics_pie, labels=topics_labels, autopct='%.2f%%', colors=topics_colors)
plt.savefig("top_200_topics_distribution_pie.png")

plt.figure()
sentiments_colors = ["C"+str(sentiments_list.index(sentiment) % 10) for sentiment in sentiments_labels]
plt.title("Sentiments Distribution of 200 Most Effective Ads")
plt.pie(sentiments_pie, labels=sentiments_labels, autopct='%.2f%%', colors=sentiments_colors)
plt.savefig("top_200_sentiments_distribution_pie.png")

#plt.show()
