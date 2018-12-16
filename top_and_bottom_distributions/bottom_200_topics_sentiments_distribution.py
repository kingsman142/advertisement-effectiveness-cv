import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt
import os
import itertools

VIDEO_EFFECTIVE_RAW_FILE = "../annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_CLEAN_FILE = "../annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "../annotations_videos/video/cleaned_result/video_Topics_clean.json"

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

topics_individual_counts = [0] * len(topics_list)
sents_individual_counts = [0] * len(sentiments_list)
for id, topic in topics_data.items():
    topics_individual_counts[topic-1] += 1
for id, sentiment in sentiments_data.items():
    sents_individual_counts[sentiment-1] += 1

with open('../useful-videos-effectiveness-ratings.txt', 'r') as useful_videos:
    bottom_200 = list(reversed([x for x in itertools.islice(useful_videos, None)][-(1):-(200+1):-1]))
for i in range(0, 200):
    bottom_200[i] = bottom_200[i].split("www.youtube.com/watch?v=")[1].split(" ")[0] # Grab the video ID for all 200 videos
    topic = topics_data_stats[bottom_200[i]]
    sentiment = sentiments_data_stats[bottom_200[i]]
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

topics_distribution = []
sentiments_distribution = []
i = 0
for topic in topics_labels:
    topic_index = topics_list.index(topic)
    topic_count = topics_individual_counts[topic_index]
    topics_distribution.append(float(topic_count)/sum(topics_individual_counts))
    print("%s: %.2f %.2f" % (topic, topics_distribution[i]*100.0, topics_pie[i]*100.0))
    i += 1
i = 0
print()
for sent in sentiments_labels:
    sent_index = sentiments_list.index(sent)
    sent_count = sents_individual_counts[sent_index]
    sentiments_distribution.append(float(sent_count)/sum(sents_individual_counts))
    print("%s: %.2f %.2f" % (sent, sentiments_distribution[i]*100.0, sentiments_pie[i]*100.0))
    i += 1
topics_pie_normalized = [topics_pie[i]/topics_distribution[i] for i in range(0, len(topics_distribution))]
sentiments_pie_normalized = [sentiments_pie[i]/sentiments_distribution[i] for i in range(0, len(sentiments_distribution))]

'''highest1topicIndex = 0
highest1topicSum = 0.0
highest1sentIndex = 0
highest1sentSum = 0.0
for i in range(0, len(topics_pie_normalized)):
    if topics_pie_normalized[i] <= 0.01 and highest1topicIndex != 0:
        highest1topicIndex = i
    if topics_pie_normalized[i] <= 0.01:
        highest1topicSum += highest1topicSum + topics_pie_normalized[i]
for i in range(0, len(sentiments_pie_normalized)):
    if sentiments_pie_normalized[i] <= 0.01 and highest1sentIndex != 0:
        highest1sentIndex = i
    if sentiments_pie_normalized[i] <= 0.01:
        highest1sentSum += highest1sentSum + sentiments_pie_normalized[i]
topics_pie_normalized = topics_pie_normalized[0:highest1topicIndex]
sentiments_pie_normalized = sentiments_pie_normalized[0:highest1sentIndex]
pie_topics_labels = list(topics_labels[0:highest1topicIndex])
pie_sentiments_labels = list(sentiments_labels[0:highest1sentIndex])
pie_topics_labels.append("other")
pie_sentiments_labels.append("other")
topics_pie_normalized.append(highest1topicSum)
sentiments_pie_normalized.append(highest1sentSum)'''

zero_to_thirty = [-i for i in range(0, 30)]
zero_to_thirtyeight = [-i for i in range(0, 38)]
plt.rcParams["figure.figsize"] = (17, 8)

plt.barh(zero_to_thirtyeight, topics_height, align='center', tick_label=topics_labels)
plt.ylabel("Topics")
plt.xlabel("Counts")
plt.title("Topics Distribution of 200 Least Effective Ads")
plt.savefig("bottom_200_topics_distribution_bar.png")

plt.figure()
plt.barh(zero_to_thirty, sentiments_height, align='center', tick_label=sentiments_labels)
plt.ylabel("Sentiments")
plt.xlabel("Counts")
plt.title("Sentiments Distribution of 200 Least Effective Ads")
plt.savefig("bottom_200_sentiments_distribution_bar.png")

sentiments_labels_copy = list(sentiments_labels)
sentiments_pie_normalized, sentiments_labels_copy = (list(t) for t in zip(*sorted(zip(sentiments_pie_normalized, sentiments_labels_copy))))
topics_labels_copy = list(topics_labels)
topics_pie_normalized, topics_labels_copy = (list(t) for t in zip(*sorted(zip(topics_pie_normalized, topics_labels_copy))))

plt.rcParams["figure.figsize"] = (8, 8)
plt.figure()
topics_colors = ["C"+str(topics_list.index(topic) % 10) for topic in topics_labels_copy if topic != "other"]
#topics_colors.append("C8")
plt.title("Topics Distribution of 200 Least Effective Ads")
plt.pie(topics_pie_normalized, labels=topics_labels_copy, autopct='%.2f%%', colors=topics_colors, textprops={'fontsize': 9})
plt.savefig("bottom_200_topics_distribution_pie_normalized.png")

plt.figure()
sentiments_colors = ["C"+str(sentiments_list.index(sentiment) % 10) for sentiment in sentiments_labels_copy if sentiment != "other"]
#sentiments_colors.append("C0")
plt.title("Sentiments Distribution of 200 Least Effective Ads")
plt.pie(sentiments_pie_normalized, labels=sentiments_labels_copy, autopct='%.2f%%', colors=sentiments_colors, textprops={'fontsize': 9})
plt.savefig("bottom_200_sentiments_distribution_pie_normalized.png")

plt.figure()
topics_colors = ["C"+str(topics_list.index(topic) % 10) for topic in topics_labels if topic != "other"]
#topics_colors.append("C8")
plt.title("Topics Distribution of 200 Least Effective Ads")
plt.pie(topics_pie, labels=topics_labels, autopct='%.2f%%', colors=topics_colors, textprops={'fontsize': 9})
plt.savefig("bottom_200_topics_distribution_pie.png")

plt.figure()
sentiments_colors = ["C"+str(sentiments_list.index(sentiment) % 10) for sentiment in sentiments_labels if sentiment != "other"]
#sentiments_colors.append("C0")
plt.title("Sentiments Distribution of 200 Least Effective Ads")
plt.pie(sentiments_pie, labels=sentiments_labels, autopct='%.2f%%', colors=sentiments_colors, textprops={'fontsize': 9})
plt.savefig("bottom_200_sentiments_distribution_pie.png")

topics_labels_copy = list(topics_labels)
topics_pie_normalized, topics_labels_copy = (list(t) for t in zip(*sorted(zip(topics_pie_normalized, topics_labels_copy))))
plt.figure()
plt.title("Normalized Topics Distribution, Bottom 200 Most Effective Ads")
plt.barh(zero_to_thirtyeight, topics_pie_normalized, align='center', tick_label=topics_labels_copy)
plt.tight_layout()
plt.savefig("bottom_200_topics_distribution_normalized.png")

sentiments_labels_copy = list(sentiments_labels)
sentiments_pie_normalized, sentiments_labels_copy = (list(t) for t in zip(*sorted(zip(sentiments_pie_normalized, sentiments_labels_copy))))
plt.figure()
plt.title("Normalized Sentiments Distribution, Bottom 200 Most Effective Ads")
plt.barh(zero_to_thirty, sentiments_pie_normalized, align='center', tick_label=sentiments_labels_copy)
plt.tight_layout()
plt.savefig("bottom_200_sentiments_distribution_normalized.png")

plt.show()
