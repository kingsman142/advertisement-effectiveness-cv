import json
import numpy as np
import scipy.stats

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

sentiments_sums = np.zeros(30)
sentiments_counts = np.zeros(30)
topics_sums = np.zeros(38)
topics_counts = np.zeros(38)

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)
    effective_data_stats[video_id] = ratings_mean

    video_sentiment = sentiments_data_stats[video_id]-1
    sentiments_sums[video_sentiment] += ratings_mean
    sentiments_counts[video_sentiment] += 1

    video_topic = topics_data_stats[video_id]-1
    topics_sums[video_topic] += ratings_mean
    topics_counts[video_topic] += 1

sentiments_averages = [sentiments_sums[i]/sentiments_counts[i] for i in range(0, 30)]
topics_averages = [topics_sums[i]/topics_counts[i] for i in range(0, 38)]

print("===== SENTIMENTS =====")
for i in range(0, 30):
    print("%s: %.2f" % (sentiments_list[i], sentiments_averages[i]))

print("\n===== TOPICS =====")
for i in range(0, 38):
    print("%s: %.2f" % (topics_list[i], topics_averages[i]))
