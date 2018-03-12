import json
import numpy as np
import scipy.stats

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Topics_clean.json"

topics_list = ["restaurant", "chocolate", "chips", "seasoning", "petfood", "alcohol", "coffee", "soda", "cars", "electronics", "phone_tv_internet_providers", "financial", "education", "security", "software", "other_service", "beauty", "healthcare", "clothing", "baby", "game", "cleaning", "home_improvement", "home_appliance", "travel", "media", "sports", "shopping", "gambling", "environment", "animal_right", "human_right", "safety", "smoking_alcohol_abuse", "domestic_violence", "self_esteem", "political", "charities"]
sentiments_list = ["active", "afraid", "alarmed", "alert", "amazed", "amused", "angry", "calm", "cheerful", "confident", "conscious", "creative", "disturbed", "eager", "educated", "emotional", "empathetic", "fashionable", "feminine", "grateful", "inspired", "jealous", "loving", "manly", "persuaded", "pessimistic", "proud", "sad", "thrifty", "youthful"]

positive_sentiments = ["active", "amazed", "amused", "cheerful", "confident", "eager", "grateful", "inspired", "loving", "proud", "youthful"]
negative_sentiments = ["afraid", "angry", "disturbed", "jealous", "pessimistic", "sad"]

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

positive_sentiments_sum = 0
positive_sentiments_count = 0
negative_sentiment_sum = 0
negative_sentiment_count = 0

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    effective_data_stats[video_id] = ratings_mean
    video_sentiment = sentiments_list[sentiments_data_stats[video_id]-1]
    if video_sentiment in positive_sentiments_sentiments:
        positive_sentiments_sum += ratings_mean
        positive_sentiments_count += 1
    elif video_sentiment in negative_sentiment_sentiments:
        negative_sentiment_sum += ratings_mean
        negative_sentiment_count += 1

if positive_sentiments_count == 0:
    print("Average positive sentiment effectiveness rating: N/A")
else:
    print("Average positive sentiment effectiveness rating: %.2f" % (positive_sentiments_sum/positive_sentiments_count))

if negative_sentiment_count == 0:
    print("Average negative sentiment effectiveness rating: N/A")
else:
    print("Average negative sentiment effectiveness rating: %.2f" % (negative_sentiment_sum/negative_sentiment_count))
