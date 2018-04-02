import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import math
import collections
from sklearn.preprocessing import normalize

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
SENTIMENTS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Topics_clean.json"
NUM_SENTIMENTS = 30
NUM_TOPICS = 38

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

    one_hot_topics = np.zeros((NUM_TOPICS))
    topic_val = topics_data_stats[video_id]
    one_hot_topics[topic_val-1] = 1

    one_hot_sentiments = np.zeros((NUM_SENTIMENTS))
    sentiment_val = sentiments_data_stats[video_id]
    one_hot_sentiments[sentiment_val-1] = 1

    topics_data_stats[video_id] = one_hot_topics
    sentiments_data_stats[video_id] = one_hot_sentiments
    effective_data_stats[video_id] = ratings_mean

video_ids = list(effective_data_stats.keys())

train_n = math.floor(len(video_ids) * .7)
test_n = len(video_ids) - train_n
train, test = train_test_split(video_ids, train_size = train_n, test_size = test_n)

train_topics = [topics_data_stats[x] for x in train if x in topics_data_stats]
train_topics_out = [int(effective_data_stats[x]) for x in train if x in topics_data_stats]
train_topics_out_counts = collections.Counter(train_topics_out)

test_topics = [topics_data_stats[x] for x in test if x in topics_data_stats]
test_topics_out = [int(effective_data_stats[x]) for x in test if x in topics_data_stats]
test_topics_out_counts = collections.Counter(test_topics_out)

train_sents = [sentiments_data_stats[x] for x in train if x in sentiments_data_stats]
train_sents_out = [int(effective_data_stats[x]) for x in train if x in sentiments_data_stats]
train_sents_out_counts = collections.Counter(train_sents_out)

test_sents = [sentiments_data_stats[x] for x in test if x in sentiments_data_stats]
test_sents_out = [int(effective_data_stats[x]) for x in test if x in sentiments_data_stats]
test_sents_out_counts = collections.Counter(test_sents_out)

topics_SVM = SVC(kernel='linear', degree=3)
#print(train_topics[0].shape)
#print(len(train_topics_out))
topics_SVM.fit(train_topics, train_topics_out)
topics_score = topics_SVM.score(test_topics, test_topics_out)
print(topics_SVM.n_support_[2])
print(train_topics_out_counts[2])
print(topics_SVM.n_support_)
print(train_topics_out_counts)
class_support_vectors_topics = normalize([[topics_SVM.n_support_[i] / train_topics_out_counts[i+1] for i in range(0, len(topics_SVM.n_support_))]], norm='l1')
print("Topics SVM Support Vectors: %s" % (class_support_vectors_topics))
print("Weights: %s" % (topics_SVM.coef_))

sents_SVM = SVC(kernel='linear')
sents_SVM.fit(train_sents, train_sents_out)
sents_score = sents_SVM.score(test_sents, test_sents_out)
class_support_vectors_sentiments = normalize([[sents_SVM.n_support_[i] / train_sents_out_counts[i+1] for i in range(0, len(sents_SVM.n_support_))]], norm='l1')
print("Sentiments SVM Support Vectors: %s" % (class_support_vectors_sentiments))
print("Weights: %s" % (sents_SVM.coef_))

print("Number of video ids: %d" % (len(video_ids)))
print("Topics score: %.4f" % (topics_score))
print("Sents score: %.4f" % (sents_score))
