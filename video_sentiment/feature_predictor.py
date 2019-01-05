import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import math
import collections

VIDEO_EFFECTIVE_RAW_FILE = "../annotations_videos/video/raw_result/video_Effective_raw.json"
VIDEO_EFFECTIVE_CLEAN_FILE = "../annotations_videos/video/cleaned_result/video_Effective_clean.json"
SENTIMENTS_CLEAN_FILE = "../annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "../annotations_videos/video/cleaned_result/video_Topics_clean.json"
NUM_SENTIMENTS = 30
NUM_TOPICS = 38

def normalize_data_binary(names, labels, stats):
    topics_0_1 = {1: [], 2: []}
    names_two = {1: [], 2: []}
    for x in names:
        rating = int(labels[x])
        if x in stats and not rating == 3:
            if rating < 3:
                topics_0_1[1].append(stats[x])
                names_two[1].append(x)
            else:
                topics_0_1[2].append(stats[x])
                names_two[2].append(x)
    min_class = min(len(topics_0_1[1]), len(topics_0_1[2]))
    idx_0 = np.random.choice(np.arange(len(topics_0_1[1])), min_class, replace=False)
    idx_1 = np.random.choice(np.arange(len(topics_0_1[2])), min_class, replace=False)
    topics_0_1_0 = []
    topics_0_1_1 = []
    names_two_num = [[], []]
    for item in idx_0:
        topics_0_1_0.append(topics_0_1[1][item])
        names_two_num[0].append(names_two[1][item])
    for item in idx_1:
        topics_0_1_0.append(topics_0_1[2][item])
        names_two_num[1].append(names_two[2][item])
    new_list = topics_0_1_0 + topics_0_1_1
    new_list_output = [1]*min_class + [2]*min_class
    new_names_list = names_two_num[0] + names_two_num[1]
    indices = [i for i in range(min_class*2)]
    np.random.shuffle(indices)
    output_items = [new_list[indices[i]] for i in range(min_class*2)]
    output_labels = [new_list_output[indices[i]] for i in range(min_class*2)]
    output_names = [new_names_list[item] for item in indices]
    return output_items, output_labels, output_names


with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(VIDEO_EFFECTIVE_CLEAN_FILE, "r") as video_effective_data_clean:
    data = video_effective_data_clean.read()
    effective_data_clean = json.loads(data)

with open(SENTIMENTS_CLEAN_FILE, "r") as sentiments_data_file:
    data = sentiments_data_file.read()
    sentiments_data = json.loads(data)

with open(TOPICS_CLEAN_FILE, "r") as topics_data_file:
    data = topics_data_file.read()
    topics_data = json.loads(data)

topics_data_stats = dict(topics_data)
sentiments_data_stats = dict(sentiments_data)
effective_data_stats = {}
audio_stats = {}

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
train_n = math.floor(len(video_ids) * .8)
test_n = len(video_ids) - train_n
train, test = train_test_split(video_ids, train_size = train_n, test_size = test_n)

test_classes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for id in test:
    test_classes[int(effective_data_stats[id])] += 1
print("test classes: %s" % test_classes)

# Split out training samples
train_topics, train_out, train_ids = normalize_data_binary(train, effective_data_clean, topics_data_stats)
train_out_counts = collections.Counter(train_out)
wrong_labels = 0
for i in range(len(train_ids)):
    id = train_ids[i]
    if not int(effective_data_clean[id]) == train_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

# Split out testing samples
test_topics, test_out, test_ids = normalize_data_binary(test, effective_data_clean, topics_data_stats)
test_out_counts = collections.Counter(test_out)
wrong_labels = 0
for i in range(len(test_ids)):
    id = test_ids[i]
    if not int(effective_data_clean[id]) == test_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

# Gather average audio per video
for video_id in video_ids:
    filename = "./audio/" + video_id + ".npy"
    with open(filename, 'rb') as fp:
        audio = np.load(fp)
        avg_audio = np.mean(audio)
        audio_stats[video_id] = avg_audio

# Gather objects in the video
yes = 0
for video_id in video_ids:
    filename = "./common_object_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        objects = json.loads(data)
        if yes == 0:
            print(objects)
            yes = 1
