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
import os

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

with open("audio.json", "w+") as audio_file:
    audio_file.write(json.dumps(audio_stats))

# Audio SVM Score
train_audio = [[audio_stats[x]] for x in train_ids]
test_audio = [[audio_stats[x]] for x in test_ids]
audio_SVM = SVC(kernel='linear', decision_function_shape='ovr')
audio_SVM.fit(train_audio, train_out)
audio_score = audio_SVM.score(test_audio, test_out)
print("Audio SVM score: %.4f" % (audio_score))

# Audio Correlation with Effectiveness
effective_ratings = [float(effective_data_clean[x]) for x in video_ids if x in audio_stats]
audio_vals = [audio_stats[x] for x in video_ids if x in audio_stats]
correlation = pearsonr(effective_ratings, audio_vals)[0]
print("Audio and Effectiveness correlation: " + str(correlation))

objects = {}
num_objects = 0
# Gather most common objects per video
for video_id in video_ids:
    filename = "./common_object_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        for frame in frames:
            for object_dict in frame:
                object = object_dict["cname"]
                num_objects += 1
                if object in objects:
                    objects[object] += 1
                else:
                    objects[object] = 1
for object in objects:
    objects[object] = objects[object] / num_objects
object_names = objects.keys()
print("Number of objects found: " + str(num_objects))
print("Number of unique objects found: " + str(len(object_names)))

object_stats = {}
# Gather most common objects per video
for video_id in video_ids:
    filename = "./common_object_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        video_objects = {}
        video_num_objects = 0
        for frame in frames:
            for object_dict in frame:
                object = object_dict["cname"]
                video_num_objects += 1
                if object in video_objects:
                    video_objects[object] += 1
                else:
                    video_objects[object] = 1
        for object_name in object_names:
            if object_name not in video_objects:
                video_objects[object_name] = 0.0
        for object in video_objects:
            video_objects[object] = video_objects[object] / video_num_objects # calculate the probability of each object occuring in this specific video
            video_objects[object] = video_objects[object] / objects[object] # normalize the distribution of this video's objects probabilities
        object_stats[video_id] = [video_objects[object] for object in object_names]

with open("objects.json", "w+") as objects_file:
    objects_file.write(json.dumps(object_stats))

train_objects = [object_stats[x] for x in train_ids]
test_objects = [object_stats[x] for x in test_ids]
objects_SVM = SVC(kernel='linear', decision_function_shape='ovr')
objects_SVM.fit(train_objects, train_out)
objects_score = objects_SVM.score(test_objects, test_out)
print("Objects SVM score: %.4f" % (objects_score))

places = {}
num_places = 0
# Gather most common places per video
for video_id in video_ids:
    filename = "./place_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        for frame in frames:
            for place_dict in frame:
                place = place_dict["cname"]
                num_places += 1
                if place in places:
                    places[place] += 1
                else:
                    places[place] = 1
for place in places:
    places[place] = places[place] / num_places
place_names = places.keys()
print("Number of places found: " + str(num_places))
print("Number of unique places found: " + str(len(place_names)))

place_stats = {}
# Gather most common places per video
for video_id in video_ids:
    filename = "./place_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        video_places = {}
        video_num_places = 0
        for frame in frames:
            for place_dict in frame:
                place = place_dict["cname"]
                video_num_places += 1
                if place in video_places:
                    video_places[place] += 1
                else:
                    video_places[place] = 1
        for place_name in place_names:
            if place_name not in video_places:
                video_places[place_name] = 0.0
        for place in video_places:
            video_places[place] = video_places[place] / video_num_places # calculate the probability of each place occuring in this specific video
            video_places[place] = video_places[place] / places[place] # normalize the distribution of this video's places probabilities
        place_stats[video_id] = [video_places[place] for place in place_names]

with open("places.json", "w+") as places_file:
    places_file.write(json.dumps(place_stats))

train_places = [place_stats[x] for x in train_ids]
test_places = [place_stats[x] for x in test_ids]
places_SVM = SVC(kernel='linear', decision_function_shape='ovr')
places_SVM.fit(train_places, train_out)
places_score = places_SVM.score(test_places, test_out)
print("Places SVM score: %.4f" % (places_score))

expressions = {}
num_expressions = 0
# Gather most common expressions per video
for video_id in video_ids:
    filename = "./affectnet_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        for frame in frames:
            for expression_dict in frame:
                expression = expression_dict["expression"]
                num_expressions += 1
                if expression in expressions:
                    expressions[expression] += 1
                else:
                    expressions[expression] = 1
for expression in expressions:
    expressions[expression] = expressions[expression] / num_expressions
expression_names = expressions.keys()
print("Number of expressions found: " + str(num_expressions))
print("Number of unique expressions found: " + str(len(expression_names)))

expression_stats = {}
# Gather most common expressions per video
for video_id in video_ids:
    filename = "./affectnet_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        video_expressions = {}
        video_num_expressions = 0
        for frame in frames:
            for expression_dict in frame:
                expression = expression_dict["expression"]
                video_num_expressions += 1
                if expression in video_expressions:
                    video_expressions[expression] += 1
                else:
                    video_expressions[expression] = 1
        for expression_name in expression_names:
            if expression_name not in video_expressions:
                video_expressions[expression_name] = 0.0
        for expression in video_expressions:
            if video_num_expressions > 0:
                video_expressions[expression] = video_expressions[expression] / video_num_expressions # calculate the probability of each expression occuring in this specific video
            else:
                video_expressions[expression] = 0.0
            video_expressions[expression] = video_expressions[expression] / expressions[expression] # normalize the distribution of this video's expressions probabilities
        expression_stats[video_id] = [video_expressions[expression] for expression in expression_names]

with open("expressions.json", "w+") as expressions_file:
    expressions_file.write(json.dumps(expression_stats))

train_expressions = [expression_stats[x] for x in train_ids]
test_expressions = [expression_stats[x] for x in test_ids]
expressions_SVM = SVC(kernel='linear', decision_function_shape='ovr')
expressions_SVM.fit(train_expressions, train_out)
expressions_score = expressions_SVM.score(test_expressions, test_out)
print("Expressions SVM score: %.4f" % (expressions_score))

emotions = {}
num_emotions = 0
# Gather most common emotions per video
for video_id in video_ids:
    filename = "./emotic_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        for frame in frames:
            for emotion_dict in frame:
                for category_dict in emotion_dict["emotic"]["categories"]:
                    emotion = category_dict["category"]
                    num_emotions += 1
                    if emotion in emotions:
                        emotions[emotion] += 1
                    else:
                        emotions[emotion] = 1
for emotion in emotions:
    emotions[emotion] = emotions[emotion] / num_emotions
emotion_names = emotions.keys()
print("Number of emotions found: " + str(num_emotions))
print("Number of unique emotions found: " + str(len(emotion_names)))

emotion_stats = {}
# Gather most common emotions per video
for video_id in video_ids:
    filename = "./emotic_feature/" + video_id + ".json"
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        video_emotions = {}
        video_num_emotions = 0
        for frame in frames:
            for emotion_dict in frame:
                for category_dict in emotion_dict["emotic"]["categories"]:
                    emotion = category_dict["category"]
                    video_num_emotions += 1
                    if emotion in video_emotions:
                        video_emotions[emotion] += 1
                    else:
                        video_emotions[emotion] = 1
        for emotion_name in emotion_names:
            if emotion_name not in video_emotions:
                video_emotions[emotion_name] = 0.0
        for emotion in video_emotions:
            video_emotions[emotion] = video_emotions[emotion] / video_num_emotions # calculate the probability of each emotion occuring in this specific video
            video_emotions[emotion] = video_emotions[emotion] / emotions[emotion] # normalize the distribution of this video's emotions probabilities
        emotion_stats[video_id] = [video_emotions[emotion] for emotion in emotion_names]

with open("emotions.json", "w+") as emotions_file:
    emotions_file.write(json.dumps(emotion_stats))

train_emotions = [emotion_stats[x] for x in train_ids]
test_emotions = [emotion_stats[x] for x in test_ids]
emotions_SVM = SVC(kernel='linear', decision_function_shape='ovr')
emotions_SVM.fit(train_emotions, train_out)
emotions_score = emotions_SVM.score(test_emotions, test_out)
print("Emotions SVM score: %.4f" % (emotions_score))

climax_stats = {}
new_video_ids = []
for video_id in video_ids:
    filename = "./climax_feature_v2/" + video_id + ".json"
    if not os.path.isfile(filename):
        continue
    new_video_ids.append(video_id)
    with open(filename, 'r') as fp:
        data = fp.read()
        frames = json.loads(data)
        climaxes = 0
        for frame in frames:
            if frame["score"] == 1.0: # there is a climax here
                climaxes += 1
        climax_stats[video_id] = climaxes

with open("climax.json", "w+") as climax_file:
    climax_file.write(json.dumps(climax_stats))

# climax SVM Score
train_climax = [[climax_stats[x]] for x in train_ids if x in new_video_ids]
train_climax_out = [[effective_data_clean[x]] for x in train_ids if x in new_video_ids]
test_climax = [[climax_stats[x]] for x in test_ids if x in new_video_ids]
test_climax_out = [[effective_data_clean[x]] for x in test_ids if x in new_video_ids]
climax_SVM = SVC(kernel='linear', decision_function_shape='ovo')
climax_SVM.fit(train_climax, train_climax_out)
climax_score = climax_SVM.score(test_climax, test_climax_out)
print("Climax SVM score: %.4f" % (climax_score))

# climax Correlation with Effectiveness
effective_ratings = [float(effective_data_clean[x]) for x in video_ids if x in climax_stats]
climax_vals = [climax_stats[x] for x in video_ids if x in climax_stats]
correlation = pearsonr(effective_ratings, climax_vals)[0]
print("Climax and Effectiveness correlation: " + str(correlation))
