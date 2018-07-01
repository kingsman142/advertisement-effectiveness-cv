import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import math
import collections
from sklearn.preprocessing import normalize
import pickle
from scipy.stats import mode
import warnings
import random
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
VIDEO_EFFECTIVE_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Effective_clean.json"
SENTIMENTS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Sentiments_clean.json"
TOPICS_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Topics_clean.json"
MEM_FILE = "./video_average_memorabilities.json"
OP_FLOW_FILE = "./optical_flow_bins.json"
CROPPED_30_FILE = "./video_average_intensities_cropped_30percent.json"
CROPPED_60_FILE = "./video_average_intensities_cropped_60percent.json"
AVG_HUE_FILE = "./video_average_hue.json"
MED_HUE_FILE = "./video_median_hue.json"
NUM_SENTIMENTS = 30
NUM_TOPICS = 38
TOPICS = ["restaurant", "chocolate", "chips", "seasoning", "petfood", "alcohol", "coffee", "soda", "cars", "electronics", "phone_tv_internet_providers", "financial", "education", "security", "software", "other_service", "beauty", "healthcare", "clothing", "baby", "game", "cleaning", "home_improvement", "home_appliance", "travel", "media", "sports", "shopping", "gambling", "environment", "animal_right", "human_right", "safety", "smoking_alcohol_abuse", "domestic_violence", "self_esteem", "political", "charities"]
SENTIMENTS = ["active", "afraid", "alarmed", "alert", "amazed", "amused", "angry", "calm", "cheerful", "confident", "conscious", "creative", "disturbed", "eager", "educated", "emotional", "empathetic", "fashionable", "feminine", "grateful", "inspired", "jealous", "loving", "manly", "persuaded", "pessimistic", "proud", "sad", "thrifty", "youthful"]

def normalize_data_binary(names, labels, stats):
    topics_0_1 = {0: [], 1: []}
    names_two = {0: [], 1: []}
    for x in names:
        rating = int(labels[x])
        if x in stats and not rating == 3:
            if rating < 3:
                topics_0_1[0].append(stats[x])
                names_two[0].append(x)
            else:
                topics_0_1[1].append(stats[x])
                names_two[1].append(x)
    min_class = min(len(topics_0_1[0]), len(topics_0_1[1]))
    idx_0 = np.random.choice(np.arange(len(topics_0_1[0])), min_class, replace=False)
    idx_1 = np.random.choice(np.arange(len(topics_0_1[1])), min_class, replace=False)
    topics_0_1_0 = []
    topics_0_1_1 = []
    names_two_num = [[], []]
    for item in idx_0:
        topics_0_1_0.append(topics_0_1[0][item])
        names_two_num[i].append(names_five[0][item])
    for item in idx_1:
        topics_0_1_0.append(topics_0_1[1][item])
        names_two_num[1].append(names_five[1][item])
    new_list = topics_0_1_0 + topics_0_1_1
    new_list_output = [0]*min_class + [1]*min_class
    new_names_list = names_two_num[0] + names_two_num[1]
    indices = [i for i in range(min_class*2)]
    np.random.shuffle(indices)
    output_items = [new_list[indices[i]] for i in range(min_class*2)]
    output_labels = [new_list_output[indices[i]] for i in range(min_class*2)]
    output_names = [new_names_list[item] for item in indices]
    return output_items, output_labels, output_names

def normalize_data_five(names, labels, stats):
    five_effectivenss_bins = {1: [], 2: [], 3: [], 4: [], 5: []} # store the data (e.g. optical flow values)
    names_five = {1: [], 2: [], 3: [], 4: [], 5: []} # store the IDs
    for x in names:
        rating = int(labels[x])
        if x in stats:
            five_effectivenss_bins[rating].append(stats[x])
            names_five[rating].append(x)
    min_class = min(len(five_effectivenss_bins[1]), len(five_effectivenss_bins[2]), len(five_effectivenss_bins[3]), len(five_effectivenss_bins[4]), len(five_effectivenss_bins[5]))
    idx_1 = np.random.choice(np.arange(len(five_effectivenss_bins[1])), min_class, replace=False)
    idx_2 = np.random.choice(np.arange(len(five_effectivenss_bins[2])), min_class, replace=False)
    idx_3 = np.random.choice(np.arange(len(five_effectivenss_bins[3])), min_class, replace=False)
    idx_4 = np.random.choice(np.arange(len(five_effectivenss_bins[4])), min_class, replace=False)
    idx_5 = np.random.choice(np.arange(len(five_effectivenss_bins[5])), min_class, replace=False)
    idx = [idx_1, idx_2, idx_3, idx_4, idx_5]
    five_effectivenss_bins_num = [[], [], [], [], []]
    names_five_num = [[], [], [], [], []]
    for i in range(5):
        for item in idx[i]:
            five_effectivenss_bins_num[i].append(five_effectivenss_bins[i+1][item])
            names_five_num[i].append(names_five[i+1][item])
    new_list = five_effectivenss_bins_num[0] + five_effectivenss_bins_num[1] + five_effectivenss_bins_num[2] + five_effectivenss_bins_num[3] + five_effectivenss_bins_num[4]
    new_names_list = names_five_num[0] + names_five_num[1] + names_five_num[2] + names_five_num[3] + names_five_num[4]
    new_list_output = [1]*min_class + [2]*min_class + [3]*min_class + [4]*min_class + [5]*min_class # store the ratings
    indices = [i for i in range(min_class*5)]
    np.random.shuffle(indices)
    output_items = [new_list[item] for item in indices] # values
    output_labels = [new_list_output[item] for item in indices] # effectiveness ratings
    output_names = [new_names_list[item] for item in indices] # IDs
    return output_items, output_labels, output_names

def normalize_data_four(names, labels, stats):
    topics_four = {1: [], 2: [], 4: [], 5: []}
    names_four = {1: [], 2: [], 3: [], 4: [], 5: []} # store the IDs
    for x in names:
        rating = int(labels[x])
        if x in stats and not rating == 3:
            topics_four[rating].append(stats[x])
            names_four[rating].append(x)
    min_class = min(len(topics_four[1]), len(topics_four[2]), len(topics_four[4]), len(topics_four[5]))
    idx_1 = np.random.choice(np.arange(len(topics_four[1])), min_class, replace=False)
    idx_2 = np.random.choice(np.arange(len(topics_four[2])), min_class, replace=False)
    idx_4 = np.random.choice(np.arange(len(topics_four[4])), min_class, replace=False)
    idx_5 = np.random.choice(np.arange(len(topics_four[5])), min_class, replace=False)
    idx = [idx_1, idx_2, idx_4, idx_5]
    topics_four_num = [[], [], [], []]
    names_four_num = [[], [], [], [], []]
    for i in range(4):
        for item in idx[i]:
            rating_val = i+1 if i < 2 else i+2
            topics_four_num[i].append(topics_four[rating_val][item])
    new_list = topics_four_num[0] + topics_four_num[1] + topics_four_num[2] + topics_four_num[3]
    new_list_output = [1]*min_class + [2]*min_class + [4]*min_class + [5]*min_class
    new_names_list = names_four_num[0] + names_four_num[1] + names_four_num[3] + names_four_num[4]
    indices = [i for i in range(min_class*4)]
    np.random.shuffle(indices)
    output_items = [new_list[item] for item in indices]
    output_labels = [new_list_output[item] for item in indices]
    output_names = [new_names_list[item] for item in indices] # IDs
    return output_items, output_labels, output_names

def normalize(ids, labels, counts):
    output_video_ids = []
    output_video_labels = []
    lowest_index = 1
    for item in [2, 3, 4, 5]:
        if counts[item] < counts[lowest_index]:
            lowest_index = item
    lowest_count = counts[lowest_index]
    for item in [1, 2, 3, 4, 5]:
        high = counts[item]
        indices_of_item = [i for i in range(len(labels)) if labels[i] == item]
        item_ids = [ids[i] for i in indices_of_item]
        item_labels = [labels[i] for i in indices_of_item]
        indices = random.sample(range(0, high), lowest_count)
        out_ids = [item_ids[i] for i in indices]
        out_labels = [item_labels[i] for i in indices]
        output_video_ids += out_ids
        output_video_labels += out_labels
    return output_video_ids

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)
    print(len(effective_data))

with open(SENTIMENTS_CLEAN_FILE, "r") as sentiments_data_file:
    data = sentiments_data_file.read()
    sentiments_data = json.loads(data)
    print(len(sentiments_data))

with open(TOPICS_CLEAN_FILE, "r") as topics_data_file:
    data = topics_data_file.read()
    topics_data = json.loads(data)
    print(len(topics_data))

with open(VIDEO_EFFECTIVE_CLEAN_FILE, "r") as video_effective_data_clean:
    data = video_effective_data_clean.read()
    effective_data_clean = json.loads(data)
    print(len(effective_data_clean))

with open(MEM_FILE, "r") as memorability_data:
    data = memorability_data.read()
    mem_data = json.loads(data)
    print(len(mem_data))

with open(OP_FLOW_FILE, "r") as optical_flow_data:
    data = optical_flow_data.read()
    opflow_data = json.loads(data)
    print(len(opflow_data))

with open(CROPPED_30_FILE, "r") as intensities_30percent_data:
    data = intensities_30percent_data.read()
    video_intensities_30percent = json.loads(data)
    print(len(video_intensities_30percent))

with open(CROPPED_60_FILE, "r") as intensities_60percent_data:
    data = intensities_60percent_data.read()
    video_intensities_60percent = json.loads(data)
    print(len(video_intensities_60percent))

with open(AVG_HUE_FILE, "r") as average_hue_data:
    data = average_hue_data.read()
    avg_hue_data = json.loads(data)
    print(len(avg_hue_data))

with open(MED_HUE_FILE, "r") as median_hue_data:
    data = median_hue_data.read()
    med_hue_data = json.loads(data)
    print(len(med_hue_data))

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

train_n = math.floor(len(video_ids) * .6)
test_n = len(video_ids) - train_n
train, test = train_test_split(video_ids, train_size = train_n, test_size = test_n)

test_classes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for id in test:
    test_classes[int(effective_data_clean[id])] += 1
print("test classes: %s" % test_classes)

train_topics, train_out, train_ids = normalize_data_five(train, effective_data_clean, topics_data_stats)
#train_topics = [topics_data_stats[x] for x in train if x in topics_data_stats and not int(effective_data_clean[x]) == 3]
#train_out = [int(effective_data_stats[x]) for x in train if x in topics_data_stats]
#train_out = [(0 if int(effective_data_clean[x]) < 3 else 1) for x in train if x in topics_data_stats and not int(effective_data_clean[x]) == 3]
train_out_counts = collections.Counter(train_out)
wrong_labels = 0
for i in range(len(train_ids)):
    id = train_ids[i]
    if not int(effective_data_clean[id]) == train_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

test_topics, test_out, test_ids = normalize_data_four(test, effective_data_clean, topics_data_stats)
#test_ids = [x for x in test if x in topics_data_stats]
#test_topics = [topics_data_stats[x] for x in test if x in topics_data_stats]
#test_out = [int(effective_data_stats[x]) for x in test if x in topics_data_stats]
test_out_counts = collections.Counter(test_out)
wrong_labels = 0
for i in range(len(test_ids)):
    id = test_ids[i]
    if not int(effective_data_clean[id]) == test_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

#train_sents, train_sents_out, train_sents_ids = normalize_data_five(train, effective_data_clean, sentiments_data_stats)
#train_sents = [sentiments_data_stats[x] for x in train_ids]
train_sents_out = [int(effective_data_clean[x]) for x in train_ids]
#train_sents_out = [(0 if int(effective_data_clean[x]) < 3 else 1) for x in train if x in sentiments_data_stats and not int(effective_data_clean[x]) == 3]
train_sents_out_counts = collections.Counter(train_sents_out)

#test_sents, test_sents_out, test_sents_ids = normalize_data_five(test, effective_data_clean, sentiments_data_stats)
test_sents = [sentiments_data_stats[x] for x in test_ids]
test_sents_out = [int(effective_data_clean[x]) for x in test_ids]
test_sents_out_counts = collections.Counter(test_sents_out)

topics_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
topics_SVM.fit(train_topics, train_out)
topics_score = topics_SVM.score(test_topics, test_out)
print("Topics SVM score: %.4f" % (topics_score))
#print(topics_SVM.n_support_[2])
#print(train_out_counts[2])
print(topics_SVM.n_support_)
print(train_out_counts)
print(train_sents_out_counts)
print(test_out_counts)
print(test_sents_out_counts)
topics_pred = topics_SVM.predict(test_topics)
#class_support_vectors_topics = normalize([[topics_SVM.n_support_[i] / train_out_counts[i+1] for i in range(0, len(topics_SVM.n_support_))]], norm='l1')
#print("Topics SVM Support Vectors: %s" % (class_support_vectors_topics))
#print("Weights: %s" % (topics_SVM.coef_))

sents_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
sents_SVM.fit(train_sents, train_sents)
sents_score = sents_SVM.score(test_sents, test_out)
print("Sentiments SVM score: %.4f" % (sents_score))
sents_pred = sents_SVM.predict(test_sents)
#class_support_vectors_sentiments = normalize([[sents_SVM.n_support_[i] / train_sents_out_counts[i+1] for i in range(0, len(sents_SVM.n_support_))]], norm='l1')
#print("Sentiments SVM Support Vectors: %s" % (class_support_vectors_sentiments))
#print("Weights: %s" % (sents_SVM.coef_))

topics_DT = DecisionTreeClassifier()
topics_DT.fit(train_topics, train_out)
topics_DT_pred = topics_DT.predict(test_topics)
print("Topics DT score: %.4f" % (topics_DT.score(test_topics, test_out)))

sents_DT = DecisionTreeClassifier()
sents_DT.fit(train_sents, train_sents)
sents_DT_pred = sents_DT.predict(test_sents)
print("Sentiments DT score: %.4f" % (sents_DT.score(test_sents, test_out)))

exciting_logress = pickle.load(open("exciting_logregress.pkl", "rb"))

mem_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
#train_mem, train_mem_out, train_mem_ids = normalize_data_five(train, effective_data_clean, mem_data)
train_mem = [[mem_data[x]] for x in train_ids]
mem_SVM.fit(train_mem, train_out)
#test_mems, test_mems_out, test_mem_ids = normalize_data_five(test, effective_data_clean, mem_data)
test_mems = [[mem_data[x]] for x in test_ids]
mem_SVM_pred = mem_SVM.predict(test_mems)
print("Mem SVM score: %.4f" % (mem_SVM.score(test_mems, test_out)))

opflow_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
#train_opflow, train_opflow_out, train_opflow_ids = normalize_data_five(train, effective_data_clean, opflow_data)
train_opflow = [opflow_data[x] for x in train_ids]
opflow_SVM.fit(train_opflow, train_out)
#test_opflow, test_opflow_out, test_opflow_ids = normalize_data_five(test, effective_data_clean, opflow_data)
test_opflow = [opflow_data[x] for x in test_ids]
opflow_SVM_pred = opflow_SVM.predict(test_opflow)
print("Opflow SVM score: %.4f" % (opflow_SVM.score(test_opflow, test_out)))

cropped_30_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
#train_cropped_30, train_cropped_30_out, train_cropped_30_ids = normalize_data_five(train, effective_data_clean, video_intensities_30percent)
train_cropped_30 = [[video_intensities_30percent[x]] for x in train_ids]
cropped_30_SVM.fit(train_cropped_30, train_out)
#test_cropped_30, test_cropped_30_out, test_cropped_30_ids = normalize_data_five(test, effective_data_clean, video_intensities_30percent)
test_cropped_30 = [[video_intensities_30percent[x]] for x in test_ids]
cropped_30_SVM_pred = cropped_30_SVM.predict(test_cropped_30)
print("cropped_30 SVM score: %.4f" % (cropped_30_SVM.score(test_cropped_30, test_out)))

cropped_60_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
#train_cropped_60, train_cropped_60_out, train_cropped_60_ids = normalize_data_five(train, effective_data_clean, video_intensities_60percent)
train_cropped_60 = [[video_intensities_60percent[x]] for x in train_ids]
cropped_60_SVM.fit(train_cropped_60, train_out)
#test_cropped_60, test_cropped_60_out, test_cropped_60_ids = normalize_data_five(test, effective_data_clean, video_intensities_60percent)
test_cropped_60 = [[video_intensities_60percent[x]] for x in test_ids]
cropped_60_SVM_pred = cropped_60_SVM.predict(test_cropped_60)
print("cropped_60 SVM score: %.4f" % (cropped_60_SVM.score(test_cropped_60, test_out)))

avg_hue_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
#train_avg_hue, train_avg_hue_out, train_avg_hue_ids = normalize_data_five(train, effective_data_clean, avg_hue_data)
train_avg_hue = [avg_hue_data[x] for x in train_ids]
avg_hue_SVM.fit(train_avg_hue, train_out)
#test_avg_hue, test_avg_hue_out, test_avg_hue_ids = normalize_data_five(test, effective_data_clean, avg_hue_data)
test_avg_hue = [avg_hue_data[x] for x in test_ids]
avg_hue_SVM_pred = avg_hue_SVM.predict(test_avg_hue)
print("Average Hue SVM score: %.4f" % (avg_hue_SVM.score(test_avg_hue, test_out)))

med_hue_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
#train_med_hue, train_med_hue_out, train_med_hue_ids = normalize_data_five(train, effective_data_clean, med_hue_data)
train_med_hue = [med_hue_data[x] for x in train_ids]
med_hue_SVM.fit(train_med_hue, train_out)
#test_med_hue, test_med_hue_out, test_med_hue_ids = normalize_data_five(test, effective_data_clean, med_hue_data)
test_med_hue = [med_hue_data[x] for x in test_ids]
med_hue_SVM_pred = med_hue_SVM.predict(test_med_hue)
print("Median Hue SVM score: %.4f" % (med_hue_SVM.score(test_med_hue, test_out)))

sents_svm_correct = np.zeros(30)
topics_svm_correct = np.zeros(38)
opflow_topics_correct = np.zeros(38)
opflow_sents_correct = np.zeros(30)
cropped_topics_correct = np.zeros(38)
cropped_sents_correct = np.zeros(30)
topics_totals = np.zeros(38)
sents_totals = np.zeros(30)
topics_correct = np.zeros(38)
sents_correct = np.zeros(30)

mem_topics_correct = np.zeros(38)
mem_sents_correct = np.zeros(30)
topics_dt_correct = np.zeros(38)
sents_dt_correct = np.zeros(30)
med_hue_topics_correct = np.zeros(38)
med_hue_sents_correct = np.zeros(30)
cropped_60_topics_correct = np.zeros(38)
cropped_60_sents_correct = np.zeros(30)

correct = 0
total = 0
predictions_in = []
predictions_out = []
predictions_train_size = math.floor(len(test_ids) * .5)
predictions_test_size = len(test_ids) - predictions_train_size
print("Predictions test size: %d" % predictions_test_size)
predictions_train, predictions_test = train_test_split(test_ids, train_size = predictions_train_size, test_size = predictions_test_size)
predictions_test_labels = [int(effective_data_clean[x]) for x in predictions_test]
predictions_test_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for id in predictions_test:
    predictions_test_counts[int(effective_data_clean[id])] += 1
print("predictions test counts: %s" % predictions_test_counts)
predictions_test = normalize(predictions_test, predictions_test_labels, predictions_test_counts)
predictions_test_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for id in predictions_test:
    predictions_test_counts[int(effective_data_clean[id])] += 1
    if False and id in test_ids:# and id in test_mem_ids and id in test_opflow_ids and id in test_avg_hue_ids and id in test_med_hue_ids and id in test_cropped_30_ids and id in test_cropped_60_ids:
        topics_index = test_ids.index(id)
        true_label = test_out[topics_index]
        predictions_test_counts[true_label] += 1
print("predictions test counts: %s" % predictions_test_counts)
for sample in test:
    if sample in test_ids:# and sample in test_mem_ids and sample in test_opflow_ids and sample in test_avg_hue_ids and sample in test_med_hue_ids and sample in test_cropped_30_ids and sample in test_cropped_60_ids:
        sample_index = test_ids.index(sample)
        topics_svm_class = topics_pred[sample_index]
        sents_svm_class = sents_pred[sample_index]
        topics_dt_class = topics_DT_pred[sample_index]
        sents_dt_class = sents_DT_pred[sample_index]
        mem_svm_class = mem_SVM_pred[sample_index]
        opflow_svm_class = opflow_SVM_pred[sample_index]
        avg_hue_svm_class = avg_hue_SVM_pred[sample_index]
        med_hue_svm_class = med_hue_SVM_pred[sample_index]
        cropped_30_class = cropped_30_SVM_pred[sample_index]
        cropped_60_class = cropped_60_SVM_pred[sample_index]
        true_label = test_out[sample_index]
        predicted_class = -1

        topic = list(test_topics[sample_index]).index(1)
        sent = list(test_sents[sample_index]).index(1)
        if sents_svm_class == true_label:
            sents_svm_correct[sent] += 1
        if topics_svm_class == true_label:
            topics_svm_correct[topic] += 1
        if opflow_svm_class == true_label:
            opflow_topics_correct[topic] += 1
            opflow_sents_correct[sent] += 1
        if cropped_30_class == true_label:
            cropped_topics_correct[topic] += 1
            cropped_sents_correct[sent] += 1
        if mem_svm_class == true_label:
            mem_topics_correct[topic] += 1
            mem_sents_correct[sent] += 1
        if med_hue_svm_class == true_label:
            med_hue_topics_correct[topic] += 1
            med_hue_sents_correct[sent] += 1
        if topics_dt_class == true_label:
            topics_dt_correct[topic] += 1
        if sents_dt_class == true_label:
            sents_dt_correct[sent] += 1
        if cropped_60_class == true_label:
            cropped_60_topics_correct[topic] += 1
            cropped_60_sents_correct[sent] += 1
        topics_totals[topic] += 1
        sents_totals[sent] += 1

        if sample in predictions_train:
            predictions = [0 if sents_svm_class == true_label else 1, 0 if topics_svm_class == true_label else 1, 0 if opflow_svm_class == true_label else 1, 0 if cropped_30_class == true_label else 1, 0 if mem_svm_class == true_label else 1, 0 if med_hue_svm_class == true_label else 1, 0 if topics_dt_class == true_label else 1, 0 if sents_dt_class == true_label else 1, 0 if cropped_60_class == true_label else 1, 0 if avg_hue_svm_class == true_label else 1]
            predictions_in.append(predictions)
            predictions_out.append(true_label)

        class_counts = collections.Counter([opflow_svm_class, cropped_30_class, cropped_60_class])
        predicted_class = class_counts.most_common(1)[0][0]
        total += 1

        if predicted_class == true_label:
            topics_correct[topic] += 1
            sents_correct[sent] += 1
            correct += 1

predictions_clf = SVC()
predictions_clf.fit(predictions_in, predictions_out)
print("Combiner accuracy: %.4f (%d correct, %d total)" % (correct/total, correct, total))
sents_svm_correct = np.array([sents_svm_correct[i] / sents_totals[i] for i in range(30)])
topics_svm_correct = np.array([topics_svm_correct[i] / topics_totals[i] for i in range(38)])
opflow_topics_correct = np.array([opflow_topics_correct[i] / topics_totals[i] for i in range(38)])
opflow_sents_correct = np.array([opflow_sents_correct[i] / sents_totals[i] for i in range(30)])
cropped_topics_correct = np.array([cropped_topics_correct[i] / topics_totals[i] for i in range(38)])
cropped_sents_correct = np.array([cropped_sents_correct[i] / sents_totals[i] for i in range(30)])
topics_correct = np.array([topics_correct[i] / topics_totals[i] for i in range(38)])
sents_correct = np.array([sents_correct[i] / sents_totals[i] for i in range(30)])

sents_dt_correct = np.array([sents_dt_correct[i] / sents_totals[i] for i in range(30)])
topics_dt_correct = np.array([topics_dt_correct[i] / topics_totals[i] for i in range(38)])
mem_topics_correct = np.array([mem_topics_correct[i] / topics_totals[i] for i in range(38)])
mem_sents_correct = np.array([mem_sents_correct[i] / sents_totals[i] for i in range(30)])
med_hue_topics_correct = np.array([med_hue_topics_correct[i] / topics_totals[i] for i in range(38)])
med_hue_sents_correct = np.array([med_hue_sents_correct[i] / sents_totals[i] for i in range(30)])
cropped_60_topics_correct = np.array([cropped_60_topics_correct[i] / topics_totals[i] for i in range(38)])
cropped_60_sents_correct = np.array([cropped_60_sents_correct[i] / sents_totals[i] for i in range(30)])
print(sents_svm_correct)
print(topics_svm_correct)
print(opflow_topics_correct)
print(opflow_sents_correct)
print(cropped_topics_correct)
print(cropped_sents_correct)
print("\n\n")
print(topics_totals)
print(sents_totals)
print("\n\n")
print(topics_correct)
print(sents_correct)
for i in range(len(topics_correct)):
    if topics_correct[i] < correct/total:
        print("Topic %s" % TOPICS[i-1])
for i in range(len(sents_correct)):
    if sents_correct[i] < correct/total:
        print("Sentiment %s" % SENTIMENTS[i-1])

correct = 0
total = 0
predictions_correct = 0
predictions_total = 0
classifications = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
misclassifications = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for sample in test:
    if sample in test_ids:
        sample_index = test_ids.index(sample)
        topics_svm_class = topics_pred[sample_index]
        sents_svm_class = sents_pred[sample_index]
        topics_dt_class = topics_DT_pred[sample_index]
        sents_dt_class = sents_DT_pred[sample_index]
        mem_svm_class = mem_SVM_pred[sample_index]
        opflow_svm_class = opflow_SVM_pred[sample_index]
        avg_hue_svm_class = avg_hue_SVM_pred[sample_index]
        med_hue_svm_class = med_hue_SVM_pred[sample_index]
        cropped_30_class = cropped_30_SVM_pred[sample_index]
        cropped_60_class = cropped_60_SVM_pred[sample_index]
        true_label = test_out[sample_index]
        predicted_class = -1

        topic = list(test_topics[sample_index]).index(1)
        sent = list(test_sents[sample_index]).index(1)
        total += 1

        if sample in predictions_test:
            predictions = [0 if sents_svm_class == true_label else 1, 0 if topics_svm_class == true_label else 1, 0 if opflow_svm_class == true_label else 1, 0 if cropped_30_class == true_label else 1, 0 if mem_svm_class == true_label else 1, 0 if med_hue_svm_class == true_label else 1, 0 if topics_dt_class == true_label else 1, 0 if sents_dt_class == true_label else 1, 0 if cropped_60_class == true_label else 1, 0 if avg_hue_svm_class == true_label else 1]
            if predictions_clf.predict([predictions]) == true_label:
                classifications[true_label] += 1
                if true_label == 1:
                    print(sample + " " + str(true_label))
                predictions_correct += 1
            else:
                misclassifications[true_label] += 1
            predictions_total += 1

        sents_scores = [sents_svm_correct[sent], opflow_sents_correct[sent], cropped_sents_correct[sent], sents_dt_correct[sent], mem_sents_correct[sent], med_hue_sents_correct[sent]]
        topics_scores = [topics_svm_correct[topic], opflow_topics_correct[topic], cropped_topics_correct[topic], topics_dt_correct[topic], mem_topics_correct[topic], med_hue_topics_correct[topic]]
        #classes = [sents_svm_class, topics_svm_class, opflow_svm_class, cropped_30_class]
        classes = [sents_svm_class, opflow_svm_class, cropped_30_class, sents_dt_class, mem_svm_class, med_hue_svm_class, topics_svm_class, opflow_svm_class, cropped_30_class, topics_dt_class, mem_svm_class, med_hue_svm_class]
        high_sents_index = 0
        high_topics_index = 0
        for i in range(len(sents_scores)):
            if sents_scores[i] > sents_scores[high_sents_index]:
                high_sents_index = i
        for i in range(len(topics_scores)):
            if topics_scores[i] > topics_scores[high_topics_index]:
                high_topics_index = i
        if sents_scores[high_sents_index] > topics_scores[high_topics_index]:
            predicted_class = classes[high_sents_index]
        else:
            predicted_class = classes[len(sents_scores) + high_topics_index]

        if predicted_class == true_label:
            topics_correct[topic] += 1
            sents_correct[sent] += 1
            correct += 1
print("Combiner accuracy (NEW): %.4f (%d correct, %d total)" % (correct/total, correct, total))
print("Predictions SVC accuracy: %.4f" % (predictions_correct/predictions_total))
print("Train len: %d" % len(predictions_train))
print("Test len: %d" % len(predictions_test))
print(classifications)
print(misclassifications)
#print(predictions_clf.coef_)

print("Number of video ids: %d" % (len(video_ids)))
print("Topics score: %.4f" % (topics_score))
print("Sents score: %.4f" % (sents_score))
