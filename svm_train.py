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
from sklearn.preprocessing import normalize
import pickle
from scipy.stats import mode
import warnings
import random
import matplotlib.pyplot as plt
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
AVG_HUE_FILE = "./video_average_hue_normalized.json"
MED_HUE_FILE = "./video_median_hue_normalized.json"
EXCITING_FILE = "./video_Exciting_clean.json"
DURATION_FILE = "./video_Duration_new_raw.json"
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
        names_two_num[0].append(names_two[0][item])
    for item in idx_1:
        topics_0_1_0.append(topics_0_1[1][item])
        names_two_num[1].append(names_two[1][item])
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
            names_four_num[i].append(names_four[rating_val][item])
    new_list = topics_four_num[0] + topics_four_num[1] + topics_four_num[2] + topics_four_num[3]
    new_list_output = [1]*min_class + [2]*min_class + [4]*min_class + [5]*min_class
    new_names_list = names_four_num[0] + names_four_num[1] + names_four_num[2] + names_four_num[3]
    indices = [i for i in range(min_class*4)]
    np.random.shuffle(indices)
    output_items = [new_list[item] for item in indices]
    output_labels = [new_list_output[item] for item in indices]
    output_names = [new_names_list[item] for item in indices] # IDs
    return output_items, output_labels, output_names

def normalize(ids, labels, counts, ratings_range):
    output_video_ids = []
    output_video_labels = []
    lowest_index = -1
    for item in ratings_range:
        if lowest_index == -1 or counts[item] < counts[lowest_index]:
            lowest_index = item
    lowest_count = counts[lowest_index]
    for item in ratings_range:
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

with open(SENTIMENTS_CLEAN_FILE, "r") as sentiments_data_file:
    data = sentiments_data_file.read()
    sentiments_data = json.loads(data)

with open(TOPICS_CLEAN_FILE, "r") as topics_data_file:
    data = topics_data_file.read()
    topics_data = json.loads(data)

with open(VIDEO_EFFECTIVE_CLEAN_FILE, "r") as video_effective_data_clean:
    data = video_effective_data_clean.read()
    effective_data_clean = json.loads(data)

with open(MEM_FILE, "r") as memorability_data:
    data = memorability_data.read()
    mem_data = json.loads(data)

with open(OP_FLOW_FILE, "r") as optical_flow_data:
    data = optical_flow_data.read()
    opflow_data = json.loads(data)

with open(CROPPED_30_FILE, "r") as intensities_30percent_data:
    data = intensities_30percent_data.read()
    video_intensities_30percent = json.loads(data)

with open(CROPPED_60_FILE, "r") as intensities_60percent_data:
    data = intensities_60percent_data.read()
    video_intensities_60percent = json.loads(data)

with open(AVG_HUE_FILE, "r") as average_hue_data:
    data = average_hue_data.read()
    avg_hue_data = json.loads(data)
    new_avg_hue_data = {}
    '''for key in avg_hue_data:
        curr_data = avg_hue_data[key]
        min_data = min(curr_data)
        max_data = max(curr_data)
        new_data = [item / (255.0) for item in curr_data]
        new_avg_hue_data[key] = new_data
    with open("video_average_hue_normalized.json", "w+") as new_avg:
        json.dump(new_avg_hue_data, new_avg)'''

with open(MED_HUE_FILE, "r") as median_hue_data:
    data = median_hue_data.read()
    med_hue_data = json.loads(data)
    new_med_hue_data = {}
    '''for key in med_hue_data:
        curr_data = med_hue_data[key]
        new_data = [item / (255.0) for item in curr_data]
        new_med_hue_data[key] = new_data
    with open("video_median_hue_normalized.json", "w+") as new_med:
        json.dump(new_med_hue_data, new_med)'''

with open(EXCITING_FILE, "r") as exc_data:
    data = exc_data.read()
    exciting_data = json.loads(data)

with open(DURATION_FILE, "r") as dur_data:
    data = dur_data.read()
    duration_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
sentiments_data_stats = dict(sentiments_data) # Make a copy
topics_data_stats = dict(topics_data) # Make a copy

top = [0 for i in range(NUM_TOPICS)]
top_cou = [0 for i in range(NUM_TOPICS)]
sen = [0 for i in range(NUM_SENTIMENTS)]
sen_cou = [0 for i in range(NUM_SENTIMENTS)]

top_bars = [[0, 0, 0, 0, 0] for i in range(NUM_TOPICS)]
sen_bars = [[0, 0, 0, 0, 0] for i in range(NUM_SENTIMENTS)]

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

    # THIS BELOW STUFF IS JUST EXTRA FLUFF
    top[topic_val-1] += ratings_mean
    top_cou[topic_val-1] += 1
    sen[sentiment_val-1] += ratings_mean
    sen_cou[sentiment_val-1] += 1

    top_bars[topic_val-1][int(ratings_mean)-1] += 1
    sen_bars[sentiment_val-1][int(ratings_mean)-1] += 1

'''for i in range(len(top_bars)):
    plt.figure()
    plt.bar([1, 2, 3, 4, 5], top_bars[i])
    plt.title("Topic %d" % i)'''

'''for i in range(len(sen_bars)):
    plt.figure()
    plt.bar([1, 2, 3, 4, 5], sen_bars[i])
    plt.title("Topic %d" % i)

plt.show()'''

top = [top[i] / top_cou[i] for i in range(NUM_TOPICS)]
sen = [sen[i] / sen_cou[i] for i in range(NUM_SENTIMENTS)]
print(top)
print(sen)

video_ids = list(effective_data_stats.keys())

train_n = math.floor(len(video_ids) * .8)
test_n = len(video_ids) - train_n
train, test = train_test_split(video_ids, train_size = train_n, test_size = test_n)

test_classes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for id in test:
    test_classes[int(effective_data_stats[id])] += 1
print("test classes: %s" % test_classes)

train_topics, train_out, train_ids = normalize_data_five(train, effective_data_clean, topics_data_stats)
train_out_counts = collections.Counter(train_out)
wrong_labels = 0
for i in range(len(train_ids)):
    id = train_ids[i]
    if not int(effective_data_clean[id]) == train_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

test_topics, test_out, test_ids = normalize_data_five(test, effective_data_clean, topics_data_stats)
test_out_counts = collections.Counter(test_out)
wrong_labels = 0
for i in range(len(test_ids)):
    id = test_ids[i]
    if not int(effective_data_clean[id]) == test_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

train_sents = [sentiments_data_stats[x] for x in train_ids]
test_sents = [sentiments_data_stats[x] for x in test_ids]

topics_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
topics_SVM.fit(train_topics, train_out)
topics_score = topics_SVM.score(test_topics, test_out)
print("Topics SVM score: %.4f" % (topics_score))
#print(topics_SVM.n_support_[2])
#print(train_out_counts[2])
print(topics_SVM.n_support_)
print(train_out_counts)
print(test_out_counts)
topics_pred = topics_SVM.predict(test_topics)
#class_support_vectors_topics = normalize([[topics_SVM.n_support_[i] / train_out_counts[i+1] for i in range(0, len(topics_SVM.n_support_))]], norm='l1')
#print("Topics SVM Support Vectors: %s" % (class_support_vectors_topics))
#print("Weights: %s" % (topics_SVM.coef_))

sents_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
sents_SVM.fit(train_sents, train_out)
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
sents_DT.fit(train_sents, train_out)
sents_DT_pred = sents_DT.predict(test_sents)
print("Sentiments DT score: %.4f" % (sents_DT.score(test_sents, test_out)))

exciting_logress = LogisticRegression()
train_exciting = [[exciting_data[x]] for x in train_ids]
exciting_logress.fit(train_exciting, train_out)
test_exciting = [[exciting_data[x]] for x in test_ids]
exciting_pred = exciting_logress.predict(test_exciting)
print("Exciting score: %.4f" % (exciting_logress.score(test_exciting, test_out)))

mem_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
train_mem = [[mem_data[x]] for x in train_ids]
mem_SVM.fit(train_mem, train_out)
test_mem = [[mem_data[x]] for x in test_ids]
mem_SVM_pred = mem_SVM.predict(test_mem)
print("Mem SVM score: %.4f" % (mem_SVM.score(test_mem, test_out)))

opflow_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
train_opflow = [opflow_data[x] for x in train_ids]
opflow_SVM.fit(train_opflow, train_out)
test_opflow = [opflow_data[x] for x in test_ids]
opflow_SVM_pred = opflow_SVM.predict(test_opflow)
print("Opflow SVM score: %.4f" % (opflow_SVM.score(test_opflow, test_out)))

cropped_30_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
train_cropped_30 = [[video_intensities_30percent[x]] for x in train_ids]
cropped_30_SVM.fit(train_cropped_30, train_out)
test_cropped_30 = [[video_intensities_30percent[x]] for x in test_ids]
cropped_30_SVM_pred = cropped_30_SVM.predict(test_cropped_30)
print("cropped_30 SVM score: %.4f" % (cropped_30_SVM.score(test_cropped_30, test_out)))

cropped_60_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
train_cropped_60 = [[video_intensities_60percent[x]] for x in train_ids]
cropped_60_SVM.fit(train_cropped_60, train_out)
test_cropped_60 = [[video_intensities_60percent[x]] for x in test_ids]
cropped_60_SVM_pred = cropped_60_SVM.predict(test_cropped_60)
print("cropped_60 SVM score: %.4f" % (cropped_60_SVM.score(test_cropped_60, test_out)))

avg_hue_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_avg_hue = [avg_hue_data[x] for x in train_ids]
avg_hue_SVM.fit(train_avg_hue, train_out)
test_avg_hue = [avg_hue_data[x] for x in test_ids]
avg_hue_SVM_pred = avg_hue_SVM.predict(test_avg_hue)
print("Average Hue SVM score: %.4f" % (avg_hue_SVM.score(test_avg_hue, test_out)))

med_hue_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_med_hue = [med_hue_data[x] for x in train_ids]
med_hue_SVM.fit(train_med_hue, train_out)
test_med_hue = [med_hue_data[x] for x in test_ids]
med_hue_SVM_pred = med_hue_SVM.predict(test_med_hue)
print("Median Hue SVM score: %.4f" % (med_hue_SVM.score(test_med_hue, test_out)))

duration_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_duration = [[duration_data[x]] for x in train_ids]
duration_SVM.fit(train_duration, train_out)
test_duration = [[duration_data[x]] for x in test_ids]
duration_pred = duration_SVM.predict(test_duration)
print("Duration SVM score: %.4f" % (duration_SVM.score(test_duration, test_out)))

print(train_mem[0], train_opflow[0], train_cropped_30[0], train_avg_hue[0])
x = [list(train_topics[i]) + list(train_sents[i]) + train_mem[i] + train_opflow[i] + train_cropped_30[i] + train_cropped_60[i] + train_avg_hue[i] + train_med_hue[i] + train_exciting[i] for i in range(len(train_ids))]
total_SVM = SVC(kernel = 'rbf', decision_function_shape='ovr', C=1)
total_SVM.fit(x, train_out)
y = [list(test_topics[i]) + list(test_sents[i]) + test_mem[i] + test_opflow[i] + test_cropped_30[i] + test_cropped_60[i] + test_avg_hue[i] + test_med_hue[i] + test_exciting[i] for i in range(len(test_ids))]
total_SVM_pred = total_SVM.predict(y)
print("Total SVM score: %.4f" % (total_SVM.score(y, test_out)))

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
predictions_test_labels = [test_out[test_ids.index(x)] for x in predictions_test]
predictions_test_counts = {key : 0 for key in range(1, 6)}
for id in predictions_test:
    predictions_test_counts[test_out[test_ids.index(id)]] += 1
print("predictions test counts: %s" % predictions_test_counts)
predictions_test = normalize(predictions_test, predictions_test_labels, predictions_test_counts, [1, 2, 3, 4, 5])
predictions_test_counts = {key : 0 for key in range(1, 6)}
for id in predictions_test:
    predictions_test_counts[test_out[test_ids.index(id)]] += 1
print("predictions test counts: %s" % predictions_test_counts)
predictions_sub_train = [[], [], [], [], [], [], [], [], [], []]
#predictions_sub_train = []
predictions_sub_out = [[], [], [], [], [], [], [], [], [], []]

for sample in test_ids:
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
    exciting_class = exciting_pred[sample_index]
    total_svm_class = total_SVM_pred[sample_index]
    duration_class = duration_pred[sample_index]
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
        predictions_sub = [sents_svm_class, topics_svm_class, opflow_svm_class, cropped_30_class, mem_svm_class, med_hue_svm_class, topics_dt_class, sents_dt_class, cropped_60_class, avg_hue_svm_class]
        #predictions = [predictions_sub_clfs[0].predict(predictions_sub[0])[0], predictions_sub_clfs[1].predict(predictions_sub[1])[0], predictions_sub_clfs[2].predict(predictions_sub[2])[0], predictions_sub_clfs[3].predict(predictions_sub[3])[0], predictions_sub_clfs[4].predict(predictions_sub[4])[0], predictions_sub_clfs[5].predict(predictions_sub[5])[0], predictions_sub_clfs[6].predict(predictions_sub[6])[0], predictions_sub_clfs[7].predict(predictions[7])[0], predictions_sub_clfs[8].predict(predictions_sub[8])[0], predictions_sub_clfs[9].predict(predictions_sub[9])[0]]
        #print(predictions)
        #predictions = [(0 if sents_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if topics_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if opflow_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if cropped_30_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if mem_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if med_hue_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if topics_dt_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if sents_dt_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if cropped_60_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if avg_hue_svm_class == true_label else 1) + (random.random() - 0.5)*.6]
        predictions = [0 if sents_svm_class == true_label else 1, 0 if topics_svm_class == true_label else 1, 0 if opflow_svm_class == true_label else 1, 0 if cropped_30_class == true_label else 1, 0 if mem_svm_class == true_label else 1, 0 if med_hue_svm_class == true_label else 1, 0 if topics_dt_class == true_label else 1, 0 if sents_dt_class == true_label else 1, 0 if cropped_60_class == true_label else 1, 0 if avg_hue_svm_class == true_label else 1]
        #print(predictions)
        predictions_sub_out[0].append(predictions[0])
        predictions_sub_out[1].append(predictions[1])
        predictions_sub_out[2].append(predictions[2])
        predictions_sub_out[3].append(predictions[3])
        predictions_sub_out[4].append(predictions[4])
        predictions_sub_out[5].append(predictions[5])
        predictions_sub_out[6].append(predictions[6])
        predictions_sub_out[7].append(predictions[7])
        predictions_sub_out[8].append(predictions[8])
        predictions_sub_out[9].append(predictions[9])
        predictions_sub_train[0].append([predictions_sub[0]])
        predictions_sub_train[1].append([predictions_sub[1]])
        predictions_sub_train[2].append([predictions_sub[2]])
        predictions_sub_train[3].append([predictions_sub[3]])
        predictions_sub_train[4].append([predictions_sub[4]])
        predictions_sub_train[5].append([predictions_sub[5]])
        predictions_sub_train[6].append([predictions_sub[6]])
        predictions_sub_train[7].append([predictions_sub[7]])
        predictions_sub_train[8].append([predictions_sub[8]])
        predictions_sub_train[9].append([predictions_sub[9]])
        predictions_in.append(predictions)
        predictions_out.append(true_label)
        predictions_sub_train.append(predictions_sub)

    #class_counts = collections.Counter([opflow_svm_class, cropped_30_class, cropped_60_class])
    class_counts = collections.Counter([opflow_svm_class, sents_svm_class, topics_svm_class, exciting_class, total_svm_class])
    predicted_class = class_counts.most_common(1)[0][0]
    total += 1

    if predicted_class == true_label:
        topics_correct[topic] += 1
        sents_correct[sent] += 1
        correct += 1

predictions_clf = SVC()
predictions_clf.fit(predictions_in, predictions_out)
predictions_sub_clfs = [SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC()]
#predictions_sub_clfs = [LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression()]
for i in range(len(predictions_sub_clfs)):
    predictions_sub_clfs[i].fit(predictions_sub_train[i], predictions_sub_out[i])
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
classifications = {i : 0 for i in range(1, 6)}
misclassifications = {i : 0 for i in range(1, 6)}
for sample in test_ids:
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
        predictions_sub_test = [topics_svm_class, sents_svm_class, topics_dt_class, sents_dt_class, mem_svm_class, opflow_svm_class, avg_hue_svm_class, med_hue_svm_class, cropped_30_class, cropped_60_class]
        predictions_sub = [predictions_sub_clfs[0].predict(predictions_sub_test[0])[0], predictions_sub_clfs[1].predict(predictions_sub_test[1])[0], predictions_sub_clfs[2].predict(predictions_sub_test[2])[0], predictions_sub_clfs[3].predict(predictions_sub_test[3])[0], predictions_sub_clfs[4].predict(predictions_sub_test[4])[0], predictions_sub_clfs[5].predict(predictions_sub_test[5])[0], predictions_sub_clfs[6].predict(predictions_sub_test[6])[0], predictions_sub_clfs[7].predict(predictions_sub_test[7])[0], predictions_sub_clfs[8].predict(predictions_sub_test[8])[0], predictions_sub_clfs[9].predict(predictions_sub_test[9])[0]]
        print(predictions_sub)
        #predictions = [0 if sents_svm_class == true_label else 1, 0 if topics_svm_class == true_label else 1, 0 if opflow_svm_class == true_label else 1, 0 if cropped_30_class == true_label else 1, 0 if mem_svm_class == true_label else 1, 0 if med_hue_svm_class == true_label else 1, 0 if topics_dt_class == true_label else 1, 0 if sents_dt_class == true_label else 1, 0 if cropped_60_class == true_label else 1, 0 if avg_hue_svm_class == true_label else 1]
        predicted_label = predictions_clf.predict([predictions_sub])
        if predicted_label == true_label:
            classifications[true_label] += 1
            predictions_correct += 1
        else:
            misclassifications[true_label] += 1
            #print("sample: %s, predicted: %d, ground-truth: %d" % (sample, predicted_label, true_label))
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
