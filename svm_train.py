import json, os, math, collections, pickle, warnings, random
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import pearsonr
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from collections import Counter
from new_look_code.constants import *
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

'''def normalize_data_binary(names, labels, stats):
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
    return output_items, output_labels, output_names'''

def trainSVM(data, train_y, test_y, train_ids, test_ids, wrap_in_list, kernel, degree, decision_function_shape, C, name):
    if wrap_in_list:
        train_x = [[data[x]] for x in train_ids if x in data]
        test_x = [[data[x]] for x in test_ids if x in data]
    else:
        train_x = [data[x] for x in train_ids if x in data]
        test_x = [data[x] for x in test_ids if x in data]
    SVM = SVC(kernel=kernel, degree=degree, decision_function_shape=decision_function_shape, C=C)
    SVM.fit(train_x, train_y)
    pred = SVM.predict(test_x)
    pred_train = SVM.predict(train_x)
    score = SVM.score(test_x, test_y)
    print(name + " SVM score: %.4f" % (score))
    return pred, pred_train, train_x, test_x

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

def find_most_common_words(train_ids):
    most_common_words_master = {}
    for id in train_ids:
        video_text_data = ocr_data_clean[id]
        for word in video_text_data:
            if word in most_common_words_master:
                most_common_words_master[word] += 1
            elif not word in useless_words:
                most_common_words_master[word] = 1
    most_common_counter = Counter(most_common_words_master)
    most_common_words_master = []
    for word, count in most_common_counter.most_common(100): # grab top 50 words
        most_common_words_master.append(word)
    return most_common_words_master

def get_word_count_train_test(train_ids, test_ids):
    most_common_words_master = find_most_common_words(train_ids)
    word_count_train_in = []
    word_count_test_in = []
    for id in train_ids:
        video_text_data = ocr_data_clean[id]
        word_count = []
        for word in most_common_words_master:
            if word in video_text_data:
                word_count.append(1)
            else:
                word_count.append(0)
        word_count_train_in.append(word_count)
    for id in test_ids:
        video_text_data = ocr_data_clean[id]
        word_count = []
        for word in most_common_words_master:
            if word in video_text_data:
                word_count.append(1)
            else:
                word_count.append(0)
        word_count_test_in.append(word_count)
    return word_count_train_in, word_count_test_in

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

with open(FUNNY_CLEAN_FILE, "r") as funny_data_clean:
    data = funny_data_clean.read()
    funny_data_clean = json.loads(data)

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

with open(VIDEO_OCR_STATS_FILE, "r") as video_ocr_data:
    data = video_ocr_data.read()
    ocr_data = json.loads(data)

with open(VIDEO_OCR_WORDS_FILE, "r") as ocr_file:
    ocr_data_clean = json.loads(ocr_file.read())

with open(AUDIO_FILE, "r") as audio_file:
    audio_stats = json.loads(audio_file.read())

with open(EXPRESSIONS_FILE, "r") as expressions_file:
    expressions_stats = json.loads(expressions_file.read())

with open(PLACES_FILE, "r") as places_file:
    places_stats = json.loads(places_file.read())

with open(EMOTIONS_FILE, "r") as emotions_file:
    emotions_stats = json.loads(emotions_file.read())

with open(CLIMAX_FILE, "r") as climax_file:
    climax_stats = json.loads(climax_file.read())

with open(OBJECTS_FILE, "r") as objects_file:
    objects_stats = json.loads(objects_file.read())

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
#print(top)
#print(sen)

video_ids = list(effective_data_stats.keys())

train_n = math.floor(len(video_ids) * .8)
test_n = len(video_ids) - train_n
train, test = train_test_split(video_ids, train_size = train_n, test_size = test_n)

test_classes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for id in test:
    test_classes[int(effective_data_stats[id])] += 1
print("test classes: %s" % test_classes)

train_topics, train_out, train_ids = normalize_data_binary(train, effective_data_clean, topics_data_stats)
train_out_counts = collections.Counter(train_out)
wrong_labels = 0
for i in range(len(train_ids)):
    id = train_ids[i]
    if not int(effective_data_clean[id]) == train_out[i]:
        wrong_labels += 1
print("Wrong labels: %d" % wrong_labels)

test_topics, test_out, test_ids = normalize_data_binary(test, effective_data_clean, topics_data_stats)
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
topics_pred_train = topics_SVM.predict(train_topics)
#class_support_vectors_topics = normalize([[topics_SVM.n_support_[i] / train_out_counts[i+1] for i in range(0, len(topics_SVM.n_support_))]], norm='l1')
#print("Topics SVM Support Vectors: %s" % (class_support_vectors_topics))
#print("Weights: %s" % (topics_SVM.coef_))

sents_SVM = SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1)
sents_SVM.fit(train_sents, train_out)
sents_score = sents_SVM.score(test_sents, test_out)
print("Sentiments SVM score: %.4f" % (sents_score))
sents_pred = sents_SVM.predict(test_sents)
sents_pred_train = sents_SVM.predict(train_sents)
#class_support_vectors_sentiments = normalize([[sents_SVM.n_support_[i] / train_sents_out_counts[i+1] for i in range(0, len(sents_SVM.n_support_))]], norm='l1')
#print("Sentiments SVM Support Vectors: %s" % (class_support_vectors_sentiments))
#print("Weights: %s" % (sents_SVM.coef_))

topics_DT = DecisionTreeClassifier()
topics_DT.fit(train_topics, train_out)
topics_DT_pred = topics_DT.predict(test_topics)
topics_DT_pred_train = topics_DT.predict(train_topics)
print("Topics DT score: %.4f" % (topics_DT.score(test_topics, test_out)))

sents_DT = DecisionTreeClassifier()
sents_DT.fit(train_sents, train_out)
sents_DT_pred = sents_DT.predict(test_sents)
sents_DT_pred_train = sents_DT.predict(train_sents)
print("Sentiments DT score: %.4f" % (sents_DT.score(test_sents, test_out)))

exciting_logress = LogisticRegression()
train_exciting = [[exciting_data[x]] for x in train_ids]
exciting_logress.fit(train_exciting, train_out)
test_exciting = [[exciting_data[x]] for x in test_ids]
exciting_pred = exciting_logress.predict(test_exciting)
exciting_pred_train = exciting_logress.predict(train_exciting)
print("Exciting score: %.4f" % (exciting_logress.score(test_exciting, test_out)))

exciting_SVM_pred, exciting_SVM_pred_train, train_exciting, test_exciting = trainSVM(exciting_data, train_out, test_out, train_ids, test_ids, True, 'linear', 1, 'ovr', 1, "Exciting")
mem_SVM_pred, mem_SVM_pred_train, train_mem, test_mem = trainSVM(mem_data, train_out, test_out, train_ids, test_ids, True, 'poly', 7, 'ovr', 1, "Mem")
opflow_SVM_pred, opflow_SVM_pred_train, train_opflow, test_opflow = trainSVM(opflow_data, train_out, test_out, train_ids, test_ids, False, 'poly', 7, 'ovr', 1, "Opflow")
cropped_30_SVM_pred, cropped_30_SVM_pred_train, train_cropped_30, test_cropped_30 = trainSVM(video_intensities_30percent, train_out, test_out, train_ids, test_ids, True, 'poly', 7, 'ovr', 1, "cropped_30")
cropped_60_SVM_pred, cropped_60_SVM_pred_train, train_cropped_60, test_cropped_60 = trainSVM(video_intensities_60percent, train_out, test_out, train_ids, test_ids, True, 'poly', 7, 'ovr', 1, "cropped_60")
avg_hue_SVM_pred, avg_hue_SVM_pred_train, train_avg_hue, test_avg_hue = trainSVM(avg_hue_data, train_out, test_out, train_ids, test_ids, False, 'rbf', 1, 'ovr', 1, "Average Hue")
med_hue_SVM_pred, med_hue_SVM_pred_train, train_med_hue, test_med_hue = trainSVM(med_hue_data, train_out, test_out, train_ids, test_ids, False, 'rbf', 1, 'ovr', 1, "Median Hue")
duration_pred, duration_pred_train, train_duration, test_duration = trainSVM(duration_data, train_out, test_out, train_ids, test_ids, True, 'rbf', 1, 'ovr', 1, "Duration")

text_length_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_text_length = [[ocr_data[x][0]] for x in train_ids]
text_length_SVM.fit(train_text_length, train_out)
test_text_length = [[ocr_data[x][0]] for x in test_ids]
text_length_pred = text_length_SVM.predict(test_text_length)
text_length_pred_train = text_length_SVM.predict(train_text_length)
print("Text Length SVM score: %.4f" % (text_length_SVM.score(test_text_length, test_out)))

meaningful_words_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_meaningfulness = [[ocr_data[x][1]] for x in train_ids]
meaningful_words_SVM.fit(train_meaningfulness, train_out)
test_meaningfulness = [[ocr_data[x][1]] for x in test_ids]
meaningfulness_pred = meaningful_words_SVM.predict(test_meaningfulness)
meaningfulness_pred_train = meaningful_words_SVM.predict(train_meaningfulness)
print("Meaningful Words SVM score: %.4f" % (meaningful_words_SVM.score(test_meaningfulness, test_out)))

avg_word_len_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_avg_word_len = [[ocr_data[x][2]] for x in train_ids]
avg_word_len_SVM.fit(train_avg_word_len, train_out)
test_avg_word_len = [[ocr_data[x][2]] for x in test_ids]
avg_word_len_pred = avg_word_len_SVM.predict(test_avg_word_len)
avg_word_len_pred_train = avg_word_len_SVM.predict(train_avg_word_len)
print("Avg. Word Len SVM score: %.4f" % (avg_word_len_SVM.score(test_avg_word_len, test_out)))

sent_anal_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_sent_anal = [ocr_data[x][3] for x in train_ids]
sent_anal_SVM.fit(train_sent_anal, train_out)
test_sent_anal = [ocr_data[x][3] for x in test_ids]
sent_anal_pred = sent_anal_SVM.predict(test_sent_anal)
sent_anal_pred_train = sent_anal_SVM.predict(train_sent_anal)
print("Text Sentiment Analysis SVM score: %.4f" % (sent_anal_SVM.score(test_sent_anal, test_out)))

text_SVM = SVC(kernel='rbf', decision_function_shape='ovr', C=1)
train_text = [[train_text_length[i][0], train_meaningfulness[i][0], train_avg_word_len[i][0], train_sent_anal[i][0], train_sent_anal[i][1], train_sent_anal[i][2]] for i in range(len(train_ids))]
text_SVM.fit(train_text, train_out)
test_text = [[test_text_length[i][0], test_meaningfulness[i][0], test_avg_word_len[i][0], test_sent_anal[i][0], test_sent_anal[i][1], test_sent_anal[i][2]] for i in range(len(test_ids))]
text_pred = text_SVM.predict(test_text)
text_pred_train = text_SVM.predict(train_text)
print("Text SVM score: %.4f" % (text_SVM.score(test_text, test_out)))

x = [list(train_topics[i]) + list(train_sents[i]) + train_mem[i] + train_opflow[i] + train_cropped_30[i] + train_cropped_60[i] + train_avg_hue[i] + train_med_hue[i] + train_exciting[i] for i in range(len(train_ids))]
total_SVM = SVC(kernel = 'rbf', decision_function_shape='ovr', C=1)
total_SVM.fit(x, train_out)
y = [list(test_topics[i]) + list(test_sents[i]) + test_mem[i] + test_opflow[i] + test_cropped_30[i] + test_cropped_60[i] + test_avg_hue[i] + test_med_hue[i] + test_exciting[i] for i in range(len(test_ids))]
total_SVM_pred = total_SVM.predict(y)
total_SVM_pred_train = total_SVM.predict(x)
print("Total SVM score: %.4f" % (total_SVM.score(y, test_out)))

word_count_train_in, word_count_test_in = get_word_count_train_test(train_ids, test_ids)
word_count_SVC = SVC()
word_count_SVC.fit(word_count_train_in, train_out)
word_count_pred = word_count_SVC.predict(word_count_test_in)
word_count_pred_train = word_count_SVC.predict(word_count_train_in)
word_count_score = word_count_SVC.score(word_count_test_in, test_out)
print("Word count SVC score: %.4f" % word_count_score)

audio_pred, audio_pred_train, train_audio, test_audio = trainSVM(audio_stats, train_out, test_out, train_ids, test_ids, True, 'linear', 1, 'ovr', 1, "Audio")
objects_pred, objects_pred_train, train_objects, test_objects = trainSVM(objects_stats, train_out, test_out, train_ids, test_ids, False, 'linear', 1, 'ovr', 1, "Objects")
places_pred, places_pred_train, train_places, test_places = trainSVM(places_stats, train_out, test_out, train_ids, test_ids, False, 'linear', 1, 'ovr', 1, "Places")
expressions_pred, expressions_pred_train, train_expressions, test_expressions = trainSVM(expressions_stats, train_out, test_out, train_ids, test_ids, False, 'linear', 1, 'ovr', 1, "Expressions")
emotions_pred, emotions_pred_train, train_emotions, test_emotions = trainSVM(emotions_stats, train_out, test_out, train_ids, test_ids, False, 'linear', 1, 'ovr', 1, "Emotions")

train_climax_out = [[effective_data_clean[x]] for x in train_ids if x in climax_stats]
test_climax_out = [[effective_data_clean[x]] for x in test_ids if x in climax_stats]
climax_pred, climax_pred_train, train_climax, test_climax = trainSVM(climax_stats, train_climax_out, test_climax_out, train_ids, test_ids, True, 'linear', 1, 'ovo', 1, "Climax")

funny_pred, funny_pred_train, train_funny, test_funny = trainSVM(funny_data_clean, train_out, test_out, train_ids, test_ids, True, 'rbf', 1, 'ovr', 1, "Funny")

x = [list(train_topics[i]) + list(train_sents[i]) + train_avg_hue[i] + train_duration[i] + train_audio[i] + train_places[i] + train_exciting[i] + train_funny[i] for i in range(len(train_ids))]
kovashka_SVM = SVC(kernel = 'rbf', decision_function_shape='ovr', C=1)
kovashka_SVM.fit(x, train_out)
y = [list(test_topics[i]) + list(test_sents[i]) + test_avg_hue[i] + test_duration[i] + test_audio[i] + test_places[i] + test_exciting[i] + test_funny[i] for i in range(len(test_ids))]
kovashka_SVM_pred = kovashka_SVM.predict(y)
kovashka_SVM_pred_train = kovashka_SVM.predict(x)
print("Kovashka SVM score: %.4f" % (kovashka_SVM.score(y, test_out)))

kovashka_adaboost = AdaBoostClassifier()
kovashka_adaboost.fit(x, train_out)
print("Kovashka Adaboost score: %.4f" % (kovashka_adaboost.score(y, test_out)))
kovashka_adaboost_pred = kovashka_adaboost.predict(y)

kovashka_bagging = BaggingClassifier()
kovashka_bagging.fit(x, train_out)
print("Kovashka Bagging score: %.4f" % (kovashka_bagging.score(y, test_out)))

kovashka_gradientboosting = GradientBoostingClassifier()
kovashka_gradientboosting.fit(x, train_out)
print("Kovashka Gradient Boosting score: %.4f" % (kovashka_gradientboosting.score(y, test_out)))
kovashka_gradientboost_pred = kovashka_gradientboosting.predict(y)

kovashka_random_forest = RandomForestClassifier()
kovashka_random_forest.fit(x, train_out)
print("Kovashka Random Forest score: %.4f" % (kovashka_random_forest.score(y, test_out)))

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

text_len_topics_correct = np.zeros(38)
text_len_sents_correct = np.zeros(30)
meaningfulness_topics_correct = np.zeros(38)
meaningfulness_sents_correct = np.zeros(30)
avg_word_len_topics_correct = np.zeros(38)
avg_word_len_sents_correct = np.zeros(30)
sent_anal_topics_correct = np.zeros(38)
sent_anal_sents_correct = np.zeros(30)
duration_topics_correct = np.zeros(38)
duration_sents_correct = np.zeros(30)
word_count_topics_correct = np.zeros(38)
word_count_sents_correct = np.zeros(30)

audio_topics_correct = np.zeros(38)
audio_sents_correct = np.zeros(30)
objects_topics_correct = np.zeros(38)
objects_sents_correct = np.zeros(30)
places_topics_correct = np.zeros(38)
places_sents_correct = np.zeros(30)
expressions_topics_correct = np.zeros(38)
expressions_sents_correct = np.zeros(30)
emotions_topics_correct = np.zeros(38)
emotions_sents_correct = np.zeros(30)
climax_topics_correct = np.zeros(38)
climax_sents_correct = np.zeros(30)

correct = 0
total = 0

for sample in train_ids:
    sample_index = train_ids.index(sample)
    topics_svm_class = topics_pred_train[sample_index]
    sents_svm_class = sents_pred_train[sample_index]
    topics_dt_class = topics_DT_pred_train[sample_index]
    sents_dt_class = sents_DT_pred_train[sample_index]
    mem_svm_class = mem_SVM_pred_train[sample_index]
    opflow_svm_class = opflow_SVM_pred_train[sample_index]
    avg_hue_svm_class = avg_hue_SVM_pred_train[sample_index]
    med_hue_svm_class = med_hue_SVM_pred_train[sample_index]
    cropped_30_class = cropped_30_SVM_pred_train[sample_index]
    cropped_60_class = cropped_60_SVM_pred_train[sample_index]
    exciting_class = exciting_pred_train[sample_index]
    total_svm_class = total_SVM_pred_train[sample_index]
    text_length_class = text_length_pred_train[sample_index]
    meaningful_words_class = meaningfulness_pred_train[sample_index]
    avg_word_len_class = avg_word_len_pred_train[sample_index]
    sent_anal_class = sent_anal_pred_train[sample_index]
    duration_class = duration_pred_train[sample_index]
    word_count_class = word_count_pred_train[sample_index]
    audio_class = audio_pred_train[sample_index]
    objects_class = objects_pred_train[sample_index]
    places_class = places_pred_train[sample_index]
    expressions_class = expressions_pred_train[sample_index]
    emotions_class = emotions_pred_train[sample_index]
    #climax_class = climax_pred[sample_index]
    true_label = train_out[sample_index]

    topic = list(train_topics[sample_index]).index(1)
    sent = list(train_sents[sample_index]).index(1)
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
    if text_length_class == true_label:
        text_len_topics_correct[topic] += 1
        text_len_sents_correct[sent] += 1
    if meaningful_words_class == true_label:
        meaningfulness_topics_correct[topic] += 1
        meaningfulness_sents_correct[sent] += 1
    if avg_word_len_class == true_label:
        avg_word_len_topics_correct[topic] += 1
        avg_word_len_sents_correct[sent] += 1
    if sent_anal_class == true_label:
        sent_anal_topics_correct[topic] += 1
        sent_anal_sents_correct[sent] += 1
    if duration_class == true_label:
        duration_topics_correct[topic] += 1
        duration_sents_correct[sent] += 1
    if word_count_class == true_label:
        word_count_topics_correct[topic] += 1
        word_count_sents_correct[sent] += 1
    if audio_class == true_label:
        audio_topics_correct[topic] += 1
        audio_sents_correct[sent] += 1
    if objects_class == true_label:
        objects_topics_correct[topic] += 1
        objects_sents_correct[sent] += 1
    if places_class == true_label:
        places_topics_correct[topic] += 1
        places_sents_correct[sent] += 1
    if expressions_class == true_label:
        expressions_topics_correct[topic] += 1
        expressions_sents_correct[sent] += 1
    if emotions_class == true_label:
        emotions_topics_correct[topic] += 1
        emotions_sents_correct[sent] += 1
    #if climax_class == true_label:
    #    climax_topics_correct[topic] += 1
    #    climax_sents_correct[sent] += 1
    topics_totals[topic] += 1
    sents_totals[sent] += 1

sents_svm_correct = np.array([sents_svm_correct[i] / sents_totals[i] for i in range(30)])
topics_svm_correct = np.array([topics_svm_correct[i] / topics_totals[i] for i in range(38)])
opflow_topics_correct = np.array([opflow_topics_correct[i] / topics_totals[i] for i in range(38)])
opflow_sents_correct = np.array([opflow_sents_correct[i] / sents_totals[i] for i in range(30)])
cropped_topics_correct = np.array([cropped_topics_correct[i] / topics_totals[i] for i in range(38)])
cropped_sents_correct = np.array([cropped_sents_correct[i] / sents_totals[i] for i in range(30)])
text_len_topics_correct = np.array([text_len_topics_correct[i] / topics_totals[i] for i in range(38)])
text_len_sents_correct = np.array([text_len_sents_correct[i] / sents_totals[i] for i in range(30)])
meaningfulness_topics_correct = np.array([meaningfulness_topics_correct[i] / topics_totals[i] for i in range(38)])
meaningfulness_sents_correct = np.array([meaningfulness_sents_correct[i] / sents_totals[i] for i in range(30)])
avg_word_len_topics_correct = np.array([avg_word_len_topics_correct[i] / topics_totals[i] for i in range(38)])
avg_word_len_sents_correct = np.array([avg_word_len_sents_correct[i] / sents_totals[i] for i in range(30)])
sent_anal_topics_correct = np.array([sent_anal_topics_correct[i] / topics_totals[i] for i in range(38)])
sent_anal_sents_correct = np.array([sent_anal_sents_correct[i] / sents_totals[i] for i in range(30)])
duration_topics_correct = np.array([duration_topics_correct[i] / topics_totals[i] for i in range(38)])
duration_sents_correct = np.array([duration_sents_correct[i] / sents_totals[i] for i in range(30)])
word_count_topics_correct = np.array([word_count_topics_correct[i] / topics_totals[i] for i in range(38)])
word_count_sents_correct = np.array([word_count_sents_correct[i] / sents_totals[i] for i in range(30)])
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

audio_topics_correct = np.array([audio_topics_correct[i] / topics_totals[i] for i in range(38)])
audio_sents_correct = np.array([audio_sents_correct[i] / sents_totals[i] for i in range(30)])
objects_topics_correct = np.array([objects_topics_correct[i] / topics_totals[i] for i in range(38)])
objects_sents_correct = np.array([objects_sents_correct[i] / sents_totals[i] for i in range(30)])
places_topics_correct = np.array([places_topics_correct[i] / topics_totals[i] for i in range(38)])
places_sents_correct = np.array([places_sents_correct[i] / sents_totals[i] for i in range(30)])
expressions_topics_correct = np.array([expressions_topics_correct[i] / topics_totals[i] for i in range(38)])
expressions_sents_correct = np.array([expressions_sents_correct[i] / sents_totals[i] for i in range(30)])
emotions_topics_correct = np.array([emotions_topics_correct[i] / topics_totals[i] for i in range(38)])
emotions_sents_correct = np.array([emotions_sents_correct[i] / sents_totals[i] for i in range(30)])
#climax_topics_correct = np.array([climax_topics_correct[i] / topics_totals[i] for i in range(38)])
#climax_sents_correct = np.array([climax_sents_correct[i] / sents_totals[i] for i in range(30)])

correct = 0
voting_correct = 0
true_voting_correct = 0
final_correct = 0
total = 0
predicted_labels = []
true_labels = []
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
    duration_class = duration_pred[sample_index]
    text_length_class = text_length_pred[sample_index]
    meaningful_words_class = meaningfulness_pred[sample_index]
    avg_word_len_class = avg_word_len_pred[sample_index]
    sent_anal_class = sent_anal_pred[sample_index]
    word_count_class = word_count_pred[sample_index]
    audio_class = audio_pred[sample_index]
    objects_class = objects_pred[sample_index]
    places_class = places_pred[sample_index]
    expressions_class = expressions_pred[sample_index]
    emotions_class = emotions_pred[sample_index]
    kovashka_adaboost_class = kovashka_adaboost_pred[sample_index]
    kovashka_gradientboost_class = kovashka_gradientboost_pred[sample_index]
    total_SVM_class = total_SVM_pred[sample_index]
    exciting_class = exciting_pred[sample_index]
    #climax_class = climax_pred[sample_index]
    true_label = test_out[sample_index]
    predicted_class = -1

    topic = list(test_topics[sample_index]).index(1)
    sent = list(test_sents[sample_index]).index(1)
    total += 1

    sents_scores = [sents_svm_correct[sent], opflow_sents_correct[sent], cropped_sents_correct[sent], sents_dt_correct[sent], mem_sents_correct[sent], med_hue_sents_correct[sent], duration_sents_correct[sent], word_count_sents_correct[sent], meaningfulness_sents_correct[sent], avg_word_len_sents_correct[sent], sent_anal_sents_correct[sent], audio_sents_correct[sent], objects_sents_correct[sent], places_sents_correct[sent], expressions_sents_correct[sent], emotions_sents_correct[sent]]
    topics_scores = [topics_svm_correct[topic], opflow_topics_correct[topic], cropped_topics_correct[topic], topics_dt_correct[topic], mem_topics_correct[topic], med_hue_topics_correct[topic], duration_topics_correct[topic], word_count_topics_correct[topic], meaningfulness_topics_correct[topic], avg_word_len_topics_correct[topic], sent_anal_topics_correct[topic], audio_topics_correct[topic], objects_topics_correct[topic], places_topics_correct[topic], expressions_topics_correct[topic], emotions_topics_correct[topic]]

    classes = [sents_svm_class, opflow_svm_class, cropped_30_class, sents_dt_class, mem_svm_class, med_hue_svm_class, duration_class, word_count_class, meaningful_words_class, avg_word_len_class, sent_anal_class, audio_class, objects_class, places_class, expressions_class, emotions_class, topics_svm_class, opflow_svm_class, cropped_30_class, topics_dt_class, mem_svm_class, med_hue_svm_class, duration_class, word_count_class, meaningful_words_class, avg_word_len_class, sent_anal_class, audio_class, objects_class, places_class, expressions_class, emotions_class]

    scores = sents_scores + topics_scores
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-5:]
    top_preds = [classes[i] for i in top_indices]
    voting_pred = mode(top_preds)[0][0]

    #predictions = [topics_svm_class, sents_svm_class, topics_dt_class, sents_dt_class, mem_svm_class, opflow_svm_class, avg_hue_svm_class, med_hue_svm_class, cropped_30_class, cropped_60_class, duration_class, text_length_class, meaningful_words_class, avg_word_len_class, sent_anal_class, word_count_class, audio_class, objects_class, places_class, expressions_class, emotions_class]
    predictions = [exciting_class, audio_class, kovashka_gradientboost_class, topics_dt_class, total_svm_class]
    #predictions = [exciting_class, sent_anal_class, audio_class, kovashka_adaboost_class, topics_dt_class, total_svm_class, duration_class]
    true_voting_pred = mode(predictions)[0][0]

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

    if voting_pred == true_label:
        voting_correct += 1
    if true_voting_pred == true_label:
        true_voting_correct += 1

    final_pred = mode([predicted_class, voting_pred, true_voting_pred])[0][0]
    if final_pred == true_label:
        final_correct += 1

    predicted_labels.append(kovashka_gradientboost_class)
    true_labels.append(true_label)

print("Combiner accuracy (NEW): %.4f (%d correct, %d total)" % (correct/total, correct, total))
print("Voting accuracy (NEW): %.4f (%d correct, %d total)" % (voting_correct/total, voting_correct, total))
print("True Voting accuracy (NEW): %.4f (%d correct, %d total)" % (true_voting_correct/total, true_voting_correct, total))
print("Final accuracy (NEW): %.4f (%d correct, %d total)" % (final_correct/total, true_voting_correct, total))
#print(predictions_clf.coef_)

print("Number of video ids: %d" % (len(video_ids)))

plt.figure()
confusion_matrix = confusion_matrix(true_labels, predicted_labels)
heatmap = sb.heatmap(confusion_matrix, xticklabels = ["non-effective", "effective"], yticklabels = ["non-effective", "effective"], annot = True, cmap = sb.color_palette("Reds"))
heatmap.invert_yaxis()
plt.savefig("binary_classification_heatmap.png")

plt.show()
