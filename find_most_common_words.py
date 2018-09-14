import json
import numpy as np
from scipy.stats import pearsonr
import requests
import os
from sklearn.svm import SVC
import math
from sklearn.model_selection import train_test_split
from collections import Counter

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

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

with open("./video_ocr_data_clean_words.json", "r") as OCR_FILE:
    ocr_data_clean = json.loads(OCR_FILE.read())

useless_words = ["a", "of", "the", "to", "on", "with", "your", "is", "for", "this", "in", "how", "by", "and", "it", "you", "i", "e", "ry", "an", "not", "s", "its", "what", "are", "get", "be", "o", "at", "have", "as", "no", "do", "am", "me", "de", "my", "am", "wwwesrborg", "ed", "has", "int", "th", "com", "that", "who", "st", "y", "co", "ism", "ma", "sec", "knorr", "presents", "us", "n", "c", "l", "f", "tm", "al", "x", "v", "d", "el"]
useless_words += ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

VIDEO_SENTIMENTS = {}
VIDEO_OCR_STATS = {}

most_common_words = [{}, {}, {}, {}, {}]
num_samples_per_class = [0, 0, 0, 0, 0]
num_words_per_class = [0, 0, 0, 0, 0]
num_meaningful_words_per_class = [0, 0, 0, 0, 0]
num_unique_words_per_class = [0, 0, 0, 0, 0]
avg_word_len_per_class = [0, 0, 0, 0, 0]
sentiment_per_class = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
word_sentiments = {}

if os.path.exists("word_sentiments.json"):
    with open("./word_sentiments.json", "r") as WORD_SENTIMENTS_FILE:
        word_sentiments = json.loads(WORD_SENTIMENTS_FILE.read())

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = int(round(ratings_mean, 3))

    video_text_data = ocr_data_clean[video_id]
    num_words = len(video_text_data)
    num_words_per_class[ratings_mean - 1] += num_words
    num_samples_per_class[ratings_mean - 1] += 1
    average_word_length = 0.0
    average_sentiment = [0.0, 0.0, 0.0]
    meaningful_words = 0

    for word in video_text_data:
        average_word_length += len(word)
        if word in most_common_words[ratings_mean - 1]:
            most_common_words[ratings_mean - 1][word] += 1
        elif not word in useless_words:
            if word not in word_sentiments:
                word_sentiment = requests.post("https://japerk-text-processing.p.mashape.com/sentiment/", headers = {"X-Mashape-Key": "wPM0D3hTS0mshdoCY04bsAXKyFxqp1RhdJtjsnjhOwQ6K2B1dD"}, data = {"text": word})
                if word_sentiment.status_code == 200:
                    word_sentiment = word_sentiment.json()["probability"]
                    word_sentiment = [word_sentiment["neg"], word_sentiment["neutral"], word_sentiment["pos"]]
                    word_sentiments[word] = word_sentiment
            average_sentiment = np.add(average_sentiment, word_sentiments[word])
            meaningful_words += 1
            most_common_words[ratings_mean - 1][word] = 1
    average_word_length = average_word_length / num_words if num_words > 0 else 0
    average_sentiment = np.divide(average_sentiment, meaningful_words) if meaningful_words > 1 else average_sentiment

    avg_word_len_per_class[ratings_mean - 1] += average_word_length
    num_meaningful_words_per_class[ratings_mean - 1] += meaningful_words
    sentiment_per_class[ratings_mean - 1] = np.add(sentiment_per_class[ratings_mean - 1], average_sentiment)
    VIDEO_SENTIMENTS[video_id] = list(average_sentiment)
    VIDEO_OCR_STATS[video_id] = [num_words, meaningful_words, average_word_length, list(average_sentiment)]

i = 1
for effectiveness_bin in most_common_words:
    sorted_dict = sorted(effectiveness_bin.items(), key = lambda x : -x[1])[:15]
    print("\nEffectiveness Bin %d" % i)
    print("====================")
    for item in sorted_dict:
        print("%s: %d" % (item[0], item[1]))
    i += 1

num_words_per_class = [num_words_per_class[i] / num_samples_per_class[i] for i in range(5)]
num_meaningful_words_per_class = [num_meaningful_words_per_class[i] / num_samples_per_class[i] for i in range(5)]
avg_word_len_per_class = [avg_word_len_per_class[i] / num_samples_per_class[i] for i in range(5)]
sentiment_per_class = [list(np.divide(sentiment_per_class[i], num_samples_per_class[i])) for i in range(5)]
print("\nWords: %s" % (num_words_per_class))
print("Meaningful words: %s" % (num_meaningful_words_per_class))
print("Average word length: %s" % (avg_word_len_per_class))
print("Sentiment per class: %s" % (sentiment_per_class))
num_words_and_effectiveness_corr = pearsonr([1, 2, 3, 4, 5], num_words_per_class)[0]
num_meaningful_words_and_effectiveness_corr = pearsonr([1, 2, 3, 4, 5], num_meaningful_words_per_class)[0]
avg_word_len_and_effectiveness_corr = pearsonr([1, 2, 3, 4, 5], avg_word_len_per_class)[0]
print("Correlation for number of words and effectiveness: %f" % (num_words_and_effectiveness_corr))
print("Correlation for number of meaningful words and effectiveness: %f" % (num_meaningful_words_and_effectiveness_corr))
print("Correlation for avg number of word length and effectiveness: %f" % (avg_word_len_and_effectiveness_corr))

video_ids = list(effective_data.keys())
train_n = math.floor(len(video_ids) * .8)
test_n = len(video_ids) - train_n
train, test = train_test_split(video_ids, train_size = train_n, test_size = test_n)
train_topics, train_out, train_ids = normalize_data_five(train, effective_data, VIDEO_SENTIMENTS)
test_topics, test_out, test_ids = normalize_data_five(test, effective_data, VIDEO_SENTIMENTS)

sentiments_SVC = SVC()
sentiments_train_in = [VIDEO_SENTIMENTS[id] for id in train_ids]
sentiments_train_out = [effective_data[id] for id in train_ids]
sentiments_test_in = [VIDEO_SENTIMENTS[id] for id in test_ids]
sentiments_test_out = [effective_data[id] for id in test_ids]
sentiments_SVC.fit(sentiments_train_in, sentiments_train_out)
sentiments_score = sentiments_SVC.score(sentiments_test_in, sentiments_test_out)
print("\nSentiments SVC score: %.4f" % sentiments_score)

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
print(most_common_words_master)

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
    #print(word_count)
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
word_count_SVC = SVC()
word_count_SVC.fit(word_count_train_in, train_out)
word_count_score = word_count_SVC.score(word_count_test_in, test_out)
print("Word count SVC score: %.4f" % word_count_score)

with open("word_sentiments.json", "w+") as word_sentiment_file:
    word_sentiment_file.write(json.dumps(word_sentiments))

with open("video_ocr_stats.json", "w+") as video_ocr_file:
    video_ocr_file.write(json.dumps(VIDEO_OCR_STATS))
