import json
import numpy as np
from scipy.stats import pearsonr
import requests
import os

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

with open("./video_ocr_data_clean_words.json", "r") as OCR_FILE:
    ocr_data_clean = json.loads(OCR_FILE.read())

useless_words = ["a", "of", "the", "to", "on", "with", "your", "is", "for", "this", "in", "how", "by", "and", "it", "you", "i", "e", "ry", "an", "not", "s", "its", "what", "are", "get", "be", "o", "at", "have", "as", "no", "do", "am", "me", "de", "my", "am", "wwwesrborg", "ed", "has", "int", "th", "com", "that", "who", "st", "y", "co", "ism", "ma", "sec", "knorr", "presents", "us", "n"]
useless_words += ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

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

i = 0
for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_mean = np.mean(ratings)
    ratings_mean = int(round(ratings_mean, 3))

    video_text_data = ocr_data_clean[video_id]
    num_words = len(video_text_data)
    num_words_per_class[ratings_mean - 1] += num_words
    num_samples_per_class[ratings_mean - 1] += 1
    average_word_length = 0.0

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
            sentiment_per_class[ratings_mean - 1] = np.add(sentiment_per_class[ratings_mean - 1], word_sentiments[word])
            num_meaningful_words_per_class[ratings_mean - 1] += 1
            most_common_words[ratings_mean - 1][word] = 1
    average_word_length = average_word_length / num_words if num_words > 0 else 0
    avg_word_len_per_class[ratings_mean - 1] += average_word_length
    i += 1
    if i % 100 == 0:
        print(i)

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
print("\nWords: %s" % (num_words_per_class))
print("Meaningful words: %s" % (num_meaningful_words_per_class))
print("Average word length: %s" % (avg_word_len_per_class))
num_words_and_effectiveness_corr = pearsonr([1, 2, 3, 4, 5], num_words_per_class)[0]
num_meaningful_words_and_effectiveness_corr = pearsonr([1, 2, 3, 4, 5], num_meaningful_words_per_class)[0]
avg_word_len_and_effectiveness_corr = pearsonr([1, 2, 3, 4, 5], avg_word_len_per_class)[0]
print("Correlation for number of words and effectiveness: %f" % (num_words_and_effectiveness_corr))
print("Correlation for number of meaningful words and effectiveness: %f" % (num_meaningful_words_and_effectiveness_corr))
print("Correlation for avg number of word length and effectiveness: %f" % (avg_word_len_and_effectiveness_corr))

with open("word_sentiments.json", "w+") as word_sentiment_file:
    word_sentiment_file.write(json.dumps(word_sentiments))
