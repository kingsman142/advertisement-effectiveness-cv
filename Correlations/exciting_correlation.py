import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

VIDEO_EFFECTIVE_RAW_FILE = "../annotations_videos/video/raw_result/video_Effective_raw.json"
exciting_RAW_FILE = "../annotations_videos/video/cleaned_result/video_Exciting_clean.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(exciting_RAW_FILE, "r") as exciting_data_file:
    data = exciting_data_file.read()
    exciting_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
exciting_data_stats = dict(exciting_data) # Make a copy

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    if video_id in exciting_data_stats:
        ratings_mean = np.mean(ratings)
        ratings_mean = round(ratings_mean, 3)

        effective_data_stats[video_id] = ratings_mean

# Calculate the correlation between number of exciting and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [effective_data_stats[x] for x in video_ids if x in exciting_data_stats]
num_exciting = [exciting_data_stats[x] for x in video_ids if x in exciting_data_stats]
correlation = pearsonr(effective_ratings, num_exciting)[0]

effective_ratings_new = np.array([int(round(rating)) for rating in effective_ratings]).reshape(-1, 1)
num_exciting_new = np.array(num_exciting).reshape(-1, 1)
train_size = int(len(effective_ratings_new)*.7)
effective_train = effective_ratings_new[0:train_size]
exciting_train = num_exciting_new[0:train_size]
effective_test = effective_ratings_new[train_size:]
exciting_test = num_exciting_new[train_size:]

exciting_logregress = LogisticRegression()
exciting_logregress.fit(exciting_train, effective_train)
logress_score = exciting_logregress.score(exciting_test, effective_test)
print("Logistic Regression Score: %.4f" % (logress_score))
exciting_pred = exciting_logregress.predict(exciting_test)
pickle.dump(exciting_logregress, open("exciting_logregress.pkl", "wb+"))
#with open("log_regress_x.txt", )

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between exciting and effectiveness rating: %.3f" % (correlation))
print("Highest exciting rating on a video: %d" % (max(num_exciting)))
plt.scatter(effective_ratings, num_exciting, s = 5)
plt.show()
