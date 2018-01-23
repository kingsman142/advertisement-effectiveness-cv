import json
import numpy as np
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
action_RAW_FILE = "./annotations_videos/video/raw_result/video_QA_Action_raw.json"
reason_RAW_FILE = "./annotations_videos/video/raw_result/video_QA_Reason_raw.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(action_RAW_FILE, "r") as action_data_file:
    data = action_data_file.read()
    action_data = json.loads(data)

with open(reason_RAW_FILE, "r") as reason_data_file:
    data = reason_data_file.read()
    reason_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
action_data_stats = dict(action_data) # Make a copy
reason_data_stats = dict(reason_data) # Make a copy
action_reason_combined_stats = {}

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    if video_id in action_data_stats:
        ratings_mean = np.mean(ratings)
        ratings_mean = round(ratings_mean, 3)

        video_action = action_data_stats[video_id]
        video_reason = reason_data_stats[video_id]
        video_action = sum([len(x) for x in video_action])/len(video_action) # Average length of action response
        video_reason = sum([len(x) for x in video_reason])/len(video_reason) # Average length of action reason

        action_data_stats[video_id] = video_action
        reason_data_stats[video_id] = video_reason
        action_reason_combined_stats[video_id] = 1*video_action + 1*video_reason # Apply arbitrary weights to their values
        effective_data_stats[video_id] = ratings_mean

# Calculate the correlation between number of action and average ratings
video_ids = effective_data_stats.keys()
effective_ratings = [effective_data_stats[x] for x in video_ids if x in action_data_stats]
action_responses = [action_data_stats[x] for x in video_ids if x in action_data_stats]
reason_responses = [reason_data_stats[x] for x in video_ids if x in reason_data_stats]
action_reason_responses = [action_reason_combined_stats[x] for x in video_ids if x in action_reason_combined_stats]
correlation_action = pearsonr(effective_ratings, action_responses)[0]
correlation_reason = pearsonr(effective_ratings, reason_responses)[0]
correlation_action_reason = pearsonr(effective_ratings, action_reason_responses)[0]

print("Number of video ids: %d" % (len(video_ids)))
print("Correlation between length of action responses and effectiveness rating: %.3f" % (correlation_action))
print("Correlation between length of reason responses and effectiveness rating: %.3f" % (correlation_reason))
print("Correlation between length of (action, reason) responses and effectiveness rating: %.3f" % (correlation_action_reason))
plt.scatter(effective_ratings, action_responses, s = 5)
plt.scatter(effective_ratings, reason_responses, s = 5)
plt.scatter(effective_ratings, action_reason_responses, s = 5)
plt.show()
