import json
import numpy as np
import scipy.stats
from new_look_code.constants import *

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"
VIDEO_EXCITING_RAW_FILE = "./annotations_videos/video/raw_result/video_Exciting_raw.json"
VIDEO_FUNNY_RAW_FILE = "./annotations_videos/video/raw_result/video_Funny_raw.json"
VIDEO_LANGUAGE_RAW_FILE = "./annotations_videos/video/raw_result/video_Language_raw.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

with open(VIDEO_EFFECTIVE_CLEAN_FILE, 'r') as video_effective_clean_data:
    data = video_effective_clean_data.read()
    effective_clean_data = json.loads(data)

with open(VIDEO_EXCITING_RAW_FILE, 'r') as video_exciting_data:
    data = video_exciting_data.read()
    exciting_data = json.loads(data)

with open(VIDEO_FUNNY_RAW_FILE, 'r') as video_funny_data:
    data = video_funny_data.read()
    funny_data = json.loads(data)

with open(VIDEO_LANGUAGE_RAW_FILE, 'r') as video_language_data:
    data = video_language_data.read()
    language_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings
exciting_data_stats = dict(exciting_data)
funny_data_stats = dict(funny_data)
language_data_stats = dict(language_data)

effective_class_bins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

for video_id, ratings in effective_data.items():
    # Effective data
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers
    ratings_std = np.std(ratings) # Calculate the stdev of the annotator ratings
    ratings_std = round(ratings_std, 3)
    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)
    ratings_mode = scipy.stats.mstats.mode(ratings).mode[0] # Arbitrarily choose the 0th mode value if there are several
    ratings_cv = ratings_std / ratings_mode # Calculate coefficient of variation (std / mean), ranging from -1 to +1
    #effective_class_bins[ratings_mode] += 1

    effective_class_bins[int(effective_clean_data[video_id])] += 1

    # Exciting data
    if video_id in exciting_data_stats:
        exciting_ratings = exciting_data_stats[video_id]
        exciting_std = np.std(exciting_ratings)

    # Funny data
    if video_id in funny_data_stats:
        funny_ratings = funny_data_stats[video_id]
        funny_std = np.std(funny_ratings)

    # Language data
    if video_id in language_data_stats:
        language_ratings = np.array(language_data_stats[video_id]).astype(int)
        language_std = np.std(language_ratings)

    effective_data_stats[video_id] = (ratings_std, ratings_mean, ratings_mode, ratings_cv, video_id) # Assign this video's data as a tuple
    exciting_data_stats[video_id] = exciting_std
    funny_data_stats[video_id] = funny_std
    language_data_stats[video_id] = language_std

#print(effective_data_stats)
cov_threshold_5 = [x for x in effective_data_stats.values() if abs(x[3]) <= 0.5] # Find the number of videos with coefficient of variation v where |v| <= .5
cov_threshold_4 = [x for x in effective_data_stats.values() if abs(x[3]) <= 0.4] # Find the number of videos with coefficient of variation v where |v| <= .4
cov_threshold_3 = [x for x in effective_data_stats.values() if abs(x[3]) <= 0.3] # Find the number of videos with coefficient of variation v where |v| <= .3
avg_exciting_std = sum(exciting_data_stats.values())/len(exciting_data_stats) # Average standard deviation
avg_funny_std = sum(funny_data_stats.values())/len(funny_data_stats) # Average standard deviation
avg_language_std = sum(language_data_stats.values())/len(language_data_stats) # Average standard deviation
print("Num Videos (|cov| <= 0.5): %d" % len(cov_threshold_5))
print("Num Videos (|cov| <= 0.4): %d" % len(cov_threshold_4))
print("Num Videos (|cov| <= 0.3): %d" % len(cov_threshold_3))
print("Average exciting rating std: %.4f" % (avg_exciting_std))
print("Average funny rating std: %.4f" % (avg_funny_std))
print("Average language rating std: %.4f" % (avg_language_std))
print("Effective class bins: %s" % (effective_class_bins))

# Earlier, we computed all the videos with a certain threshold on their coefficient of variation.
# Let's prune out videos that don't pass our threshold and use the ones below as a 'useful' ones.
cov_threshold_4.sort(key = lambda tup: -tup[1])
useful_eff_ratings_ids = [x[4] for x in cov_threshold_4]
useful_eff_ratings = [x[1] for x in cov_threshold_4]
useful_videos_concat = ["www.youtube.com/watch?v="+useful_eff_ratings_ids[i]+" "+str(useful_eff_ratings[i]) for i in range(0, len(useful_eff_ratings))]
with open("useful-videos-effectiveness-ratings.txt", 'w') as useful_videos:
    for video_rating in useful_videos_concat:
        useful_videos.write(video_rating+'\n')
