import json
import numpy as np
import scipy.stats

VIDEO_EFFECTIVE_RAW_FILE = "./annotations_videos/video/raw_result/video_Effective_raw.json"

with open(VIDEO_EFFECTIVE_RAW_FILE, 'r') as video_effective_data:
    data = video_effective_data.read()
    effective_data = json.loads(data)

effective_data_stats = dict(effective_data) # Make a copy of the data that holds the standard deviation for each video's ratings

for video_id, ratings in effective_data.items():
    ratings = np.array(ratings).astype(int) # Convert the list from strings to integers

    ratings_std = np.std(ratings) # Calculate the stdev of the annotator ratings
    ratings_std = round(ratings_std, 3)

    ratings_mean = np.mean(ratings)
    ratings_mean = round(ratings_mean, 3)

    ratings_mode = scipy.stats.mstats.mode(ratings).mode[0] # Arbitrarily choose the 0th mode value if there are several
    ratings_cv = ratings_std / ratings_mode # Calculate coefficient of variation (std / mean), ranging from -1 to +1

    effective_data_stats[video_id] = (ratings_std, ratings_mean, ratings_mode, ratings_cv) # Assign this video's data as a tuple

#print(effective_data_stats)
cov_threshold_5 = [x for x in effective_data_stats.values() if abs(x[3]) <= 0.5] # Find the number of videos with coefficient of variation v where |v| <= .5
cov_threshold_3 = [x for x in effective_data_stats.values() if abs(x[3]) <= 0.3] # Find the number of videos with coefficient of variation v where |v| <= .3
print("Num Videos (|cov| <= 0.5): %d" % len(cov_threshold_5))
print("Num Videos (|cov| <= 0.3): %d" % len(cov_threshold_3))
