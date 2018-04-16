import glob
import os
import json
import numpy as np
import math
from scipy.stats import pearsonr

VIDEO_EFFECTIVE_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Effective_clean.json"
VIDEO_DURATION_RAW_FILE = "./video_Duration_new_raw.json"
SB_DIR = "./SB-Youtube-5k/"

def calculate_entropy(num_peaks, duration):
    p_0 = (duration - num_peaks)/duration
    p_1 = num_peaks / duration
    if p_0 == 0 or p_1 == 0 or num_peaks > duration:
        return 0.0
    return - p_0*math.log(p_0, 2) - p_1*math.log(p_1, 2)

with open(VIDEO_EFFECTIVE_CLEAN_FILE, "r") as video_effective_data_clean:
    data = video_effective_data_clean.read()
    effective_data_clean = json.loads(data)

with open(VIDEO_DURATION_RAW_FILE, "r") as video_duration_data:
    data = video_duration_data.read()
    duration_data = json.loads(data)

shot_boundary_files = glob.glob(SB_DIR + "*.txt")
shot_boundary_counts = {}
entropy = {}

for filename in shot_boundary_files:
    video_id = filename.split("/")[1].split("\\")[1].split(".")[0]
    if video_id not in duration_data:
        continue
    num_scene_changes = sum(1 for line in open(filename, "r"))
    shot_boundary_counts[video_id] = num_scene_changes
    if video_id in duration_data:
        entropy[video_id] = calculate_entropy(num_scene_changes, duration_data[video_id])

avg_shots_noneff = 0
count_noneff = 0
avg_shots_eff = 0
count_eff = 0

counts = np.zeros(5)
shots = np.zeros(5)

shots_avg = np.zeros(5)
shots_avg_counts = np.zeros(5)

entropy_per_class = np.zeros(5)
entropy_per_class_counts = np.zeros(5)

for video_id, ratings in effective_data_clean.items():
    ratings = int(ratings)
    if ratings < 3 and video_id in shot_boundary_counts.keys():
        avg_shots_noneff += shot_boundary_counts[video_id]
        count_noneff += 1
    elif ratings > 3 and video_id in shot_boundary_counts.keys():
        avg_shots_eff += shot_boundary_counts[video_id]
        count_eff += 1

    if video_id in entropy:
        entropy_per_class[ratings-1] += entropy[video_id]
        entropy_per_class_counts[ratings-1] += 1

    if video_id in shot_boundary_counts:
        counts[ratings-1] += 1
        shots[ratings-1] += shot_boundary_counts[video_id]
        if video_id in duration_data and duration_data[video_id] > 2:
            #print("Duration nums: %d, %d" % (shot_boundary_counts[video_id], duration_data[video_id]))
            shots_avg[ratings-1] += (shot_boundary_counts[video_id] / duration_data[video_id])
            shots_avg_counts[ratings-1] += 1
print("Avg scene changes in effective videos: %.4f" % (avg_shots_eff / count_eff))
print("Avg scene changes in non-effective videos: %.4f" % (avg_shots_noneff / count_noneff))
print(shots)
print(counts)
counts_shots_avg = [shots[i]/counts[i] for i in range(0, 5)] # Average number of shots per video for each class
shots_avg = [shots_avg[i] / shots_avg_counts[i] for i in range(0, 5)] # Average quickness of shots per video for each class
entropy_avg = [entropy_per_class[i] / entropy_per_class_counts[i] for i in range(5)]
print(counts_shots_avg)
print(shots_avg)
print(entropy_avg)
entropy_noneff = (entropy_per_class[0] + entropy_per_class[1])/(entropy_per_class_counts[0] + entropy_per_class_counts[1])
entropy_eff = (entropy_per_class[3] + entropy_per_class[4])/(entropy_per_class_counts[3] + entropy_per_class_counts[4])
print("Average entropy for non-effective classes: %.4f" % entropy_noneff)
print("Average entropy for effective classes: %.4f" % entropy_eff)
entropy_per_video = [entropy[id] for id in entropy.keys()]
effective_ratings_entropy = [int(effective_data_clean[id]) for id in entropy.keys()]
entropy_correlation = pearsonr(effective_ratings_entropy, entropy_per_video)[0]
print("Entropy correlation: %.4f" % entropy_correlation)

shots_per_video = [shot_boundary_counts[id] for id in shot_boundary_counts.keys()]
effective_ratings = [int(effective_data_clean[id]) for id in shot_boundary_counts.keys()]
shots_correlation = pearsonr(effective_ratings, shots_per_video)[0]
print("Shots correlation: %.4f" % shots_correlation)

with open("video_shot_boundary.json", "w+") as shot_boundary_file:
    shot_boundary_file.write(json.dumps(shot_boundary_counts))
with open("video_entropy.json", "w+") as video_entropy_file:
    video_entropy_file.write(json.dumps(entropy))
