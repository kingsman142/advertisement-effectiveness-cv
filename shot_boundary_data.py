import glob
import os
import json
import numpy as np

VIDEO_EFFECTIVE_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Effective_clean.json"
VIDEO_DURATION_RAW_FILE = "video_Duration_raw.json"
SB_DIR = "./SB-Youtube-5k/"

with open(VIDEO_EFFECTIVE_CLEAN_FILE, "r") as video_effective_data_clean:
    data = video_effective_data_clean.read()
    effective_data_clean = json.loads(data)

with open(VIDEO_DURATION_RAW_FILE, "r") as video_duration_data:
    data = video_duration_data.read()
    duration_data = json.loads(data)

shot_boundary_files = glob.glob(SB_DIR + "*.txt")
shot_boundary_counts = {}

for filename in shot_boundary_files:
    video_id = filename.split("/")[2].split(".")[0]
    num_scene_changes = sum(1 for line in open(filename, "r"))
    shot_boundary_counts[video_id] = num_scene_changes

avg_shots_noneff = 0
count_noneff = 0
avg_shots_eff = 0
count_eff = 0

counts = np.zeros(5)
shots = np.zeros(5)

shots_avg = np.zeros(5)
shots_avg_counts = np.zeros(5)

for video_id, ratings in effective_data_clean.items():
    ratings = int(ratings)
    if ratings < 3 and video_id in shot_boundary_counts.keys():
        avg_shots_noneff += shot_boundary_counts[video_id]
        count_noneff += 1
    elif ratings > 3 and video_id in shot_boundary_counts.keys():
        avg_shots_eff += shot_boundary_counts[video_id]
        count_eff += 1

    if video_id in shot_boundary_counts:
        counts[ratings-1] += 1
        shots[ratings-1] += shot_boundary_counts[video_id]
        if video_id in duration_data and duration_data[video_id] > 2:
            print("Duration nums: %d, %d" % (shot_boundary_counts[video_id], duration_data[video_id]))
            shots_avg[ratings-1] += (shot_boundary_counts[video_id] / duration_data[video_id])
            shots_avg_counts[ratings-1] += 1
print("Avg scene changes in effective videos: %.4f" % (avg_shots_eff / count_eff))
print("Avg scene changes in non-effective videos: %.4f" % (avg_shots_noneff / count_noneff))
print(shots)
print(counts)
counts_shots_avg = [shots[i]/counts[i] for i in range(0, 5)] # Average number of shots per video for each class
shots_avg = [shots_avg[i] / shots_avg_counts[i] for i in range(0, 5)] # Average quickness of shots per video for each class
print(counts_shots_avg)
print(shots_avg)

#print(shot_boundary_files)
