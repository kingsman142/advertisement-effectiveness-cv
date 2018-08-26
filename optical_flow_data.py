import glob
import os
import json
import numpy as np
from scipy.stats import pearsonr

VIDEO_EFFECTIVE_CLEAN_FILE = "./annotations_videos/video/cleaned_result/video_Effective_clean.json"
OF_DIR = "./Optical-Flow/"

with open(VIDEO_EFFECTIVE_CLEAN_FILE, "r") as video_effective_data_clean:
    data = video_effective_data_clean.read()
    effective_data = json.loads(data)

optical_flow_files = glob.glob(OF_DIR + "*.txt")
optical_flow = {}

for filename in optical_flow_files:
    video_id = filename.split("/")[2].split(".")[0]
    if video_id not in effective_data:
        continue
    of_vectors = open(filename, "r").readlines() # of = 'Optical Flow' vectors
    of_vectors = [float(i) for i in of_vectors]
    optical_flow[video_id] = np.mean(of_vectors)

avg_vector_eff = 0
num_vector_eff = 0
avg_vector_noneff = 0
num_vector_noneff = 0

vectors_per_class = np.zeros(5)
vectors_per_class_counts = np.zeros(5)

for video_id, ratings in effective_data.items():
    ratings = int(ratings)
    if ratings < 3 and video_id in optical_flow.keys():
        avg_vector_noneff += optical_flow[video_id]
        num_vector_noneff += 1
    elif ratings > 3 and video_id in optical_flow.keys():
        avg_vector_eff += optical_flow[video_id]
        num_vector_eff += 1

    vectors_per_class[ratings-1] += optical_flow[video_id]
    vectors_per_class_counts[ratings-1] += 1

avg_vector_eff /= num_vector_eff
avg_vector_noneff /= num_vector_noneff

print("Avg vector in effective videos: %.4f" % avg_vector_eff)
print("Avg vector in non-effective videos: %.4f" % avg_vector_noneff)
print(vectors_per_class)
print(vectors_per_class_counts)
avg_vectors_per_class = [vectors_per_class[i] / vectors_per_class_counts[i] for i in range(5)]
print(avg_vectors_per_class)

vector_per_video = [optical_flow[id] for id in optical_flow.keys()]
effective_ratings = [int(effective_data[id]) for id in optical_flow.keys()]
optical_flow_correlation = pearsonr(effective_ratings, vector_per_video)[0]
print("Shots correlation: %.4f" % optical_flow_correlation)

with open("video_optical_flow.json", "w+") as optical_flow_file:
    optical_flow_file.write(json.dumps(optical_flow))
