import json
import glob
import os
from scipy import misc
from skimage import io
import numpy as np

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

def calc_video_hue(video_id, batch_num):
    j = 0
    avg_red = 0
    avg_green = 0
    avg_blue = 0
    median_red = 0
    median_green = 0
    median_blue = 0
    #print(video_id)

    base_filename = "./batch" + batch_num + "-frames/" + video_id + "-"
    while os.path.exists(base_filename + str(j) + ".jpg"):
        img = io.imread(base_filename + str(j) + ".jpg")
        #print(img.shape)
        num_pixels = img.shape[0] * img.shape[1]
        avg_red += np.mean(img[0])
        avg_green += np.mean(img[1])
        avg_blue += np.mean(img[2])
        median_red += np.median(img[0])
        median_green += np.median(img[1])
        median_blue += np.median(img[2])
        j += 2
    num_images = j/2
    avg_red /= num_images
    avg_green /= num_images
    avg_blue /= num_images
    median_red /= num_images
    median_green /= num_images
    median_blue /= num_images
    return (avg_red, avg_green, avg_blue, median_red, median_green, median_blue)

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

effective_data_keys = effective_data.keys()

batch1_filenames = glob.glob("./batch1/*.3gp")
batch1_filenames = [filename for filename in batch1_filenames if extract_video_id(filename, "1") in effective_data_keys]
batch1_ids = [extract_video_id(filename, "1") for filename in batch1_filenames]

batch2_filenames = glob.glob("./batch2/*.3gp")
batch2_filenames = [filename for filename in batch2_filenames if extract_video_id(filename, "2") in effective_data_keys]
batch2_ids = [extract_video_id(filename, "2") for filename in batch2_filenames]
batch2_ids = [id for id in batch2_ids if id not in batch1_ids]

average_hue = {}
median_hue = {}

i = 0
for video_id in effective_data.keys():
    if video_id in batch1_ids:
        avg_red, avg_green, avg_blue, median_red, median_green, median_blue = calc_video_hue(video_id, "1")
        #average_intensities[video_id] = (avg_intensity / j)
    elif video_id in batch2_ids:
        avg_red, avg_green, avg_blue, median_red, median_green, median_blue = calc_video_hue(video_id, "2")
    average_hue[video_id] = (avg_red, avg_green, avg_blue)
    median_hue[video_id] = (median_red, median_green, median_blue)
    #print("%s: %.3f" % (video_id, average_intensities[video_id]))
    i += 1
    if i % 100 == 0:
        print(i)

with open("video_average_hue.json", "w+") as avg_hue_file:
    avg_hue_file.write(json.dumps(average_hue))

with open("video_median_hue.json", "w+") as median_hue_file:
    median_hue_file.write(json.dumps(median_hue))
